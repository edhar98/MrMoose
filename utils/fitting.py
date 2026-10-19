"""
MrMoose: A multi-wavelength, multi-resolution, multi-sources fitting code.
Allow to simultaneously fit multiple models in order to interpret the spectral
energy distribution of an astrophysical source from simple to more complex case, 
making use of a bayesian framework.

Copyright (C) 2017 Guillaume Drouart, Theresa Falkendal

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from . models import *
from . import mm_utilities as ut
import scipy.special
from scipy.optimize import differential_evolution, minimize
import pathos.multiprocessing as mp
from tqdm import tqdm
import emcee,ultranest
import pickle as pickle
import os.path
import logging
import dill
import tempfile
import h5py

#create a logger
logger = logging.getLogger('mylogger')
#set logging level
logger.setLevel(logging.DEBUG)


class _RadioHDFBackend(emcee.backends.HDFBackend):
    """Cache fixed metadata for fresh radio runs; retain every HDF write.

    The radio likelihood returns no blobs. Its shape is fixed by reset, so
    checking each proposal need not reopen the file for these two properties.
    Ordinary HDFBackend readers can inspect and resume the resulting file.
    """

    def reset(self, nwalkers, ndim):
        super().reset(nwalkers, ndim)
        self._radio_shape = (nwalkers, ndim)

    @property
    def shape(self):
        return self._radio_shape if hasattr(self, '_radio_shape') else super().shape

    def has_blobs(self):
        return False if hasattr(self, '_radio_shape') else super().has_blobs()

    def _check_blobs(self, blobs):
        if blobs is not None:
            raise ValueError('Fast radio sampler does not support blobs')


def _compact_sampler_file(filename):
    """Atomically remove HDF free-space fragmentation after a completed fit.

    HDFBackend opens the filename for each access, so an existing sampler
    remains usable after replacement. This is never called for reused chains.
    """
    before = os.stat(filename)
    directory = os.path.dirname(os.path.abspath(filename))
    descriptor, temporary = tempfile.mkstemp(prefix='.mrmoose-compact-', suffix='.h5', dir=directory)
    os.close(descriptor)
    try:
        with h5py.File(filename, 'r') as original, h5py.File(temporary, 'w') as compact:
            for key, value in original.attrs.items():
                compact.attrs[key] = value
            for key in original:
                original.copy(key, compact)
        old = emcee.backends.HDFBackend(filename, read_only=True)
        new = emcee.backends.HDFBackend(temporary, read_only=True)
        if old.iteration != new.iteration or old.shape != new.shape:
            raise IOError('Compacted sampler metadata does not match the completed chain')
        if not np.array_equal(old.accepted, new.accepted):
            raise IOError('Compacted sampler lost acceptance counters')
        old_last, new_last = old.get_last_sample(), new.get_last_sample()
        if (not np.array_equal(old_last.coords, new_last.coords)
                or not np.array_equal(old_last.log_prob, new_last.log_prob)):
            raise IOError('Compacted sampler last state does not match')
        os.chmod(temporary, before.st_mode & 0o777)
        after_bytes = os.stat(temporary).st_size
        os.replace(temporary, filename)
        return {'bytes_before': before.st_size, 'bytes_after': after_bytes}
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _RadioLikelihood:
    """Exact, vectorized filter integration for one opt-in radio component.

    Parameters and stored chains remain in the original model coordinates.
    The precomputed quadrature uses exactly the generic 2000-point grid and
    interpolation rule; no monochromatic or analytic-band approximation is
    introduced. Unsupported model combinations keep the generic code path.
    """

    def __init__(self, fit_struct, data, filters, models):
        if len(models) != 1 or models[0]['func'] not in ('sync_law', 'radio_broken_power_law'):
            raise ValueError('Fast radio likelihood requires one supported radio model')
        self.model_name = models[0]['func']
        self.redshift = float(fit_struct['redshift'][0])
        if self.redshift < 0:
            raise ValueError('Fast radio likelihood requires a fixed nonnegative redshift')
        self.minimum = np.asarray(models[0]['min'], dtype=float)
        self.maximum = np.asarray(models[0]['max'], dtype=float)
        self.groups = []
        for points, bandpasses in zip(data, filters):
            for component in points['component_number']:
                components = [int(s) for s in str(component).split(',') if s]
                if components != [0]:
                    raise ValueError('Fast radio likelihood requires component zero only')
            frequencies = np.asarray(points['lambda0'], dtype=float)
            grid = 10**np.linspace(np.log10(frequencies.min()*.01),
                                   np.log10(frequencies.max()*100), 2000)
            quadrature = np.empty_like(grid)
            quadrature[0] = (grid[1]-grid[0])/2
            quadrature[-1] = (grid[-1]-grid[-2])/2
            quadrature[1:-1] = (grid[2:]-grid[:-2])/2
            weights = []
            for freq, transmission in zip(bandpasses['wav'], bandpasses['trans']):
                response = np.interp(grid, freq, transmission, left=0., right=0.)
                weight = quadrature * grid * response
                if not np.isfinite(weight.sum()) or weight.sum() <= 0:
                    raise ValueError('Radio filter has no positive response within the model grid')
                weights.append(weight/weight.sum())
            weights = np.asarray(weights)
            used = np.any(weights != 0, axis=0)
            flux = np.asarray(points['flux'], dtype=float)
            error = np.asarray(points['flux_error'], dtype=float)
            if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(error)) or np.any(error <= 0):
                raise ValueError('Radio fluxes and positive errors must be finite')
            kind = np.asarray(points['det_type'])
            if not np.all((kind == 'd') | (kind == 'u')):
                raise ValueError('Radio data must contain detections or upper limits')
            # Keep cgs sigma in the model-independent Gaussian integral constant.
            self.groups.append((np.log10(grid[used]), weights[:, used],
                                flux*1e26, error*1e26, kind == 'd',
                                .5*np.log(2*np.pi)+np.log(error)))

    def __call__(self, parameters):
        values = np.asarray(parameters, dtype=float)
        scalar = values.ndim == 1
        values = np.atleast_2d(values)
        valid = np.all(np.isfinite(values) & (values > self.minimum) & (values < self.maximum), axis=1)
        result = np.full(len(values), -np.inf)
        if np.any(valid):
            theta = values[valid]
            objective = np.zeros(len(theta))
            log_one_plus_z = np.log10(1+self.redshift)
            with np.errstate(over='ignore', invalid='ignore'):
                for logfreq, weights, flux, error, detected, constant in self.groups:
                    if self.model_name == 'sync_law':
                        exponent = (theta[:, 0, None] + 26 + log_one_plus_z
                                    + theta[:, 1, None]*(logfreq-log_one_plus_z))
                    else:
                        delta = logfreq[None, :] + log_one_plus_z - theta[:, 1, None]
                        slope = np.where(delta <= 0, theta[:, 2, None], theta[:, 3, None])
                        exponent = theta[:, 0, None] + 26 + delta*slope
                    model = np.einsum('ij,kj->ik', 10**exponent, weights, optimize=False)
                    objective += np.sum(((flux[detected]-model[:, detected])/error[detected])**2, axis=1)
                    z = (flux[~detected]-model[:, ~detected])/error[~detected]
                    objective -= 2*np.sum(constant[~detected]+scipy.special.log_ndtr(z), axis=1)
            result[valid] = np.where(np.isfinite(objective), -objective, -np.inf)
        return float(result[0]) if scalar else result


def _radio_initial_positions(likelihood, fit_struct):
    """Find a deterministic radio maximum, then seed a finite dispersed ball."""
    seed = fit_struct.get('random_seed')
    random = np.random.default_rng(seed)
    lower, upper = likelihood.minimum, likelihood.maximum
    log_z = np.log10(1+likelihood.redshift)
    if likelihood.model_name == 'sync_law':
        # A pivot at observed 1 GHz removes the large raw-Hz normalization/slope
        # correlation during optimization; the likelihood enforces native priors.
        bounds = [(-12., 15.), (lower[1], upper[1])]

        def native(value):
            value = np.asarray(value).copy()
            value[..., 0] -= 26 + log_z + value[..., 1]*(9-log_z)
            return value
    else:
        bounds = [(lower[0]+26, upper[0]+26),
                  (lower[1]-9-log_z, upper[1]-9-log_z),
                  (lower[2], upper[2]), (lower[3], upper[3])]

        def native(value):
            value = np.asarray(value).copy()
            value[..., 0] -= 26
            value[..., 1] += 9 + log_z
            return value

    def objective(value):
        return -likelihood(native(value))

    optimum = differential_evolution(
        lambda values: -likelihood(native(values.T)), bounds,
        seed=seed, popsize=12, maxiter=int(fit_struct.get('radio_optimizer_iterations', 350)),
        tol=1e-8, polish=False, vectorized=True, updating='deferred')
    polished = minimize(objective, optimum.x, method='Nelder-Mead', bounds=bounds,
                        options={'maxiter': 2500, 'xatol': 1e-9, 'fatol': 1e-8})
    if polished.fun < optimum.fun:
        optimum = polished
    center = native(optimum.x)
    if not np.isfinite(likelihood(center)):
        raise ValueError('Radio optimizer could not find a finite initialization')
    fit_struct['radio_initialization'] = {
        'method': 'differential_evolution_then_simplex', 'seed': seed,
        'optimizer_success': bool(optimum.success),
        'parameters': center.tolist(), 'lnL': float(likelihood(center)),
        'note': 'Initialization does not establish posterior convergence or exclude other modes.',
    }
    nwalkers = int(fit_struct['nwalkers'])
    positions = []
    for walker in range(nwalkers):
        # Include wider starting walkers to reduce reliance on one tiny ball.
        spread = .02 if walker % 5 == 0 else .001
        for attempt in range(10000):
            proposal = center + spread*(upper-lower)*random.normal(size=len(center))
            if np.isfinite(likelihood(proposal)):
                positions.append(proposal)
                break
        else:
            raise ValueError('Could not initialize finite radio walkers within native priors')
    return np.asarray(positions)

def calc_chi2(obs_flux, error, model):
    """
    Calculated the classical chi2 likelihood value when all the data
    points are detections

    :param obs_flux: The flux given as an array containing the observed data in different bands
    :param error: Corresponding flux uncertainty, array of the same length as data
    :param model: The flux from the model at the same wavelengths
    :return: The classical chi2 value for observations without upperlimits.
    """

    return np.sum(((obs_flux-model)/error)**2)


def calc_chi2_mod(upper_lim, rms, model):
    """
    Calculates the modified chi2 likelihood from data containing upper limits
    Formula from Sawicki 2012

    :param upper_lim: Flux of the upper limit
    :param rms: The 1 sigma noise level
    :param model: Flux from the model
    :return: The modified chi2
    """
    upper_lim = np.asarray(upper_lim)
    rms = np.asarray(rms)
    model = np.asarray(model)

    if np.any(rms <= 0):
        raise ValueError("Upper-limit RMS values must be strictly positive")

    # Sawicki (2012): integrate each Gaussian likelihood from -infinity
    # to the reported upper limit. log_ndtr evaluates log(Phi(z))
    # robustly even when the model lies many sigma above the limit.
    z = (upper_lim - model) / rms
    log_integral = 0.5 * np.log(2. * np.pi) + np.log(rms) + scipy.special.log_ndtr(z)
    return -2. * np.sum(log_integral)



def lnlike(theta, fit_struct, data, filters, param, detection_mask):
    """
    Calculated the chi2 likelihood, by calculates the model data from the initial/current guess and summing
    chi2 of each different arrangement to get the total chi2.

    :param theta: This is are the free parameters that will be fitted in the emcee process, the MCMC.
    :param fit_struct: the structure containing the information about fit parameters
    :param data: This is the observed data that is compared with the model
    :param filters: same structure than data, but to reference the filters to be used
    :param param: This contains the name of the function to be called on the different combinations of models
    :param detection_mask: List of true/false to indicate with observation is a detection or upper limit
    :return: Returns the total chi2 of the current guess.
    """
    # feeding the theta into the current parameter values
    theta = theta.flatten()
    ub = 0
    for i in range(len(param)):
        lb = ub
        ub = param[i]['dim'] + ub
        param[i]['current'] = theta[lb:ub]

    model_data = []
    number_of_component = []
    detections = []
    model_detections = []
    upper_limits = []
    model_upper_limits = []

    for i in range(len(data)):
        # make a list of list containing the number index of the model functions to use,
        # in the case of several models for
        # the same data point, the list element will be a list of numbers
        number_of_component.append(list(map(int, str.split(data[i]['component_number'][0], ','))))

        # call on the model functions via globals()
        # selects the right model (sync_lax or BB_law) via the name from parameter['func']
        # it then also sends in the data, the frequencies/wavelength and the initial or current guess
        # for the right model parameter

        min_tmp = np.log10(min(data[i]['lambda0'])*0.01)
        max_tmp = np.log10(max(data[i]['lambda0'])*100.)
        xscale = 10**np.linspace(min_tmp, max_tmp, 2000)
        
        # TODO: this is not the index i for redshift but number_of_component[i][j] to pass
        # need to figure out a way of minimising conditions and loop
        # for j
        #    if z >0
        #        global+z
        #    else
        #        global
        temp = np.zeros(2000)

        for j in range(len(number_of_component[i])):
            if fit_struct['redshift'][number_of_component[i][j]] >= 0:
                temp2 = globals()[param[number_of_component[i][j]]['func']]\
                    (xscale, param[number_of_component[i][j]]['current'], fit_struct['redshift'][number_of_component[i][j]])
            else:
                temp2 = globals()[param[number_of_component[i][j]]['func']]\
                    (xscale, param[number_of_component[i][j]]['current'])
            temp += temp2

        # Making the sum of models to go through filters
        ndim2=len(filters[i]['wav'])
        temp_mod_filter=np.array(list(map(ut.integrate_filter,[xscale,]*ndim2,[temp,]*ndim2,filters[i]['wav'],filters[i]['trans'])))

        model_data.append(temp_mod_filter)

        detections.append(data[i][detection_mask[i]])
        model_detections.append(model_data[i][np.array(detection_mask[i])])

        upper_limits.append(data[i][~detection_mask[i]])
        model_upper_limits.append(model_data[i][~np.array(detection_mask[i])])

    chi2_classic = [calc_chi2(detections[i]['flux'],detections[i]['flux_error'],model_detections[i]) for i in range(len(detections))]
    chi2_modified = [calc_chi2_mod(upper_limits[i]['flux'],upper_limits[i]['flux_error'],model_upper_limits[i]) for i in range(len(detections))]

    return -(sum(chi2_classic)+sum(chi2_modified))


def lnprior(theta, models):
    """
    This function checks to see it the current guess it within the priors. If it is not then -inf will be returned.

    :param theta: The free parameters that are to be fitted
    :param models: Contains the min and max values of the prior
    :return: Zero or -inf, if the guess are within the allowed parameter space or not
    """

    logger.debug(theta.shape)
    # defining priors, here only uniform possible for now
    #type_func = ut.flatten_model_keyword(models, 'type')
    tmin = ut.flatten_model_keyword(models, 'min')
    tmax = ut.flatten_model_keyword(models, 'max')

    # trick to get the dimension of table right...
    theta = theta.flatten()
    # print(theta.shape)

    list_prior = [0.0 if tmin[i] < theta[i] < tmax[i] else -np.inf for i in range(len(tmin))]

    # list_prior = 0
    # for i in range(len(tmin)):
    #     if type_func[i] == 'uniform':
    #         list_prior += 0.0 if tmin[i] < theta[i] < tmax[i] else -np.inf
    #     else if type_func[i] == 'normal':
    #         list_prior += np.log10(1/(tmax[i]*(2.*np.pi)**0.5)-0.5*(param-tmin[i])**2/(tmin[i]**2)
    #     else:
    #         print "pb"
    #         sys.exit()
    
    return sum(list_prior)

def lnprior_un(theta,models):
    """
    This function check if current guess is in prior range. Ultranest implementation.
    Priors transformation in the 0 - 1 range. Only support uniform, same input file.

    :param theta: The free parameters that are to be fitted
    :param models: Contains the min and max values of the prior
    :return: Zero or -inf, if the guess are within the allowed parameter space or not
    """
    tmin = ut.flatten_model_keyword(models, 'min')
    tmax = ut.flatten_model_keyword(models, 'max')
    theta = theta.flatten()
    list_prior=np.zeros(tmin.size)
    for i,elem in enumerate(theta):
        list_prior[i] = elem*np.abs(tmax[i]-tmin[i])+tmin[i]
    return list_prior

def lnprob(theta, fit_struct, data, filters, models, detection_mask):
    # TODO implement non-uniform prior (see lnprior function)
    """
    This is the function that is called in the emcee MCMC process,it checks so that the current
    guess (theta) is within the prior and then calculated the chi2 of the current fit via lnlike.

    :param theta: The free parameters that are fitted
    :param data: Contains the observed data
    :param filters: Contains the filter used for observations
    :param models: Contains the name of which model to associate each data point
    :param detection_mask: list of true/false to decided if the data point is an detection or upperlimit
    :param redshift: redshift of the object
    :return: The maximum at posteriori estimation
    """
    lp = lnprior(theta, models)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, fit_struct, data, filters, models, detection_mask)

def decorator_like(fit_struct, data_struct, filter_struct, model_struct, detection_mask):
    """
    Decorator for the likelihood function in Ultranest
    """
    def outer(fnc):
        def inner(*args,**kwargs):
            return fnc(*args, fit_struct, data_struct, filter_struct, model_struct, detection_mask)
        return inner
    return outer

def decorator_prior(model_struct):
    """
    Decorator for the prior function in Ultranest
    """
    def outer(fnc):
        def inner(*args,**kwargs):
            return fnc(*args, model_struct)
        return inner
    return outer

def fit_source(fit_struct, data_struct, filter_struct, model_struct, Parallel=0):
    # detection mask to differenciate
    # the upper limits from detections
    # necessary to feed the chi2 calculation
    detection_mask = []
    for i in range(len(data_struct)):
        detection_mask.append(data_struct[i]['det_type'] == 'd')

    # flatten the parameters structure for processing
    ndim = len(ut.flatten_model_keyword(model_struct, 'param'))
    flat_param = ut.flatten_model_keyword(model_struct, 'current')
    flat_min = ut.flatten_model_keyword(model_struct, 'min')
    flat_max = ut.flatten_model_keyword(model_struct, 'max')
    coef_param = np.array([(flat_max - flat_min) / 2.]).flatten()

    # # initiate the walkers "ball" - TODO update with emcee function
    # if os.path.isfile(fit_struct['sampler_file']):
    #     print("file found at ",fit_struct['sampler_file'])
    #     #with open(fit_struct['sampler_file'],"rb") as f:
    #     #    chain = pickle.load(f)
    #     #last_index = chain.chain[0].nonzero()[0][-1]
    #     #print last_index
    #     #pos = [chain.chain[i][last_index] for i in range(fit_struct['nwalkers'])]
    # else:
    radio_likelihood = None
    radio_supported = (len(model_struct) == 1 and model_struct[0]['func'] in
                       ('sync_law', 'radio_broken_power_law')
                       and fit_struct['redshift'][0] >= 0)
    if fit_struct.get('radio_fast', False) and radio_supported:
        radio_likelihood = _RadioLikelihood(fit_struct, data_struct, filter_struct, model_struct)
    fit_struct['radio_fast_used'] = radio_likelihood is not None
    if fit_struct.get('radio_initialize', False) and radio_likelihood is not None:
        pos = _radio_initial_positions(radio_likelihood, fit_struct)
    else:
        rng = np.random.default_rng(fit_struct.get('random_seed'))
        pos = [flat_param + 1e-4 * coef_param * rng.normal(size=ndim)
               for i in range(fit_struct['nwalkers'])]
    if fit_struct['fit_method'] == 'emcee':
        initial_probability = (radio_likelihood(pos) if radio_likelihood is not None else
                               np.array([lnprob(np.asarray(p), fit_struct, data_struct,
                                                filter_struct, model_struct, detection_mask) for p in pos]))
        if not np.all(np.isfinite(initial_probability)):
            raise ValueError('Initial walkers have non-finite log probabilities; fix inputs or initialization')
    print("initialise walker positions")
    last_index = 0

    print()
    print('HMC attempt for ' + fit_struct['source'])
    # single processor
#    if os.path.isfile(fit_struct['sampler_file']):
#        with open(fit_struct['sampler_file'],"rb") as f:
#            sampler = pickle.load(f)
#    else:

    if fit_struct['fit_method']=='emcee': 
    # create the backend for emcee
        filename = fit_struct['sampler_file']
        backend_class = _RadioHDFBackend if radio_likelihood is not None else emcee.backends.HDFBackend
        backend = backend_class(filename)
        # if restart:
        #     print('continuing last run')
        # else:
        #     backend.reset(fit_struct['nwalkers'],ndim)
        backend.reset(fit_struct['nwalkers'],ndim)
    
        if radio_likelihood is not None:
            sampler = emcee.EnsembleSampler(fit_struct['nwalkers'], ndim,
                                            radio_likelihood, vectorize=True, backend=backend)
        elif Parallel == 0:
            sampler = emcee.EnsembleSampler(fit_struct['nwalkers'], ndim, lnprob,
                                            args=(fit_struct, data_struct, filter_struct, model_struct, detection_mask), backend=backend)
        else:
            # multi-processing (pool created via pathos, allow to pickle the sampler)
            tmp_pool = mp.ProcessingPool(Parallel)
            sampler = emcee.EnsembleSampler(fit_struct['nwalkers'], ndim, lnprob,
                                            args=(fit_struct, data_struct, filter_struct, model_struct, detection_mask),
                                            pool=tmp_pool, backend=backend)
        if fit_struct.get('random_seed') is not None:
            sampler.random_state = np.random.RandomState(int(fit_struct['random_seed'])).get_state()
            
    # progress bar (work for multiprocess or single process)
#    for sample in sampler.sample(pos,iterations=fit_source['nsteps'],progress=True):

        with tqdm(total=fit_struct['nsteps']) as pbar:
            for i, result in enumerate(sampler.sample(pos, iterations=fit_struct['nsteps'])):
                pbar.update()
            print('HMC done!')
        if fit_struct.get('compact_sampler', radio_likelihood is not None):
            fit_struct['sampler_storage'] = _compact_sampler_file(filename)

    #pbar = tqdm(total=fit_struct['nsteps'],initial=last_index)
    #print fit_struct['nsteps']-last_index

#    sampler.run_mcmc(None,fit_struct['nsteps'])

#    for i,_ in enumerate(sampler.sample(None, fit_struct['nsteps'])):
#        pbar.update()
#        if i % 10 == 0:
#            pass
#            with open(fit_struct['sampler_file'], 'w') as output_savefile:
#                pickle.dump(sampler, output_savefile, pickle.HIGHEST_PROTOCOL)
#                print i,' sampler saved!'
    #print('HMC done!')

    # TODO: transformation of chains
    # save the modified sampler (allows to save pools as well - pathos library allows to serialise pools)
    #with open(fit_struct['sampler_file'], 'ab') as output_savefile:
    #    pickle.dump(sampler, output_savefile, pickle.HIGHEST_PROTOCOL)
    #    print 'sampler saved!'

        return sampler
    elif fit_struct['fit_method']=='ultranest':
        param_names=list(ut.flatten_model_keyword(model_struct,'param'))

        # create the inputs functions from decorators
        @decorator_like(fit_struct, data_struct, filter_struct, model_struct, detection_mask)
        def like_func(theta,*args):
            return lnlike(theta,*args)
        @decorator_prior(model_struct)
        def prior_func(theta,*args):
            return lnprior_un(theta,*args)

        # sampler_un = ultranest.ReactiveNestedSampler(param_names, like_func, prior_func, resume=True, log_dir='outputs/tmp_ultra')
        sampler_un = ultranest.ReactiveNestedSampler(param_names, like_func, prior_func)
        sampler_un.run()
        print('Ultranest done!')
        with open(fit_struct['sampler_file'],'wb') as file:
            dill.dump(sampler_un,file)
        return sampler_un
    else:
        print('error, wrong fit_method provided')
        return 0
