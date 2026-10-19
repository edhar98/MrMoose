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

from . import mm_utilities as ut
import numpy as np
import emcee
from astropy.cosmology import WMAP9 as cosmos
from astropy import constants

##############################################
# some function to use
# to be moved in the utilities
def calcul_luminosity(x, y, lim, redshift):
    # TODO make as Theresa - analysing file and move this there
    if len(lim) != 2:
        stop
    # luminosity distance
    DL = cosmos.luminosity_distance(redshift)
    dist_factor = 4. * np.pi * DL.value * DL.value * 1e4 * 1e12 * constants.pc.value * constants.pc.value

    # index corresponding to the limits of the integration -
    int_index = np.argwhere((x > (lim[0] / (1 + redshift)) * (x < (lim[1] / (1. + redshift)))))
    int_index = np.squeeze(int_index)
    y = y * (dist_factor * (1 + redshift))

    # integration
    lum = np.trapz(y[int_index], x=x[int_index])
    return lum
##############################################


def _chain_diagnostics(sampler, samples, lnprob, fit_struct):
    """Reject failed inference and describe, rather than assume, mixing."""
    if samples.ndim != 3 or lnprob.shape != samples.shape[:2] or samples.shape[0] < 2:
        raise ValueError('Invalid or empty post-burn emcee chain')
    if not np.all(np.isfinite(samples)):
        raise ValueError('Sampler chain contains non-finite parameter values')
    if not np.all(np.isfinite(lnprob)):
        raise ValueError('Sampler has non-finite post-burn log probabilities; rerun the fit')
    acceptance = np.asarray(sampler.acceptance_fraction, dtype=float)
    if acceptance.shape != (samples.shape[1],) or not np.all(np.isfinite(acceptance)):
        raise ValueError('Invalid sampler acceptance fractions')
    if not np.any(acceptance > 0):
        raise ValueError('Sampler accepted no proposals; initialization is not a fit')
    post_burn_moves = np.any(np.diff(samples, axis=0) != 0, axis=2)
    if not np.any(post_burn_moves):
        raise ValueError('Post-burn chain never moved; initialization is not a fit')
    minimum_tau = float(fit_struct.get('convergence_min_tau', 50))
    try:
        tau = emcee.autocorr.integrated_time(samples, tol=minimum_tau)
        length_ok = bool(np.all(np.isfinite(tau)) and np.all(tau > 0))
    except emcee.autocorr.AutocorrError as exc:
        tau = exc.tau
        length_ok = False
    except (ValueError, FloatingPointError):
        tau = np.full(samples.shape[-1], np.nan)
        length_ok = False
    pooled = samples.reshape(-1, samples.shape[-1])
    low, high = np.percentile(pooled, [16, 84], axis=0)
    split = samples.shape[0] // 2
    if split:
        first = np.percentile(samples[:split].reshape(-1, samples.shape[-1]), [16, 50, 84], axis=0)
        second = np.percentile(samples[split:].reshape(-1, samples.shape[-1]), [16, 50, 84], axis=0)
        interval_shift = np.divide(np.abs(first - second), high - low,
                                   out=np.full_like(first, np.inf), where=high > low)
    else:
        interval_shift = np.full((3, samples.shape[-1]), np.inf)
    shift = interval_shift[1]
    shift_limit = float(fit_struct.get('convergence_split_tolerance', .1))
    split_ok = bool(np.all(shift <= shift_limit))
    zero_walkers = int(np.count_nonzero(~np.any(post_burn_moves, axis=0)))
    parameter_length_ok = np.isfinite(tau) & (tau > 0) & (samples.shape[0] >= minimum_tau * tau)
    parameter_converged = parameter_length_ok & (shift <= shift_limit) & (zero_walkers == 0)
    parameter_interval_stable = np.all(interval_shift <= shift_limit, axis=0)
    diagnostics = {
        'valid': True, 'converged': bool(length_ok and split_ok and not zero_walkers),
        'retained_steps': int(samples.shape[0]), 'walkers': int(samples.shape[1]),
        'burn_steps': int(fit_struct['nsteps_cut']),
        'finite_log_probability_fraction': 1.0,
        'acceptance_min': float(acceptance.min()),
        'acceptance_mean': float(acceptance.mean()),
        'acceptance_max': float(acceptance.max()), 'zero_acceptance_walkers': zero_walkers,
        'post_burn_move_fraction': float(post_burn_moves.mean()),
        'autocorrelation_steps': [float(t) if np.isfinite(t) else None for t in tau],
        'minimum_autocorrelation_times': minimum_tau, 'meets_length_check': length_ok,
        'split_median_shift_fraction': [float(s) if np.isfinite(s) else None for s in shift],
        'split_median_tolerance': shift_limit, 'meets_split_check': split_ok,
        # Retain the legacy all-parameter flag; callers can assess reported
        # quantities separately. Passing checks does not prove convergence.
        'parameter_converged': parameter_converged.tolist(),
        'interval_split_fraction': [[float(s) if np.isfinite(s) else None for s in row]
                                    for row in interval_shift.T],
        'parameter_interval_stable': parameter_interval_stable.tolist(),
        'parameter_sampling_verified': (parameter_converged & parameter_interval_stable).tolist(),
        'note': 'Length and split checks do not prove exploration of disconnected modes.',
    }
    return diagnostics


def find_stats_fit(sampler, models, fit_struct, data_struct):
    """Summarize the complete finite post-burn ensemble in parameter space."""
    ndim = len(ut.flatten_model_keyword(models, 'param'))
    samples = np.asarray(sampler.get_chain(discard=fit_struct['nsteps_cut']))
    lnprob = np.asarray(sampler.get_log_prob(discard=fit_struct['nsteps_cut']))
    fit_struct['convergence'] = _chain_diagnostics(sampler, samples, lnprob, fit_struct)
    if samples.shape[-1] != ndim:
        raise ValueError('Sampler dimensionality differs from model parameters')
    best_index = np.unravel_index(lnprob.argmax(), lnprob.shape)
    best_param = samples[best_index]
    # Pool steps AND walkers before calculating marginal quantiles. Slicing
    # parameter columns is then also correct for multiple model components.
    perc_param = np.percentile(samples.reshape(-1, ndim), fit_struct['percentiles'], axis=0)
    lb = 0
    for model in models:
        ub = lb + model['dim']
        model['bestfit'] = best_param[lb:ub]
        model['perc'] = [row[lb:ub] for row in perc_param]
        lb = ub
    fit_struct['best_lnL'] = float(lnprob[best_index])
    initial_maximum = fit_struct.get('radio_initialization', {}).get('lnL')
    if initial_maximum is not None:
        fit_struct['convergence']['optimizer_minus_sample_max_lnL'] = float(initial_maximum - fit_struct['best_lnL'])
    fit_struct['likelihood_convention'] = 'mrmoose_minus_chi2'
    fit_struct['likelihood_note'] = 'lnL=-Q, with Q=chi2_detection-2*sum(log Gaussian integrals); historical MrMoose scale retained.'
    ndata = sum(len(elem['filter']) for elem in data_struct)
    fit_struct['AICc'] = (
        2*ndim - 2*fit_struct['best_lnL'] + 2*ndim*(ndim+1)/(ndata-ndim-1)
        if ndata > ndim + 1 else None)
    if not fit_struct['convergence']['converged']:
        print('Sampler is finite, but convergence checks are not yet satisfied.')

def find_stats_fit_ultranest(sampler, models, fit_struct, data_struct):
    """to finish"""
    samples=np.array(sampler.results['weighted_samples']['points'])
    weights = np.array(sampler.results['weighted_samples']['weights'])
    cumsumweights = np.cumsum(weights)
    mask = cumsumweights > 1e-4                   
    
    best_param = sampler.results['maximum_likelihood']['point']
    perc_param = np.percentile(samples[mask,:], fit_struct['percentiles'], axis=0)
    # add to the model structure
    lb = 0
    for i in range(len(models)):
        ub = lb + models[i]['dim']
        models[i]['bestfit'] = best_param[lb:ub]
        models[i]['perc'] = [perc_param[j][lb:ub] for j in range(len(perc_param))]
        lb += models[i]['dim']
    # the float() is a trick to force saving the value in a variable and not only create a pointer
    fit_struct['best_lnL'] = float(sampler.results['maximum_likelihood']['logl'])
