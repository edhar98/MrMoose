import unittest

import numpy as np

from utils.fitting import calc_chi2_mod


class UpperLimitLikelihoodTests(unittest.TestCase):
    def test_reported_upper_limit_changes_likelihood(self):
        rms = np.array([1.0])
        model = np.array([3.0])

        tighter = calc_chi2_mod(np.array([2.0]), rms, model)
        looser = calc_chi2_mod(np.array([4.0]), rms, model)

        self.assertGreater(tighter, looser)

    def test_penalty_increases_monotonically_above_upper_limit(self):
        upper_limit = np.array([3.0])
        rms = np.array([1.0])
        models = [0.0, 3.0, 6.0, 20.0]

        penalties = [
            calc_chi2_mod(upper_limit, rms, np.array([model]))
            for model in models
        ]

        self.assertEqual(penalties, sorted(penalties))
        self.assertTrue(np.all(np.isfinite(penalties)))

    def test_nonpositive_rms_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            calc_chi2_mod(np.array([3.0]), np.array([0.0]), np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
