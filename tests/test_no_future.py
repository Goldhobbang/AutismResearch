import unittest
import numpy as np
from src.features.core import window_features
from src.loaders.wesad import RATES


def make_wrist(duration=120):
    rng=np.random.default_rng(42)
    return {'EDA':rng.uniform(.1,2,(duration*4,1)),
            'BVP':rng.normal(size=(duration*64,1)),
            'ACC':rng.normal(size=(duration*32,3))}


class NoFutureTests(unittest.TestCase):
    def test_future_and_exact_endpoint_never_used(self):
        wrist=make_wrist()
        expected=window_features(wrist,0,60)
        for key,rate in RATES.items():
            wrist[key][60*rate:]=1e8
        observed=window_features(wrist,0,60)
        self.assertEqual(expected.keys(),observed.keys())
        for key in expected:
            if isinstance(expected[key],str):
                self.assertEqual(expected[key],observed[key])
            else:
                np.testing.assert_allclose(expected[key],observed[key],equal_nan=True)

    def test_outside_history_never_used(self):
        wrist=make_wrist()
        expected=window_features(wrist,30,90)
        for key,rate in RATES.items():
            wrist[key][:30*rate]=-1e9
        observed=window_features(wrist,30,90)
        for key in expected:
            if isinstance(expected[key],str):
                self.assertEqual(expected[key],observed[key])
            else:
                np.testing.assert_allclose(expected[key],observed[key],equal_nan=True)


if __name__ == '__main__':
    unittest.main()
