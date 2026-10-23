import unittest
import numpy as np
from src.features.core import native_slice
from src.loaders.physionet import parse_start


class TimeAlignmentTests(unittest.TestCase):
    def test_native_rates_share_same_time_interval(self):
        for rate in [4, 32, 64, 700]:
            timestamps = np.arange(120*rate)/rate
            selected = native_slice(timestamps, 15, 75, rate)
            self.assertEqual(len(selected),60*rate)
            self.assertEqual(selected[0],15)
            self.assertLess(selected[-1],75)
            self.assertTrue(np.all(selected >= 15))

    def test_non_grid_boundaries(self):
        selected = native_slice(np.arange(8)/4,.125,.625,4)
        np.testing.assert_array_equal(selected,[.25,.5])

    def test_utc_and_epoch_parse(self):
        self.assertEqual(parse_start('1970-01-01 00:00:01 UTC'),1)
        self.assertEqual(parse_start('1.25'),1.25)

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            native_slice(np.zeros(10),-1,1,4)


if __name__ == '__main__':
    unittest.main()
