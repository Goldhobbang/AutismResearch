import unittest
import numpy as np
from src.features.core import label_window, window_features
from src.quality.checks import signal_quality


class LabelBoundsTests(unittest.TestCase):
    def setUp(self):
        self.labels=np.repeat([1,2,3],60*700)

    def test_exact_transition_end_excluded(self):
        self.assertEqual(label_window(self.labels,0,60),(1,0))
        self.assertEqual(label_window(self.labels,60,120),(2,1))
        self.assertEqual(label_window(self.labels,120,180),(3,0))

    def test_one_sample_mixture_rejected(self):
        self.labels[60*700-1]=2
        self.assertEqual(label_window(self.labels,0,60),(0,-1))

    def test_unsupported_phase_and_truncated_labels(self):
        self.assertEqual(label_window(np.full(60*700,4),0,60),(4,-1))
        self.assertEqual(label_window(np.full(60*700-1,1),0,60),(0,-1))

    def test_quality_cutoff_and_motion_ambiguity(self):
        x=np.arange(100,dtype=float)
        x[:5]=np.nan
        self.assertTrue(signal_quality(x,'eda')['qc_eda_valid'])
        x[5]=np.nan
        self.assertFalse(signal_quality(x,'eda')['qc_eda_valid'])
        constant=np.tile([0.,0.,64.],(100,1))
        quality=signal_quality(constant,'acc')
        self.assertTrue(quality['qc_acc_flatline'])
        self.assertTrue(quality['qc_acc_valid'])

    def test_invalid_sensor_features_masked(self):
        rng=np.random.default_rng(4)
        wrist={'EDA':np.ones((240,1)), 'BVP':rng.normal(size=(3840,1)),
               'ACC':rng.normal(size=(1920,3))}
        row=window_features(wrist,0,60)
        self.assertFalse(row['qc_common_valid'])
        self.assertTrue(all(np.isnan(v) for k,v in row.items() if k.startswith('eda_')))


if __name__ == '__main__':
    unittest.main()
