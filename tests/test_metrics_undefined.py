import unittest
import numpy as np
from src.evaluate import score


class MetricDefinitionTests(unittest.TestCase):
    def test_no_alerts_has_undefined_precision_not_perfect_or_zero(self):
        result = score([0, 1, 0, 1], [.1, .2, .3, .4], .5)
        self.assertTrue(np.isnan(result['precision']))
        self.assertEqual(result['recall'], 0)
        self.assertEqual(result['f1'], 0)
        self.assertIn('No predicted positives', result['excluded_reason'])

    def test_single_class_auc_is_not_fabricated(self):
        result = score([0, 0], [.1, .7], .5)
        self.assertTrue(np.isnan(result['auroc']))
        self.assertTrue(np.isnan(result['auprc']))
        self.assertTrue(np.isnan(result['recall']))
        self.assertEqual(result['fp'], 1)

    def test_auroc_not_accuracy(self):
        result = score([0, 0, 1, 1], [.6, .7, .8, .9], .5)
        self.assertEqual(result['auroc'], 1)
        self.assertEqual(result['balanced_acc'], .5)
        self.assertEqual(result['precision'], .5)


if __name__ == '__main__':
    unittest.main()
