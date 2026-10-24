"""A frozen WESAD feature-named pipeline must receive a named DataFrame."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import warnings

import numpy as np
import pandas as pd

from experiments.phase2 import zero_shot_physionet as zero_shot


class FeatureNameInferenceTest(unittest.TestCase):
    def test_named_frame_and_saved_feature_order(self):
        # A real SimpleImputer/StandardScaler pipeline would warn when fed an
        # ndarray despite having been fitted with feature names. Verify that
        # our wrapper preserves both names and order across the 15 folds.
        class TrainedFeatureNames:
            def predict_proba(self, frame):
                if not isinstance(frame, pd.DataFrame):
                    raise AssertionError("Feature names lost during inference")
                if list(frame.columns) != ["acc_b", "acc_a"]:
                    raise AssertionError("Saved model feature ordering lost")
                p = np.full(len(frame), 0.3)
                return np.column_stack((1.0 - p, p))

        rows = pd.DataFrame({
            "acc_a": [1.0, 2.0],
            "acc_b": [3.0, 4.0],
            "has_bvp": [False, False],
        })
        with TemporaryDirectory() as folder:
            root = Path(folder)
            # Require all 15 frozen folds for each non-BVP model family.
            for family in ["lr_acc", "lr_eda", "lr_eda_acc"]:
                for fold in range(15):
                    (root / f"{family}_fold{fold}.joblib").touch()
            fake_bundle = {
                "model": TrainedFeatureNames(),
                "columns": ["acc_b", "acc_a"],
                "threshold": 0.5,
            }
            with patch.object(zero_shot, "MODELS_DIR", root):
                with patch.object(zero_shot.joblib, "load", return_value=fake_bundle):
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        out = zero_shot.evaluate_models_on_windows(rows)
            self.assertEqual(len(captured), 0)
            self.assertTrue(np.allclose(out["prob_lr_acc"], 0.3))
            self.assertTrue((out["pred05_lr_acc"] == 0).all())


if __name__ == "__main__":
    unittest.main()
