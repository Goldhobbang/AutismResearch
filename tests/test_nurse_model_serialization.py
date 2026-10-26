"""Unit tests for final Nurse Stress forecasting model serialization, JSON spec parity, NaN handling, and raw session replay."""

import hashlib
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.nurse_forecasting import SENSOR_FEATURE_MAP, extract_nurse_window_features
from src.forecasting.calibration import apply_session_causal_calibration
from src.loaders.nurse_stress import load_nurse_session_zip

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "nurse_forecasting"
MOD_DIR = OUT_DIR / "models"


class TestNurseModelSerialization(unittest.TestCase):
    def test_saved_lr_artifact_and_json_spec_consistency(self):
        lr_path = MOD_DIR / "nurse_stress_forecast_lr_p1.joblib"
        spec_path = MOD_DIR / "nurse_stress_forecast_lr_p1_spec.json"
        self.assertTrue(lr_path.exists(), f"Missing {lr_path}")
        self.assertTrue(spec_path.exists(), f"Missing {spec_path}")

        art = joblib.load(lr_path)
        spec = json.loads(spec_path.read_text(encoding="utf-8"))

        self.assertEqual(art["model_version"], "1.0.0-nurse-p1")
        self.assertEqual(art["task"], "P1_future_stress_event_onset_60_to_180s")
        self.assertEqual(art["n_features"], 62)
        self.assertEqual(spec["n_features"], 62)
        self.assertEqual(art["feature_columns"], spec["feature_columns"])
        self.assertEqual(art["feature_columns"], SENSOR_FEATURE_MAP["all"])
        self.assertAlmostEqual(float(art["decision_threshold"]), float(spec["decision_threshold"]), places=12)

        # Verify training data hash matches features.csv
        feat_csv = OUT_DIR / "features.csv"
        if feat_csv.exists():
            actual_hash = hashlib.sha256(feat_csv.read_bytes()).hexdigest()
            self.assertEqual(art["training_data_hash_sha256"], actual_hash)
            self.assertEqual(spec["training_data_hash_sha256"], actual_hash)

    def test_pipeline_vs_json_manual_inference_and_nan_handling(self):
        art = joblib.load(MOD_DIR / "nurse_stress_forecast_lr_p1.joblib")
        spec = json.loads((MOD_DIR / "nurse_stress_forecast_lr_p1_spec.json").read_text(encoding="utf-8"))
        pipe = art["pipeline"]
        cols = art["feature_columns"]

        rng = np.random.default_rng(2026)
        X_df = pd.DataFrame(rng.normal(0.0, 2.0, size=(25, len(cols))), columns=cols)
        # Inject all-NaN row and partial-NaN row to test imputer robustness
        X_df.iloc[0, :] = np.nan
        X_df.iloc[1, ::2] = np.nan

        probs_pipe = pipe.predict_proba(X_df)[:, 1]
        self.assertTrue(np.all(np.isfinite(probs_pipe)))
        self.assertTrue(np.all((probs_pipe >= 0.0) & (probs_pipe <= 1.0)))

        # Compute manual C/JSON inference formula
        X_np = X_df.to_numpy(dtype=float)
        med = np.array([spec["imputer_median"][c] for c in cols], dtype=float)
        mean = np.array([spec["scaler_mean"][c] for c in cols], dtype=float)
        scale = np.array([spec["scaler_scale"][c] for c in cols], dtype=float)
        w = np.array([spec["coefficients"][c] for c in cols], dtype=float)
        b = float(spec["intercept"])

        nan_rows, nan_cols = np.where(np.isnan(X_np))
        X_imp = X_np.copy()
        X_imp[nan_rows, nan_cols] = med[nan_cols]
        Z = (X_imp - mean) / scale
        logits = Z @ w + b
        probs_json = 1.0 / (1.0 + np.exp(-logits))

        np.testing.assert_allclose(probs_pipe, probs_json, atol=1e-12)

    def test_raw_session_replay_end_to_end(self):
        art = joblib.load(MOD_DIR / "nurse_stress_forecast_lr_p1.joblib")
        pipe = art["pipeline"]
        cols = art["feature_columns"]
        manifest = json.loads((OUT_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
        annot_sess = [s for s in manifest["sessions"] if s["has_survey_annotation"]][0]

        sdata = load_nurse_session_zip(annot_sess["zip_path"], include_temp=False)
        f1 = extract_nurse_window_features(sdata, pred_time_rel_s=180.0, window_s=180.0)
        f2 = extract_nurse_window_features(sdata, pred_time_rel_s=240.0, window_s=180.0)
        replay_df = pd.DataFrame(
            [
                {
                    "subject": annot_sess["subject"],
                    "session": annot_sess["session"],
                    "start_s": 0.0,
                    "end_s": 180.0,
                    "pred_ts": sdata["start_ts"] + 180.0,
                    "state": "VERIFIED_NON_EVENT",
                    **f1,
                },
                {
                    "subject": annot_sess["subject"],
                    "session": annot_sess["session"],
                    "start_s": 60.0,
                    "end_s": 240.0,
                    "pred_ts": sdata["start_ts"] + 240.0,
                    "state": "VERIFIED_NON_EVENT",
                    **f2,
                },
            ]
        )
        replay_cal = apply_session_causal_calibration(replay_df, method="low_motion", duration_s=180.0)
        probs = pipe.predict_proba(replay_cal[cols])[:, 1]
        self.assertEqual(len(probs), 2)
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.all((probs > 0.0) & (probs < 1.0)))


if __name__ == "__main__":
    unittest.main()
