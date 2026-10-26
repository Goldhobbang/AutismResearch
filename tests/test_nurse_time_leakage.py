"""Unit tests verifying zero future information leakage in feature extraction and personal calibration."""

import unittest

import numpy as np
import pandas as pd

from src.features.nurse_forecasting import extract_nurse_window_features
from src.forecasting.calibration import apply_session_causal_calibration


class TestNurseTimeLeakage(unittest.TestCase):
    def _make_synthetic_session(self, duration_s: int = 600):
        rng = np.random.default_rng(123)
        return {
            "subject": "S_TEST",
            "session": "SESS_01",
            "start_ts": 10000.0,
            "end_ts": 10000.0 + duration_s,
            "duration_s": float(duration_s),
            "start_timestamps": {"EDA": 10000.0, "BVP": 10000.0, "ACC": 10000.0, "HR": 10010.0, "IBI": 10000.0},
            "rates": {"EDA": 4.0, "BVP": 64.0, "ACC": 32.0, "HR": 1.0},
            "EDA": rng.uniform(0.5, 2.5, size=duration_s * 4),
            "BVP": rng.normal(0.0, 10.0, size=duration_s * 64),
            "ACC": rng.normal(0.0, 20.0, size=(duration_s * 32, 3)),
            "HR": rng.uniform(70.0, 90.0, size=duration_s),
            "IBI": np.column_stack([np.arange(15.0, duration_s - 5.0, 0.8), np.full(int((duration_s - 20.0) / 0.8), 0.8)]),
        }

    def test_feature_extraction_strictly_ignores_future_samples(self):
        s1 = self._make_synthetic_session(600)
        s2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s1.items()}

        pred_t = 300.0  # Prediction at t = 300s
        # Mutate all samples at t >= 300s with extreme values in s2
        s2["EDA"][int(pred_t * 4) :] = 9999.0
        s2["BVP"][int(pred_t * 64) :] = -9999.0
        s2["ACC"][int(pred_t * 32) :, :] = 5000.0
        hr_idx = int((pred_t - 10.0) * 1.0)
        s2["HR"][hr_idx:] = 250.0

        f1 = extract_nurse_window_features(s1, pred_time_rel_s=pred_t, window_s=180.0)
        f2 = extract_nurse_window_features(s2, pred_time_rel_s=pred_t, window_s=180.0)

        for k in f1:
            if isinstance(f1[k], (int, float, np.floating)) and np.isfinite(f1[k]):
                self.assertAlmostEqual(
                    float(f1[k]),
                    float(f2[k]),
                    places=9,
                    msg=f"Feature {k} leaked future signal at >= t!",
                )

    def test_personal_calibration_strictly_ignores_future_windows(self):
        rows = []
        for i in range(10):
            rows.append(
                {
                    "subject": "15",
                    "session": "15_1",
                    "start_s": float(i * 60),
                    "end_s": float(180 + i * 60),
                    "state": "VERIFIED_NON_EVENT",
                    "eda_mean": 1.0 + 0.1 * i,
                    "eda_std": 0.1,
                    "eda_q10": 0.8,
                    "eda_q90": 1.2,
                    "eda_positive_diff_sum": 0.5,
                    "eda_scr_candidate_count": 2.0,
                    "eda_slope": 0.01,
                    "ppg_peak_hz": 1.3,
                    "ppg_band_fraction": 0.5,
                    "ppg_absdiff_per_s": 5.0,
                    "hr_mean_bpm": 78.0 + i,
                    "acc_motion_change_per_s": 20.0,
                }
            )
        df1 = pd.DataFrame(rows)
        df2 = df1.copy()
        # Corrupt future windows i >= 5
        df2.loc[5:, "eda_mean"] = 500.0
        df2.loc[5:, "hr_mean_bpm"] = 220.0

        for method in ("uncalibrated", "cal_1m", "cal_3m", "cal_5m", "low_motion", "adaptive_fallback"):
            c1 = apply_session_causal_calibration(df1, method=method)
            c2 = apply_session_causal_calibration(df2, method=method)
            # Windows 0..4 must be 100% identical
            np.testing.assert_allclose(
                c1.loc[:4, "eda_log_delta"].to_numpy(),
                c2.loc[:4, "eda_log_delta"].to_numpy(),
                atol=1e-12,
                err_msg=f"Calibration method {method} leaked future windows into past!",
            )


if __name__ == "__main__":
    unittest.main()
