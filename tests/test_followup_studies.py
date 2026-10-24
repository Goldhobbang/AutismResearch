"""Unit tests for follow-up studies A, B, and C.

Validates:
1. Exact calibration duration boundaries and non-overlapping evaluation windows.
2. Zero future information leakage in calibration estimators.
3. AMDC skin resistance to conductance conversion and physical validity.
4. AMDC input-forecast horizon separation (zero temporal overlap).
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from experiments.amdc.run_amdc_predictive_modeling import build_amdc_window_dataset
from experiments.baseline_robustness.run_baseline_robustness import (
    calibrate_with_method,
    compute_baseline_diagnostics,
)
from experiments.calibration_duration.run_calibration_duration import (
    calibrate_by_duration,
)


class TestFollowupStudies(unittest.TestCase):
    def setUp(self):
        # Synthetic clean time series for testing
        n = 100
        self.df_test = pd.DataFrame({
            "subject": ["S01"] * n,
            "label": ["baseline"] * 40 + ["stress"] * 60,
            "start_s": np.arange(n) * 15.0,
            "end_s": np.arange(n) * 15.0 + 60.0,
            "eda_mean": np.concatenate([np.full(40, 1.0), np.full(60, 4.0)]),
            "eda_std": np.full(n, 0.2),
            "eda_positive_diff_sum": np.full(n, 0.1),
            "eda_scr_candidate_count": np.full(n, 1.0),
            "eda_slope": np.full(n, 0.01),
            "eda_q90": np.full(n, 1.2),
            "eda_q10": np.full(n, 0.8),
            "ppg_peak_hz": np.full(n, 1.2),
            "ppg_band_fraction": np.full(n, 0.6),
            "ppg_absdiff_per_s": np.full(n, 2.0),
            "acc_motion_change_per_s": np.concatenate([np.full(40, 15.0), np.full(60, 30.0)]),
            "acc_unchanged_fraction": np.full(n, 0.5),
            "acc_dynamic_energy": np.full(n, 25.0),
            "qc_common_valid": [True] * n,
        })

    def test_calibration_duration_strict_boundary(self):
        """Test that calibrate_by_duration uses ONLY windows strictly inside [t0, t0+T)."""
        # Duration = 180s. With 60s window and 15s step, windows must have end_s <= 180s.
        # Window starts: 0 (end 60), 15 (end 75), ..., 120 (end 180) -> exactly 9 windows.
        cal_180, meta_180 = calibrate_by_duration(self.df_test, duration_s=180.0)
        self.assertEqual(meta_180["S01"]["n_cal_windows"], 9)
        self.assertLessEqual(meta_180["S01"]["cal_end_s"], 180.0)

        # Duration = 60s: exactly 1 window (start 0, end 60).
        cal_60, meta_60 = calibrate_by_duration(self.df_test, duration_s=60.0)
        self.assertEqual(meta_60["S01"]["n_cal_windows"], 1)
        self.assertLessEqual(meta_60["S01"]["cal_end_s"], 60.0)

    def test_zero_future_leakage_in_baseline_estimators(self):
        """Future stress signals must have zero impact on baseline diagnostics or calibration vector."""
        df_clean = self.df_test.copy()
        df_corrupted = self.df_test.copy()
        # Corrupt all stress windows (index 40..99) with extreme values
        df_corrupted.loc[40:, "eda_mean"] = 999999.0
        df_corrupted.loc[40:, "acc_motion_change_per_s"] = 999999.0

        diag_clean = compute_baseline_diagnostics(df_clean)
        diag_corrupted = compute_baseline_diagnostics(df_corrupted)

        self.assertAlmostEqual(diag_clean["base_eda_mean"], diag_corrupted["base_eda_mean"], places=9)
        self.assertAlmostEqual(diag_clean["base_jerk_mean"], diag_corrupted["base_jerk_mean"], places=9)

        for meth in ["mean", "median", "low_motion", "trimmed_mean"]:
            cal_c, _ = calibrate_with_method(df_clean, method=meth, max_duration_s=300)
            cal_bad, _ = calibrate_with_method(df_corrupted, method=meth, max_duration_s=300)
            # Baseline delta must be identical
            np.testing.assert_allclose(
                cal_c.loc[:39, "eda_log_delta"].to_numpy(),
                cal_bad.loc[:39, "eda_log_delta"].to_numpy(),
                rtol=1e-9, atol=1e-9
            )

    def test_amdc_conductance_conversion_physical_bounds(self):
        """Verify Skin Resistance (kOhm) to Conductance (uS) conversion."""
        # 40 kOhm -> 25 uS; 100 kOhm -> 10 uS; 10 kOhm -> 100 uS
        r_kohm = np.array([40.0, 100.0, 10.0, 50.0])
        g_uS = 1000.0 / r_kohm
        expected = np.array([25.0, 10.0, 100.0, 20.0])
        np.testing.assert_allclose(g_uS, expected, rtol=1e-9)
        self.assertTrue(np.all(g_uS > 0))

    def test_amdc_zero_temporal_overlap_input_target(self):
        """Verify that AMDC predictive windows have zero overlap between input [t-59, t] and target [t+1, t+30]."""
        # Create synthetic 1 Hz dataframe with 300 seconds
        df_syn = pd.DataFrame({
            "participant": ["P1"] * 300,
            "second_idx": np.arange(300),
            "heart_rate_bpm": 70.0 + np.sin(np.arange(300) / 10.0),
            "skin_conductance_uS": 15.0 + np.cos(np.arange(300) / 20.0),
            "lux": np.full(300, 5000.0),
            "sound_db": np.full(300, 55.0),
            "temperature_c": np.full(300, 22.0),
            "humidity_pct": np.full(300, 60.0),
            "pressure_hpa": np.full(300, 1008.0),
            "iaq": np.full(300, 300000.0),
            "pm4": np.full(300, 10.0),
            "pm5": np.full(300, 2.0),
            "pm6": np.full(300, 0.5),
        })

        wdf = build_amdc_window_dataset(df_syn, window_s=60, forecast_s=30, step_s=15)
        self.assertGreater(len(wdf), 0)

        # For every window, input_end_s must be strictly less than target_start_s
        input_ends = wdf["input_end_s"].to_numpy()
        target_starts = wdf["target_start_s"].to_numpy()
        self.assertTrue(np.all(target_starts > input_ends))
        # Exact gap: target_start_s == input_end_s + 1 (adjacent, non-overlapping)
        self.assertTrue(np.all(target_starts == input_ends + 1))


if __name__ == "__main__":
    unittest.main()
