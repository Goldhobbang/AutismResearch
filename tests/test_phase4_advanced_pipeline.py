"""Unit and regression test suite for Phase 4 advanced feature pipeline.

Verifies:
- TC-01: TestOrientationInvariance (3D SO(3) rotation invariance of ACC features)
- TC-02: TestCausalCalibrationAndTemporalNoFutureLeakage (Zero future leakage in personal calibration, online baseline updates, and temporal EMA/diff features)
- TC-03: TestHighTonicAndLowResponderMonotonicity (Monotonicity and AUROC >= 0.85 across extreme low-responder and high-tonic EDA baselines without quadratic explosion)
- TC-04: TestMotionGateRecallPreservationAndExerciseSuppression (Dual objective: gesture-accompanied stress recall preservation vs sustained aerobic exercise false-alarm suppression, plus rank preservation)
- TC-05: TestTagAlignedManifestIntegrity (Official exclusion rules and segment duration bounds on PhysioNet manifests)
"""

from __future__ import annotations

import pathlib
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features.advanced import (
    CALIBRATED_PHYSIO_FEATURES,
    CAUSAL_TEMPORAL_FEATURES,
    add_causal_temporal_features,
    apply_causal_personal_calibration,
    apply_cbara_probability_calibration,
    apply_dynamic_motion_gate,
    compute_causal_exercise_score,
    compute_extended_window_features,
    extract_orientation_invariant_acc_features,
    infer_joint_4state,
)


def _euler_rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """Construct 3D rotation matrix R = R_z(psi) @ R_y(theta) @ R_x(phi) in SO(3)."""
    cx, sx = np.cos(phi), np.sin(phi)
    cy, sy = np.cos(theta), np.sin(theta)
    cz, sz = np.cos(psi), np.sin(psi)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


class TestOrientationInvariance(unittest.TestCase):
    """TC-01: Verify 3D SO(3) orientation invariance of accelerometer features."""

    def test_so3_rotation_invariance_all_invariant_features(self) -> None:
        rng = np.random.default_rng(42)
        fs = 32.0
        n_samples = 1920  # 60s at 32 Hz
        t = np.arange(n_samples, dtype=float) / fs

        # Synthetic 3-axis ACC: gravity vector + multi-frequency sinusoids + Gaussian vibration
        gravity = np.array([32.0, -45.0, 28.0], dtype=float)
        dynamic_osc = np.column_stack(
            [
                4.0 * np.sin(2.0 * np.pi * 1.8 * t) + 1.2 * rng.standard_normal(n_samples),
                3.0 * np.cos(2.0 * np.pi * 1.8 * t + 0.4) + 0.9 * rng.standard_normal(n_samples),
                2.5 * np.sin(2.0 * np.pi * 3.6 * t - 0.2) + 1.1 * rng.standard_normal(n_samples),
            ]
        )
        acc = gravity + dynamic_osc
        # Inject stationary plateaus so acc_unchanged_fraction is non-trivial
        for start_idx in (100, 600, 1200):
            acc[start_idx : start_idx + 80] = acc[start_idx]

        feat_orig = extract_orientation_invariant_acc_features(acc, fs=fs)

        invariant_keys = [
            "acc_mag_mean",
            "acc_mag_std",
            "acc_mag_range",
            "acc_mag_absdiff_per_s",
            "acc_motion_change_per_s",
            "acc_unchanged_fraction",
            "acc_peak_hz",
            "acc_band_fraction",
            "acc_dynamic_energy",
        ]

        # Test multiple arbitrary 3D rotations in SO(3)
        angles_list = [
            (0.73, -1.12, 2.41),
            (-2.15, 0.48, -0.93),
            (np.pi / 2.0, np.pi / 3.0, np.pi / 4.0),
        ]
        for phi, theta, psi in angles_list:
            rot = _euler_rotation_matrix(phi, theta, psi)
            # Verify R is strictly in SO(3)
            np.testing.assert_allclose(rot.T @ rot, np.eye(3), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(rot), 1.0, rtol=1e-12, atol=1e-12)

            acc_rot = acc @ rot.T
            feat_rot = extract_orientation_invariant_acc_features(acc_rot, fs=fs)

            for k in invariant_keys:
                np.testing.assert_allclose(
                    feat_rot[k],
                    feat_orig[k],
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"Feature {k} changed under 3D rotation ({phi}, {theta}, {psi})",
                )


class TestCausalCalibrationAndTemporalNoFutureLeakage(unittest.TestCase):
    """TC-02: Verify zero future leakage in personal calibration, online baseline update, and temporal features."""

    def _make_subject_timeseries(self) -> pd.DataFrame:
        rng = np.random.default_rng(102)
        starts = np.arange(0, 1230, 30, dtype=float)  # t in [0, 1200]s (41 windows)
        n = len(starts)
        labels = np.where(starts <= 240.0, "baseline", "stress")
        df = pd.DataFrame(
            {
                "subject": ["S01"] * n,
                "session": ["S01_STRESS"] * n,
                "start_s": starts,
                "end_s": starts + 60.0,
                "label": labels,
                "eda_mean": 1.5 + 0.1 * rng.standard_normal(n),
                "eda_std": 0.08 + 0.01 * np.abs(rng.standard_normal(n)),
                "eda_positive_diff_sum": 0.4 + 0.05 * np.abs(rng.standard_normal(n)),
                "eda_scr_candidate_count": rng.integers(1, 6, size=n).astype(float),
                "eda_slope": 0.002 * rng.standard_normal(n),
                "eda_iqr": 0.10 + 0.02 * np.abs(rng.standard_normal(n)),
                "ppg_peak_hz": 1.25 + 0.08 * rng.standard_normal(n),
                "ppg_band_fraction": 0.65 + 0.05 * rng.standard_normal(n),
                "ppg_absdiff_per_s": 12.0 + 1.5 * rng.standard_normal(n),
                "acc_unchanged_fraction": np.full(n, 0.55, dtype=float),
                "acc_motion_change_per_s": np.full(n, 18.0, dtype=float),
                "acc_dynamic_energy": np.full(n, 12.0, dtype=float),
            }
        )
        return df

    def test_future_contamination_zero_impact_on_past(self) -> None:
        df_clean = self._make_subject_timeseries()
        df_corrupted = df_clean.copy()

        future_mask = df_corrupted["start_s"] > 600.0
        feature_cols_to_corrupt = [
            "eda_mean",
            "eda_std",
            "eda_positive_diff_sum",
            "eda_scr_candidate_count",
            "eda_slope",
            "eda_iqr",
            "ppg_peak_hz",
            "ppg_band_fraction",
            "ppg_absdiff_per_s",
            "acc_unchanged_fraction",
            "acc_motion_change_per_s",
            "acc_dynamic_energy",
        ]
        for col in feature_cols_to_corrupt:
            df_corrupted.loc[future_mask, col] = 1e6
        # Also corrupt future labels to "baseline" to verify future baseline labels are ignored
        df_corrupted.loc[future_mask, "label"] = "baseline"

        for drift_rate in (0.0, 0.05):
            cal_clean = apply_causal_personal_calibration(
                df_clean,
                is_baseline_candidate=(df_clean["start_s"] <= 240.0),
                max_init_windows=10,
                online_drift_rate=drift_rate,
            )
            cal_clean = add_causal_temporal_features(cal_clean)

            cal_corrupted = apply_causal_personal_calibration(
                df_corrupted,
                is_baseline_candidate=(df_corrupted["start_s"] <= 240.0),
                max_init_windows=10,
                online_drift_rate=drift_rate,
            )
            cal_corrupted = add_causal_temporal_features(cal_corrupted)

            past_mask = df_clean["start_s"] <= 600.0
            check_cols = CALIBRATED_PHYSIO_FEATURES + CAUSAL_TEMPORAL_FEATURES
            for col in check_cols:
                np.testing.assert_allclose(
                    cal_corrupted.loc[past_mask, col].to_numpy(),
                    cal_clean.loc[past_mask, col].to_numpy(),
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"Future leakage detected in {col} (online_drift_rate={drift_rate})",
                )

            # Verify downstream probability & motion gate outputs up to t <= 600s
            raw_score_clean = 1.0 / (
                1.0 + np.exp(-(2.5 * cal_clean["eda_log_delta_ema"].to_numpy() - 0.3))
            )
            raw_score_corrupted = 1.0 / (
                1.0 + np.exp(-(2.5 * cal_corrupted["eda_log_delta_ema"].to_numpy() - 0.3))
            )
            p_cbara_clean = apply_cbara_probability_calibration(
                raw_score_clean, (df_clean["start_s"] <= 240.0).to_numpy()
            )
            p_cbara_corrupted = apply_cbara_probability_calibration(
                raw_score_corrupted, (df_corrupted["start_s"] <= 240.0).to_numpy()
            )
            p_gated_clean, thr_clean, _ = apply_dynamic_motion_gate(
                p_cbara_clean, cal_clean, mode="soft_dynamic"
            )
            p_gated_corrupted, thr_corrupted, _ = apply_dynamic_motion_gate(
                p_cbara_corrupted, cal_corrupted, mode="soft_dynamic"
            )

            np.testing.assert_allclose(
                p_gated_corrupted[past_mask],
                p_gated_clean[past_mask],
                rtol=1e-12,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                thr_corrupted[past_mask],
                thr_clean[past_mask],
                rtol=1e-12,
                atol=1e-12,
            )


class TestHighTonicAndLowResponderMonotonicity(unittest.TestCase):
    """TC-03: Verify calibration invariance and monotonicity on Low-Responder and High-Tonic subjects."""

    def _build_subject_df(
        self, subject_id: str, eda_base: float, eda_stress: float, seed: int
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n_base = 10
        n_stress = 12
        n = n_base + n_stress
        starts = np.arange(n, dtype=float) * 30.0
        is_stress = np.array([0] * n_base + [1] * n_stress, dtype=int)

        eda_levels = np.where(is_stress == 1, eda_stress, eda_base)
        noise_scale = 0.02 * eda_levels
        eda_mean = eda_levels + rng.normal(0.0, noise_scale, size=n)
        eda_std = np.where(is_stress == 1, 0.08 * eda_levels, 0.02 * eda_levels) * (
            1.0 + 0.1 * rng.uniform(-1.0, 1.0, size=n)
        )
        eda_pos_diff = np.where(is_stress == 1, 0.35 * eda_levels, 0.06 * eda_levels) * (
            1.0 + 0.1 * rng.uniform(-1.0, 1.0, size=n)
        )
        eda_scr = np.where(is_stress == 1, 5.0, 1.0) + rng.uniform(-0.3, 0.3, size=n)
        eda_slope = np.where(is_stress == 1, 0.003 * eda_levels, 0.0)
        eda_iqr = np.where(is_stress == 1, 0.10 * eda_levels, 0.03 * eda_levels)

        return pd.DataFrame(
            {
                "subject": [subject_id] * n,
                "session": [f"{subject_id}_STRESS"] * n,
                "start_s": starts,
                "end_s": starts + 60.0,
                "label": np.where(is_stress == 1, "stress", "baseline"),
                "y": is_stress,
                "eda_mean": eda_mean,
                "eda_std": eda_std,
                "eda_positive_diff_sum": eda_pos_diff,
                "eda_scr_candidate_count": eda_scr,
                "eda_slope": eda_slope,
                "eda_iqr": eda_iqr,
                "ppg_peak_hz": np.where(is_stress == 1, 1.45, 1.15)
                + rng.normal(0.0, 0.03, size=n),
                "ppg_band_fraction": np.full(n, 0.65, dtype=float),
                "ppg_absdiff_per_s": np.where(is_stress == 1, 16.0, 11.0),
                "acc_unchanged_fraction": np.full(n, 0.50, dtype=float),
                "acc_motion_change_per_s": np.full(n, 15.0, dtype=float),
                "acc_dynamic_energy": np.full(n, 10.0, dtype=float),
            }
        )

    def test_low_responder_and_high_tonic_monotonicity_and_auroc(self) -> None:
        df_low = self._build_subject_df(
            subject_id="S_LOW", eda_base=0.20, eda_stress=0.35, seed=11
        )
        df_high = self._build_subject_df(
            subject_id="S_HIGH", eda_base=20.0, eda_stress=28.0, seed=22
        )

        for sdf in (df_low, df_high):
            cal = apply_causal_personal_calibration(
                sdf,
                is_baseline_candidate=(sdf["label"] == "baseline").to_numpy(),
                max_init_windows=10,
            )
            cal = add_causal_temporal_features(cal)

            # Linear sympathetic activation score from calibrated log features
            lin_logit = (
                4.0 * cal["eda_log_delta_ema"].to_numpy()
                + 2.0 * cal["eda_pos_diff_log_delta_ema"].to_numpy()
                + 1.5 * cal["eda_scr_log_delta_ema"].to_numpy()
                + 0.05 * cal["hr_bpm_delta_ema"].to_numpy()
                - 0.4
            )
            p_raw = 1.0 / (1.0 + np.exp(-lin_logit))
            p_cal = apply_cbara_probability_calibration(
                p_raw, (sdf["label"] == "baseline").to_numpy()
            )

            y = sdf["y"].to_numpy()
            p_base_mean = float(np.mean(p_cal[y == 0]))
            p_stress_mean = float(np.mean(p_cal[y == 1]))
            auroc = float(roc_auc_score(y, p_cal))

            self.assertGreater(
                p_stress_mean,
                p_base_mean,
                msg=f"Monotonicity violated for {sdf['subject'].iloc[0]}: p_stress={p_stress_mean} <= p_base={p_base_mean}",
            )
            self.assertGreaterEqual(
                auroc,
                0.85,
                msg=f"AUROC {auroc:.4f} < 0.85 for {sdf['subject'].iloc[0]}",
            )


class TestMotionGateRecallPreservationAndExerciseSuppression(unittest.TestCase):
    """TC-04: Verify motion gate preserves gesture-accompanied stress recall while suppressing sustained aerobic exercise."""

    def test_gesture_stress_recall_and_exercise_suppression(self) -> None:
        base_thr = 0.45

        # Scenario 1: Real stress accompanied by intermittent hand gestures
        df_gesture_stress = pd.DataFrame(
            {
                "subject": ["S01", "S01", "S01"],
                "session": ["S01_STRESS", "S01_STRESS", "S01_STRESS"],
                "start_s": [300.0, 330.0, 360.0],
                "acc_unchanged_fraction": [0.25, 0.25, 0.25],
                "acc_motion_change_per_s": [65.0, 65.0, 65.0],
                "acc_dynamic_energy": [220.0, 220.0, 220.0],
                "eda_log_delta": [0.45, 0.50, 0.55],
                "eda_scr_log_delta": [0.60, 0.65, 0.70],
            }
        )
        p_raw_stress = np.array([0.78, 0.82, 0.86], dtype=float)

        # Legacy hard gate (b46a2e9) falsely suppresses gesture stress because acc_motion_change_per_s > 40
        p_legacy, thr_legacy, pred_legacy = apply_dynamic_motion_gate(
            p_raw_stress, df_gesture_stress, mode="legacy_hard", base_threshold=base_thr
        )
        self.assertTrue(np.all(pred_legacy == 0), "Legacy hard gate should have clamped gesture stress")

        # Improved soft_dynamic gate preserves positive stress detection (P_gated >= theta)
        p_soft_1, thr_soft_1, pred_soft_1 = apply_dynamic_motion_gate(
            p_raw_stress, df_gesture_stress, mode="soft_dynamic", base_threshold=base_thr
        )
        self.assertTrue(
            np.all(p_soft_1 >= thr_soft_1),
            f"Soft dynamic gate failed to preserve gesture stress recall: {p_soft_1} vs {thr_soft_1}",
        )
        self.assertTrue(np.all(pred_soft_1 == 1))

        # Scenario 2: Sustained aerobic cycling exercise (sweating -> high raw EDA + continuous pedaling)
        df_exercise = pd.DataFrame(
            {
                "subject": ["S01", "S01", "S01"],
                "session": ["S01_AEROBIC", "S01_AEROBIC", "S01_AEROBIC"],
                "start_s": [180.0, 210.0, 240.0],
                "acc_unchanged_fraction": [0.005, 0.005, 0.005],
                "acc_motion_change_per_s": [165.0, 165.0, 165.0],
                "acc_dynamic_energy": [950.0, 950.0, 950.0],
                "eda_log_delta": [0.80, 0.85, 0.90],
                "eda_scr_log_delta": [0.35, 0.40, 0.45],
            }
        )
        p_raw_exercise = np.array([0.78, 0.82, 0.86], dtype=float)

        p_soft_2, thr_soft_2, pred_soft_2 = apply_dynamic_motion_gate(
            p_raw_exercise, df_exercise, mode="soft_dynamic", base_threshold=base_thr
        )
        self.assertTrue(
            np.all(p_soft_2 < thr_soft_2),
            f"Soft dynamic gate failed to suppress sustained exercise false alarm: {p_soft_2} vs {thr_soft_2}",
        )
        self.assertTrue(np.all(pred_soft_2 == 0))

        # Verify relative ranking is strictly preserved (not collapsed to a flat constant 0.05)
        self.assertTrue(
            np.all(np.diff(p_soft_2) > 0.0),
            f"Soft gating destroyed relative window ranking: {p_soft_2}",
        )

        # Verify joint 4-state inference classifies Scenario 2 as exercise_routine (state 2)
        joint_ex = infer_joint_4state(p_raw_exercise, df_exercise, base_threshold=base_thr)
        self.assertTrue(np.all(joint_ex["joint_state_id"].to_numpy() == 2))


class TestTagAlignedManifestIntegrity(unittest.TestCase):
    """TC-05: Verify official exclusion rules and segment duration bounds on PhysioNet manifests."""

    def test_manifests_exclusion_and_duration_bounds(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[1]
        manifest_paths = [
            root / "data" / "processed" / "physionet" / "verified_segments_tag_aligned.csv",
            root / "data" / "processed" / "physionet" / "verified_segments.csv",
        ]

        forbidden_sessions = {
            ("S02", "S02_STRESS"),
            ("S03", "S03_AEROBIC"),
            ("S07", "S07_AEROBIC"),
            ("S11", "S11_AEROBIC"),
            ("S12", "S12_AEROBIC"),
        }

        for path in manifest_paths:
            self.assertTrue(path.exists(), f"Manifest file missing: {path}")
            df = pd.read_csv(path)
            verified_mask = df["verified"].astype(str).str.lower() == "true"
            eligible = df.loc[verified_mask].copy()
            self.assertGreater(len(eligible), 0, f"No verified segments found in {path.name}")

            # 1. Official excluded sessions must never appear in verified == True
            for _, row in eligible.iterrows():
                pair = (str(row["subject"]).strip(), str(row["session"]).strip())
                self.assertNotIn(
                    pair,
                    forbidden_sessions,
                    msg=f"Excluded session {pair} marked verified=True in {path.name}",
                )

            # 2. Every eligible segment must satisfy end_s > start_s and duration >= 60s
            starts = eligible["start_s"].astype(float).to_numpy()
            ends = eligible["end_s"].astype(float).to_numpy()
            durations = ends - starts
            self.assertTrue(
                np.all(ends > starts),
                msg=f"Found segment with end_s <= start_s in {path.name}",
            )
            self.assertTrue(
                np.all(durations >= 60.0),
                msg=f"Found verified segment shorter than 60s in {path.name}: min={np.min(durations)}",
            )


if __name__ == "__main__":
    unittest.main()
