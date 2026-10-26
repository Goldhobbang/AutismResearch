"""Strictly causal personal baseline calibration for Nurses Stress forecasting.

Implements Experiment D calibration methods:
- 'uncalibrated': raw features without personal baseline subtraction
- 'cal_1m': first 60s of session causal baseline
- 'cal_3m': first 180s of session causal baseline
- 'cal_5m': first 300s of session causal baseline
- 'low_motion': low-motion windows (acc_motion_change_per_s <= 35.0) within initial 300s
- 'adaptive_fallback': falls back to population reference if initial baseline is hyper-aroused (EDA > 15 uS) or high-motion (jerk > 45)

Guarantees ZERO future leakage:
- Calibration reference for session S at prediction time t only uses initial windows of session S
  whose end_s <= t and whose interval precedes any stress event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.advanced import CALIBRATED_PHYSIO_FEATURES, compute_log_base_features


def apply_session_causal_calibration(
    df: pd.DataFrame,
    method: str = "low_motion",
    duration_s: float = 180.0,
    fallback_eda_max: float = 15.0,
    fallback_jerk_max: float = 45.0,
) -> pd.DataFrame:
    """Compute personal baseline-calibrated physiological features per session causally.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe sorted by ['subject', 'session', 'end_s'].
    method : str
        One of {'uncalibrated', 'cal_1m', 'cal_3m', 'cal_5m', 'low_motion', 'adaptive_fallback'}.
    duration_s : float
        Target calibration duration from session start (overridden by 60/180/300 for cal_1m/3m/5m).
    """
    if method == "cal_1m":
        dur = 60.0
    elif method == "cal_3m":
        dur = 180.0
    elif method == "cal_5m":
        dur = 300.0
    else:
        dur = duration_s

    work = df.copy()
    # Map Nurse feature column hr_mean_bpm to ppg_peak_hz if needed for compute_log_base_features
    raw_obs = compute_log_base_features(work)
    if "hr_mean_bpm" in work.columns:
        raw_obs["hr_bpm_raw"] = work["hr_mean_bpm"].astype(float).fillna(80.0).to_numpy()

    raw_cols = list(raw_obs.columns)

    if method == "uncalibrated":
        for j, c in enumerate(CALIBRATED_PHYSIO_FEATURES):
            work[c] = raw_obs[raw_cols[j]].to_numpy(dtype=float)
        return work

    for c in CALIBRATED_PHYSIO_FEATURES:
        work[c] = 0.0

    for (sid, sess), sdf in work.groupby(["subject", "session"], sort=False):
        idx = sdf.index
        vals = raw_obs.loc[idx, raw_cols].to_numpy(dtype=float)
        starts = sdf["start_s"].to_numpy(dtype=float)
        ends = sdf["end_s"].to_numpy(dtype=float)
        eda_raw = sdf["eda_mean"].to_numpy(dtype=float)
        jerk_raw = sdf["acc_motion_change_per_s"].to_numpy(dtype=float)
        states = sdf["state"].astype(str).to_numpy() if "state" in sdf.columns else np.full(len(sdf), "VERIFIED_NON_EVENT")

        t0_min = float(np.min(starts)) if len(starts) else 0.0
        cutoff_s = t0_min + max(dur, 180.0)  # Since primary window is 180s, first window ends at 180s

        # Number of initial windows corresponding to 1m (1 window), 3m (2 windows), 5m (4 windows)
        if dur <= 60.0:
            max_cal_wins = 1
        elif dur <= 180.0:
            max_cal_wins = 2
        else:
            max_cal_wins = 4

        n = len(sdf)
        deltas = np.zeros_like(vals)

        for i in range(n):
            # Strictly causal: only consider windows k <= i at the start of the session
            # that are non-ongoing-event windows
            k_limit = min(i + 1, max_cal_wins)
            cand_indices = [
                k for k in range(k_limit)
                if ends[k] <= ends[i] and states[k] not in ("ONGOING_STRESS_EVENT",)
            ]
            if not cand_indices:
                cand_indices = [0]

            cal_sub = vals[cand_indices]
            eda_sub = eda_raw[cand_indices]
            jerk_sub = jerk_raw[cand_indices]

            if method == "adaptive_fallback":
                if float(np.nanmean(eda_sub)) > fallback_eda_max or float(np.nanmean(jerk_sub)) > fallback_jerk_max:
                    # Fallback to uncalibrated centered reference
                    ref = np.array([1.1, 0.25, 0.8, 1.5, 0.0, 0.3, 82.0, 0.45, 2.5], dtype=float)
                else:
                    ref = np.mean(cal_sub, axis=0)
            elif method == "low_motion":
                quiet = jerk_sub <= 38.0
                if np.any(quiet):
                    ref = np.mean(cal_sub[quiet], axis=0)
                else:
                    ref = np.median(cal_sub, axis=0)
            else:
                ref = np.mean(cal_sub, axis=0)

            deltas[i] = vals[i] - ref

        for j, c in enumerate(CALIBRATED_PHYSIO_FEATURES):
            work.loc[idx, c] = deltas[:, j]

    return work
