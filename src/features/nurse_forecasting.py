"""Causal multi-scale feature extraction for Nurses Stress future event forecasting.

Extracts 180s (primary), 60s (short-scale), and 300s (long-scale) causal features
strictly from observations in [t - W, t) prior to prediction timestamp t.
Reuses validated feature primitives and signal quality checks from `src.features.core`
and `src.features.advanced`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.features.advanced import (
    CALIBRATED_PHYSIO_FEATURES,
    ORIENTATION_INVARIANT_ACC_FEATURES,
    compute_log_base_features,
    extract_orientation_invariant_acc_features,
)
from src.features.core import basic, native_slice, spectrum
from src.loaders.nurse_stress import (
    audit_nurse_dataset,
    load_nurse_session_zip,
    resolve_nurse_root,
)
from src.quality.checks import signal_quality

# Feature group definitions for sensor ablation and modeling
EDA_RAW_FEATURES = [
    "eda_mean",
    "eda_median",
    "eda_std",
    "eda_range",
    "eda_q10",
    "eda_q90",
    "eda_slope",
    "eda_absdiff_per_s",
    "eda_positive_diff_sum",
    "eda_scr_candidate_count",
    "eda_log_mean",
    "eda_log_std",
    "eda_log_slope",
    "eda_rise_fraction",
    "eda_recovery_ratio",
    "eda_recent60_minus_prev120",
    "eda_60s_mean",
    "eda_60s_slope",
    "eda_300s_mean",
    "eda_300s_slope",
]

PPG_RAW_FEATURES = [
    "ppg_mean",
    "ppg_std",
    "ppg_range",
    "ppg_q10",
    "ppg_q90",
    "ppg_absdiff_per_s",
    "ppg_peak_hz",
    "ppg_band_fraction",
    "hr_mean_bpm",
    "hr_std_bpm",
    "hr_range_bpm",
    "hr_slope_bpm_per_s",
    "hr_delta_recent60_bpm",
    "hr_60s_mean_bpm",
    "hr_300s_mean_bpm",
    "ibi_rmssd_ms",
    "ibi_sdnn_ms",
    "ibi_valid_flag",
]

ACC_RAW_FEATURES = [
    "acc_mag_mean",
    "acc_mag_median",
    "acc_mag_std",
    "acc_mag_range",
    "acc_mag_q10",
    "acc_mag_q90",
    "acc_mag_slope",
    "acc_mag_absdiff_per_s",
    "acc_dynamic_energy",
    "acc_motion_change_per_s",
    "acc_unchanged_fraction",
    "acc_peak_hz",
    "acc_band_fraction",
    "acc_60s_motion_change",
    "acc_300s_motion_change",
]

CALIBRATED_FORECAST_FEATURES = list(CALIBRATED_PHYSIO_FEATURES) + [
    "eda_recent60_minus_prev120",
    "hr_slope_bpm_per_s",
    "hr_delta_recent60_bpm",
]

SENSOR_FEATURE_MAP = {
    "eda": EDA_RAW_FEATURES + [
        "eda_log_delta",
        "eda_log_std_delta",
        "eda_pos_diff_log_delta",
        "eda_scr_log_delta",
        "eda_slope_norm",
        "eda_iqr_log_delta",
    ],
    "ppg": PPG_RAW_FEATURES + [
        "hr_bpm_delta",
        "ppg_band_delta",
        "ppg_absdiff_log_ratio",
    ],
    "acc": ACC_RAW_FEATURES,
}
SENSOR_FEATURE_MAP["eda_ppg"] = SENSOR_FEATURE_MAP["eda"] + SENSOR_FEATURE_MAP["ppg"]
SENSOR_FEATURE_MAP["all"] = (
    SENSOR_FEATURE_MAP["eda"] + SENSOR_FEATURE_MAP["ppg"] + SENSOR_FEATURE_MAP["acc"]
)


def compute_ibi_hrv_in_window(
    ibi_arr: np.ndarray, t0_ibi: float, t0_session: float, start_s: float, end_s: float
) -> dict[str, float]:
    """Compute RMSSD and SDNN (ms) from E4 IBI beats strictly inside [start_s, end_s)."""
    if ibi_arr is None or ibi_arr.ndim != 2 or len(ibi_arr) < 4 or not np.isfinite(t0_ibi):
        return {"ibi_rmssd_ms": 0.0, "ibi_sdnn_ms": 0.0, "ibi_valid_flag": 0.0}

    offset = t0_ibi - t0_session
    beat_rel_s = ibi_arr[:, 0] + offset
    mask = (beat_rel_s >= start_s) & (beat_rel_s < end_s)
    beats = ibi_arr[mask, 1]
    beats = beats[np.isfinite(beats) & (beats >= 0.30) & (beats <= 2.00)]
    if len(beats) < 4:
        return {"ibi_rmssd_ms": 0.0, "ibi_sdnn_ms": 0.0, "ibi_valid_flag": 0.0}

    beats_ms = beats * 1000.0
    diffs = np.diff(beats_ms)
    rmssd = float(np.sqrt(np.mean(diffs * diffs))) if len(diffs) >= 2 else 0.0
    sdnn = float(np.std(beats_ms, ddof=1))
    return {"ibi_rmssd_ms": rmssd, "ibi_sdnn_ms": sdnn, "ibi_valid_flag": 1.0}


def extract_nurse_window_features(
    session_data: dict,
    pred_time_rel_s: float,
    window_s: float = 180.0,
) -> dict[str, float]:
    """Extract strictly causal features at prediction time `t = pred_time_rel_s`.

    Only samples in `[max(0, t - 300), t)` are accessed. Samples at `>= t` are never touched.
    Primary observation window is `[t - window_s, t)` (default 180s).
    Also computes auxiliary 60s (`[t - 60, t)`) and 300s (`[t - 300, t)`) summary features.
    """
    if pred_time_rel_s < window_s:
        raise ValueError(f"pred_time_rel_s ({pred_time_rel_s}) must be >= window_s ({window_s})")

    t_end = float(pred_time_rel_s)
    t_start_180 = max(0.0, t_end - window_s)
    t_start_60 = max(0.0, t_end - 60.0)
    t_start_300 = max(0.0, t_end - 300.0)

    rates = session_data["rates"]
    fs_eda = float(rates["EDA"])
    fs_bvp = float(rates["BVP"])
    fs_acc = float(rates["ACC"])
    fs_hr = float(rates.get("HR", 1.0))

    eda_180 = native_slice(session_data["EDA"], t_start_180, t_end, fs_eda).reshape(-1)
    bvp_180 = native_slice(session_data["BVP"], t_start_180, t_end, fs_bvp).reshape(-1)
    acc_180 = native_slice(session_data["ACC"], t_start_180, t_end, fs_acc)

    out: dict[str, float] = {}
    for key, values in [("eda", eda_180), ("ppg", bvp_180), ("acc", acc_180)]:
        out.update(signal_quality(values, key))

    # 1. EDA features (180s primary)
    out.update(basic(eda_180, fs_eda, "eda_"))
    delta_eda = np.diff(eda_180)
    valid_delta = delta_eda[np.isfinite(delta_eda)]
    pos_sum = float(np.maximum(valid_delta, 0.0).sum()) if len(valid_delta) else 0.0
    neg_sum = float(np.maximum(-valid_delta, 0.0).sum()) if len(valid_delta) else 0.0
    out["eda_positive_diff_sum"] = pos_sum
    if np.isfinite(eda_180).all() and len(eda_180) >= 8:
        peaks, _ = find_peaks(eda_180, prominence=0.05, distance=max(1, int(fs_eda)))
        out["eda_scr_candidate_count"] = float(len(peaks))
    else:
        out["eda_scr_candidate_count"] = 0.0

    eda_clamped = np.maximum(np.nan_to_num(eda_180, nan=0.0), 0.0)
    eda_log = np.log1p(eda_clamped)
    out["eda_log_mean"] = float(np.mean(eda_log)) if len(eda_log) else 0.0
    out["eda_log_std"] = float(np.std(eda_log)) if len(eda_log) else 0.0
    t_grid_eda = np.arange(len(eda_log), dtype=float) / fs_eda
    if len(eda_log) >= 3:
        tc = t_grid_eda - t_grid_eda.mean()
        out["eda_log_slope"] = float(np.dot(tc, eda_log - eda_log.mean()) / max(np.dot(tc, tc), 1e-12))
    else:
        out["eda_log_slope"] = 0.0
    out["eda_rise_fraction"] = float(np.mean(valid_delta > 0.0)) if len(valid_delta) else 0.0
    out["eda_recovery_ratio"] = float(neg_sum / (pos_sum + 1e-6))

    # Multi-scale EDA: 60s and 300s
    eda_60 = native_slice(session_data["EDA"], t_start_60, t_end, fs_eda).reshape(-1)
    eda_prev120 = native_slice(session_data["EDA"], t_start_180, t_start_60, fs_eda).reshape(-1)
    m60 = float(np.nanmean(eda_60)) if len(eda_60) else out["eda_mean"]
    m120 = float(np.nanmean(eda_prev120)) if len(eda_prev120) else m60
    out["eda_60s_mean"] = m60
    out["eda_recent60_minus_prev120"] = float(np.log1p(max(m60, 0.0)) - np.log1p(max(m120, 0.0)))
    if len(eda_60) >= 3 and np.isfinite(eda_60).all():
        t60 = np.arange(len(eda_60), dtype=float) / fs_eda
        tc60 = t60 - t60.mean()
        out["eda_60s_slope"] = float(np.dot(tc60, eda_60 - eda_60.mean()) / max(np.dot(tc60, tc60), 1e-12))
    else:
        out["eda_60s_slope"] = 0.0

    eda_300 = native_slice(session_data["EDA"], t_start_300, t_end, fs_eda).reshape(-1)
    out["eda_300s_mean"] = float(np.nanmean(eda_300)) if len(eda_300) else out["eda_mean"]
    if len(eda_300) >= 3 and np.isfinite(eda_300).all():
        t300 = np.arange(len(eda_300), dtype=float) / fs_eda
        tc300 = t300 - t300.mean()
        out["eda_300s_slope"] = float(np.dot(tc300, eda_300 - eda_300.mean()) / max(np.dot(tc300, tc300), 1e-12))
    else:
        out["eda_300s_slope"] = 0.0

    # 2. BVP/PPG + HR + IBI features
    out.update(basic(bvp_180, fs_bvp, "ppg_"))
    # Avoid slow FFT on full 180s BVP if needed by using 32s Welch segments (already in spectrum())
    out["ppg_peak_hz"], out["ppg_band_fraction"] = spectrum(bvp_180, fs_bvp, 0.7, 3.5)
    est_hr = float(out["ppg_peak_hz"] * 60.0) if np.isfinite(out["ppg_peak_hz"]) else 80.0

    hr_arr = session_data.get("HR", np.empty((0,), dtype=float))
    t0_hr = session_data["start_timestamps"].get("HR", np.nan)
    t0_sess = session_data["start_ts"]
    hr_offset = (t0_hr - t0_sess) if (np.isfinite(t0_hr) and np.isfinite(t0_sess)) else 10.0

    if len(hr_arr) >= 4:
        hr_s_180 = max(0.0, t_start_180 - hr_offset)
        hr_e_180 = max(0.0, t_end - hr_offset)
        hr_180 = native_slice(hr_arr, hr_s_180, hr_e_180, fs_hr).reshape(-1) if hr_e_180 > hr_s_180 else np.empty((0,))
        hr_180 = hr_180[np.isfinite(hr_180)]
    else:
        hr_180 = np.empty((0,))

    if len(hr_180) >= 3:
        out["hr_mean_bpm"] = float(np.mean(hr_180))
        out["hr_std_bpm"] = float(np.std(hr_180))
        out["hr_range_bpm"] = float(np.ptp(hr_180))
        thr = np.arange(len(hr_180), dtype=float) / fs_hr
        tchr = thr - thr.mean()
        out["hr_slope_bpm_per_s"] = float(np.dot(tchr, hr_180 - hr_180.mean()) / max(np.dot(tchr, tchr), 1e-12))
        hr_60_len = min(len(hr_180), int(60 * fs_hr))
        hr_rec60 = float(np.mean(hr_180[-hr_60_len:]))
        hr_prev = float(np.mean(hr_180[:-hr_60_len])) if len(hr_180) > hr_60_len else hr_rec60
        out["hr_60s_mean_bpm"] = hr_rec60
        out["hr_delta_recent60_bpm"] = hr_rec60 - hr_prev
    else:
        out["hr_mean_bpm"] = est_hr
        out["hr_std_bpm"] = 0.0
        out["hr_range_bpm"] = 0.0
        out["hr_slope_bpm_per_s"] = 0.0
        out["hr_60s_mean_bpm"] = est_hr
        out["hr_delta_recent60_bpm"] = 0.0

    if len(hr_arr) >= 4:
        hr_s_300 = max(0.0, t_start_300 - hr_offset)
        hr_e_300 = max(0.0, t_end - hr_offset)
        hr_300 = native_slice(hr_arr, hr_s_300, hr_e_300, fs_hr).reshape(-1) if hr_e_300 > hr_s_300 else np.empty((0,))
        hr_300 = hr_300[np.isfinite(hr_300)]
        out["hr_300s_mean_bpm"] = float(np.mean(hr_300)) if len(hr_300) else out["hr_mean_bpm"]
    else:
        out["hr_300s_mean_bpm"] = out["hr_mean_bpm"]

    out.update(
        compute_ibi_hrv_in_window(
            session_data.get("IBI", np.empty((0, 2))),
            session_data["start_timestamps"].get("IBI", np.nan),
            t0_sess,
            t_start_180,
            t_end,
        )
    )

    # 3. Orientation-invariant ACC features (180s primary + 60s/300s motion activity)
    out.update(extract_orientation_invariant_acc_features(acc_180, fs_acc))

    acc_60 = native_slice(session_data["ACC"], t_start_60, t_end, fs_acc)
    if acc_60.ndim == 2 and len(acc_60) >= 4 and np.isfinite(acc_60).all():
        out["acc_60s_motion_change"] = float(np.mean(np.linalg.norm(np.diff(acc_60, axis=0), axis=1)) * fs_acc)
    else:
        out["acc_60s_motion_change"] = out.get("acc_motion_change_per_s", 0.0)

    acc_300 = native_slice(session_data["ACC"], t_start_300, t_end, fs_acc)
    if acc_300.ndim == 2 and len(acc_300) >= 4 and np.isfinite(acc_300).all():
        out["acc_300s_motion_change"] = float(np.mean(np.linalg.norm(np.diff(acc_300, axis=0), axis=1)) * fs_acc)
    else:
        out["acc_300s_motion_change"] = out.get("acc_motion_change_per_s", 0.0)

    # Contact & validity check: require finite signals and non-zero skin contact (EDA mean >= 0.005 uS)
    eda_contact_ok = bool(np.isfinite(out["eda_mean"]) and out["eda_mean"] >= 0.005)
    out["qc_eda_valid"] = bool(out["qc_eda_valid"] and eda_contact_ok)
    out["qc_common_valid"] = bool(out["qc_eda_valid"] and out["qc_ppg_valid"] and out["qc_acc_valid"])

    return out
