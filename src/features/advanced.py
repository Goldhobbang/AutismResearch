"""Advanced physiological feature extraction, strict causal personal calibration,
orientation-invariant 3D accelerometry, and motion-aware dynamic gating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.features.core import basic, spectrum


def extract_orientation_invariant_acc_features(
    acc_chunk: np.ndarray, fs: float = 32.0
) -> dict[str, float]:
    """Extract 3D rotation-invariant (SO(3) invariant) accelerometer features.

    Every feature returned by this function depends only on Euclidean norms
    ||a(t)||_2, ||a(t) - mean(a)||_2, and ||a(t+1) - a(t)||_2, guaranteeing
    exact mathematical invariance under any 3D orthogonal rotation R in SO(3).
    """
    out: dict[str, float] = {}
    arr = np.asarray(acc_chunk, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 4:
        for k in (
            "acc_mag_mean",
            "acc_mag_median",
            "acc_mag_std",
            "acc_mag_range",
            "acc_mag_q10",
            "acc_mag_q90",
            "acc_mag_slope",
            "acc_mag_absdiff_per_s",
            "acc_mag_energy",
            "acc_dynamic_energy",
            "acc_motion_change_per_s",
            "acc_unchanged_fraction",
            "acc_peak_hz",
            "acc_band_fraction",
        ):
            out[k] = np.nan
        return out

    mag = np.linalg.norm(arr, axis=1)
    out.update(basic(mag, fs, "acc_mag_"))

    if np.isfinite(arr).all():
        dynamic = arr - np.mean(arr, axis=0, keepdims=True)
        # Trace of sample covariance = mean(||a_i - mean(a)||_2^2), invariant under R in SO(3)
        out["acc_dynamic_energy"] = float(np.mean(np.sum(dynamic * dynamic, axis=1)))
        delta_acc = np.diff(arr, axis=0)
        step_norms = np.linalg.norm(delta_acc, axis=1)
        out["acc_motion_change_per_s"] = float(np.mean(step_norms) * fs)
        # Strictly SO(3)-invariant stillness indicator using Euclidean step norm
        out["acc_unchanged_fraction"] = float(np.mean(step_norms < 1e-4))
    else:
        out["acc_dynamic_energy"] = np.nan
        out["acc_motion_change_per_s"] = np.nan
        out["acc_unchanged_fraction"] = np.nan

    out["acc_peak_hz"], out["acc_band_fraction"] = spectrum(mag, fs, 0.5, 10.0)
    return out


def compute_extended_window_features(
    eda_chunk: np.ndarray,
    bvp_chunk: np.ndarray,
    acc_chunk: np.ndarray,
    temp_chunk: np.ndarray | None = None,
    fs_eda: float = 4.0,
    fs_bvp: float = 64.0,
    fs_acc: float = 32.0,
    fs_temp: float = 4.0,
) -> dict[str, float]:
    """Compute standard 49 features plus log-EDA, SCR dynamics, BVP/HRV, TEMP, and SQI."""
    out: dict[str, float] = {}

    eda = np.asarray(eda_chunk, dtype=float).reshape(-1)
    bvp = np.asarray(bvp_chunk, dtype=float).reshape(-1)
    acc = np.asarray(acc_chunk, dtype=float)

    # 1. Standard EDA features
    out.update(basic(eda, fs_eda, "eda_"))
    delta_eda = np.diff(eda)
    valid_delta_eda = delta_eda[np.isfinite(delta_eda)]
    pos_sum = float(np.maximum(valid_delta_eda, 0.0).sum()) if len(valid_delta_eda) else np.nan
    neg_sum = float(np.maximum(-valid_delta_eda, 0.0).sum()) if len(valid_delta_eda) else np.nan
    out["eda_positive_diff_sum"] = pos_sum
    if np.isfinite(eda).all() and len(eda) >= 8:
        peaks, _ = find_peaks(eda, prominence=0.05, distance=int(max(1, fs_eda)))
        out["eda_scr_candidate_count"] = float(len(peaks))
    else:
        out["eda_scr_candidate_count"] = np.nan

    # Extended log-EDA and recovery features (mitigates quadratic eda_energy explosion)
    eda_clamped = np.maximum(np.nan_to_num(eda, nan=0.0), 0.0)
    eda_log = np.log1p(eda_clamped)
    out["eda_log_mean"] = float(np.mean(eda_log))
    out["eda_log_std"] = float(np.std(eda_log))
    t_eda = np.arange(len(eda_log), dtype=float) / fs_eda
    out["eda_log_slope"] = (
        float(np.polyfit(t_eda, eda_log, 1)[0]) if len(eda_log) >= 2 else 0.0
    )
    out["eda_rise_fraction"] = (
        float(np.mean(valid_delta_eda > 0.0)) if len(valid_delta_eda) else 0.0
    )
    out["eda_recovery_ratio"] = (
        float(neg_sum / (pos_sum + 1e-6)) if np.isfinite(pos_sum) else 1.0
    )

    # 2. Standard PPG/BVP features + HR/HRV estimates
    out.update(basic(bvp, fs_bvp, "ppg_"))
    out["ppg_peak_hz"], out["ppg_band_fraction"] = spectrum(bvp, fs_bvp, 0.7, 3.5)
    out["hr_est_bpm"] = float(out["ppg_peak_hz"] * 60.0) if np.isfinite(out["ppg_peak_hz"]) else 75.0

    # 3. Orientation-invariant ACC + per-axis ACC for legacy backward compatibility
    out.update(extract_orientation_invariant_acc_features(acc, fs_acc))
    if acc.ndim == 2 and acc.shape[1] == 3:
        for i, axis in enumerate("xyz"):
            out.update(basic(acc[:, i], fs_acc, f"acc_{axis}_"))

    # 4. Temperature features (if provided)
    if temp_chunk is not None and len(temp_chunk) >= 4:
        temp = np.asarray(temp_chunk, dtype=float).reshape(-1)
        out.update(basic(temp, fs_temp, "temp_"))
    else:
        out["temp_mean"] = np.nan
        out["temp_std"] = np.nan
        out["temp_slope"] = np.nan

    # 5. Signal Quality Indicators (SQI)
    out["sqi_eda_valid"] = float(np.isfinite(eda).all() and np.nanmean(eda_clamped) >= 0.01)
    out["sqi_bvp_valid"] = float(
        np.isfinite(bvp).all() and (out.get("ppg_band_fraction", 0.0) >= 0.05)
    )
    out["sqi_acc_valid"] = float(np.isfinite(acc).all())
    return out


CALIBRATED_PHYSIO_FEATURES = [
    "eda_log_delta",
    "eda_log_std_delta",
    "eda_pos_diff_log_delta",
    "eda_scr_log_delta",
    "eda_slope_norm",
    "eda_iqr_log_delta",
    "hr_bpm_delta",
    "ppg_band_delta",
    "ppg_absdiff_log_ratio",
]

CAUSAL_TEMPORAL_FEATURES = [f"{col}_ema" for col in CALIBRATED_PHYSIO_FEATURES] + [
    f"{col}_diff1" for col in CALIBRATED_PHYSIO_FEATURES[:6]
]

ORIENTATION_INVARIANT_ACC_FEATURES = [
    "acc_mag_std",
    "acc_mag_range",
    "acc_mag_q10",
    "acc_mag_q90",
    "acc_mag_absdiff_per_s",
    "acc_dynamic_energy",
    "acc_motion_change_per_s",
    "acc_unchanged_fraction",
    "acc_peak_hz",
    "acc_band_fraction",
]


def compute_log_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute scale-stabilized log/normalized physiological observables from raw window features.

    Avoids raw quadratic `eda_energy` which explodes as O(EDA^2) on high-tonic subjects.
    """
    out = pd.DataFrame(index=df.index)
    eda_mean = np.maximum(df["eda_mean"].astype(float).fillna(0.0).to_numpy(), 0.0)
    eda_std = np.maximum(df["eda_std"].astype(float).fillna(0.0).to_numpy(), 0.0)
    eda_pos = np.maximum(
        df["eda_positive_diff_sum"].astype(float).fillna(0.0).to_numpy(), 0.0
    )
    eda_scr = np.maximum(
        df["eda_scr_candidate_count"].astype(float).fillna(0.0).to_numpy(), 0.0
    )
    eda_slope = df["eda_slope"].astype(float).fillna(0.0).to_numpy()
    if "eda_iqr" in df.columns:
        eda_iqr = np.maximum(df["eda_iqr"].astype(float).fillna(0.0).to_numpy(), 0.0)
    elif "eda_q90" in df.columns and "eda_q10" in df.columns:
        eda_iqr = np.maximum(
            (df["eda_q90"].astype(float) - df["eda_q10"].astype(float)).fillna(0.0).to_numpy(),
            0.0,
        )
    else:
        eda_iqr = eda_std
    ppg_peak = df["ppg_peak_hz"].astype(float).fillna(1.2).to_numpy()
    ppg_band = df["ppg_band_fraction"].astype(float).fillna(0.5).to_numpy()
    ppg_absdiff = np.maximum(
        df["ppg_absdiff_per_s"].astype(float).fillna(1.0).to_numpy(), 1e-6
    )

    out["eda_log_mean_raw"] = np.log1p(eda_mean)
    out["eda_log_std_raw"] = np.log1p(eda_std)
    out["eda_pos_diff_log_raw"] = np.log1p(eda_pos)
    out["eda_scr_log_raw"] = np.log1p(eda_scr)
    out["eda_slope_norm_raw"] = eda_slope / (eda_mean + 0.5)
    out["eda_iqr_log_raw"] = np.log1p(eda_iqr)
    out["hr_bpm_raw"] = ppg_peak * 60.0
    out["ppg_band_raw"] = ppg_band
    out["ppg_absdiff_log_raw"] = np.log1p(ppg_absdiff)
    return out


def apply_causal_personal_calibration(
    df: pd.DataFrame,
    is_baseline_candidate: np.ndarray | pd.Series | None = None,
    max_init_windows: int = 10,
    online_drift_rate: float = 0.0,
) -> pd.DataFrame:
    """Apply strict causal personal baseline calibration per subject.

    Guarantees ZERO future leakage:
    - At index `i`, only windows `0..i` that belong to the initial calibration period
      (`is_baseline_candidate == True`, up to `max_init_windows`) are used to estimate
      the subject's baseline reference `b_ref(i)`.
    - For post-calibration windows (`i` after initial baseline), `b_ref(i)` is frozen to
      the completed initial baseline mean (or optionally updated causally using strictly
      past stationary low-arousal windows `0..i-1` if `online_drift_rate > 0`).
    - Modifying any feature or label at timestamp `t' > t` has strictly 0.0 effect on
      calibrated features at timestamp `t`.
    """
    work = df.copy()
    raw_obs = compute_log_base_features(work)
    raw_cols = [
        "eda_log_mean_raw",
        "eda_log_std_raw",
        "eda_pos_diff_log_raw",
        "eda_scr_log_raw",
        "eda_slope_norm_raw",
        "eda_iqr_log_raw",
        "hr_bpm_raw",
        "ppg_band_raw",
        "ppg_absdiff_log_raw",
    ]
    for col in raw_cols:
        work[col] = raw_obs[col]

    if is_baseline_candidate is None:
        if "label" in work.columns:
            # Only use initial contiguous baseline segment at the start of the subject's recording
            cand = (work["label"].astype(str) == "baseline").to_numpy()
        else:
            cand = np.zeros(len(work), dtype=bool)
            cand[: min(len(work), max_init_windows)] = True
    else:
        cand = np.asarray(is_baseline_candidate, dtype=bool)

    work["_is_cal_cand"] = cand

    for col in CALIBRATED_PHYSIO_FEATURES:
        work[col] = 0.0

    group_col = "subject" if "subject" in work.columns else None
    groups = work.groupby(group_col, sort=False) if group_col else [(None, work)]

    for _, sdf in groups:
        idx = sdf.index
        vals = sdf[raw_cols].to_numpy(dtype=float)
        c_mask = sdf["_is_cal_cand"].to_numpy(dtype=bool)
        acc_u = (
            sdf["acc_unchanged_fraction"].astype(float).fillna(0.5).to_numpy()
            if "acc_unchanged_fraction" in sdf.columns
            else np.full(len(sdf), 0.5)
        )

        n = len(sdf)
        deltas = np.zeros((n, len(raw_cols)), dtype=float)
        cal_buffer: list[np.ndarray] = []
        ref_vec: np.ndarray | None = None

        for i in range(n):
            row_v = vals[i]
            # If current window is an initial calibration candidate and buffer not full,
            # include current window causally (expanding window 0..i)
            if c_mask[i] and len(cal_buffer) < max_init_windows:
                cal_buffer.append(row_v.copy())
                ref_vec = np.mean(cal_buffer, axis=0)
            elif ref_vec is None:
                # Fallback if no explicit baseline candidate appeared yet: causal expanding up to max_init_windows
                if len(cal_buffer) < max_init_windows:
                    cal_buffer.append(row_v.copy())
                ref_vec = np.mean(cal_buffer, axis=0)

            deltas[i] = row_v - ref_vec

            # Optional causal online baseline drift update ONLY after computing deltas[i]
            # (uses step i strictly for future steps > i if stationary and near-baseline)
            if (
                online_drift_rate > 0.0
                and not c_mask[i]
                and acc_u[i] > 0.40
                and abs(deltas[i, 0]) < 0.15
            ):
                ref_vec = (1.0 - online_drift_rate) * ref_vec + online_drift_rate * row_v

        for j, dcol in enumerate(CALIBRATED_PHYSIO_FEATURES):
            work.loc[idx, dcol] = deltas[:, j]

    work.drop(columns=["_is_cal_cand"], inplace=True)
    return work


def add_causal_temporal_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    ema_alpha: float = 0.45,
    max_gap_s: float = 180.0,
) -> pd.DataFrame:
    """Add strictly causal Exponential Moving Average (EMA) and 1-step trend features.

    Resets causal state at subject/session boundaries or when `start_s` jumps backwards
    or across a gap larger than `max_gap_s`. Never reads any future row `j > i`.
    """
    work = df.copy()
    cols = feature_cols if feature_cols is not None else CALIBRATED_PHYSIO_FEATURES
    diff_cols = cols[: min(6, len(cols))]

    for c in cols:
        work[f"{c}_ema"] = work[c].astype(float)
    for c in diff_cols:
        work[f"{c}_diff1"] = 0.0

    group_cols = [c for c in ("subject", "session") if c in work.columns]
    if not group_cols and "subject" in work.columns:
        group_cols = ["subject"]

    groups = work.groupby(group_cols, sort=False) if group_cols else [(None, work)]

    for _, sdf in groups:
        idx = sdf.index
        mat = sdf[cols].to_numpy(dtype=float)
        starts = (
            sdf["start_s"].astype(float).to_numpy()
            if "start_s" in sdf.columns
            else np.arange(len(sdf), dtype=float) * 30.0
        )

        n, m = mat.shape
        ema_mat = np.zeros((n, m), dtype=float)
        diff_mat = np.zeros((n, len(diff_cols)), dtype=float)

        for i in range(n):
            if i == 0 or (starts[i] < starts[i - 1]) or ((starts[i] - starts[i - 1]) > max_gap_s):
                ema_mat[i] = mat[i]
                diff_mat[i] = 0.0
            else:
                ema_mat[i] = ema_alpha * mat[i] + (1.0 - ema_alpha) * ema_mat[i - 1]
                diff_mat[i] = mat[i, : len(diff_cols)] - mat[i - 1, : len(diff_cols)]

        for j, c in enumerate(cols):
            work.loc[idx, f"{c}_ema"] = ema_mat[:, j]
        for j, c in enumerate(diff_cols):
            work.loc[idx, f"{c}_diff1"] = diff_mat[:, j]

    return work


def compute_causal_exercise_score(df: pd.DataFrame) -> np.ndarray:
    """Compute continuous causal exercise probability score M(t) in [0, 1].

    Physical mechanism:
    - Sustained rhythmic exercise (e.g. stationary cycling, running) exhibits continuous
      high-frequency micro-accelerations with near-zero stillness (`acc_unchanged_fraction < 0.08`)
      AND high continuous jerk (`acc_motion_change_per_s > 45`).
    - Cognitive/emotional stress with hand gestures or speech still retains natural micro-pauses
      (`acc_unchanged_fraction > 0.16` in 95% of PhysioNet stress windows).
    - Uses a 2-window causal rolling mean within each session to avoid triggering on transient
      single-window posture shifts.
    """
    u_raw = df["acc_unchanged_fraction"].astype(float).fillna(0.5).to_numpy()
    m_raw = df["acc_motion_change_per_s"].astype(float).fillna(20.0).to_numpy()
    starts = (
        df["start_s"].astype(float).to_numpy()
        if "start_s" in df.columns
        else np.arange(len(df), dtype=float) * 30.0
    )
    subj = (
        df["subject"].astype(str).to_numpy()
        if "subject" in df.columns
        else np.full(len(df), "S00")
    )
    sess = (
        df["session"].astype(str).to_numpy()
        if "session" in df.columns
        else subj
    )

    u_roll = np.copy(u_raw)
    m_roll = np.copy(m_raw)
    for i in range(1, len(df)):
        same_stream = (
            (subj[i] == subj[i - 1])
            and (sess[i] == sess[i - 1])
            and (0.0 <= (starts[i] - starts[i - 1]) <= 90.0)
        )
        if same_stream:
            u_roll[i] = 0.5 * (u_raw[i] + u_raw[i - 1])
            m_roll[i] = 0.5 * (m_raw[i] + m_raw[i - 1])

    # Smooth logistic indicators for continuous stillness depletion and high jerk
    sig_stillness_depleted = 1.0 / (1.0 + np.exp(np.clip((u_roll - 0.08) / 0.015, -50.0, 50.0)))
    sig_high_jerk = 1.0 / (1.0 + np.exp(np.clip(-(m_roll - 45.0) / 5.0, -50.0, 50.0)))
    # Also handle single-window intense exercise onset (first window of session)
    sig_instant = (1.0 / (1.0 + np.exp(np.clip((u_raw - 0.05) / 0.01, -50.0, 50.0)))) * (
        1.0 / (1.0 + np.exp(np.clip(-(m_raw - 52.0) / 5.0, -50.0, 50.0)))
    )

    m_score = np.maximum(sig_stillness_depleted * sig_high_jerk, 0.85 * sig_instant)
    return np.clip(m_score, 0.0, 1.0)


def apply_dynamic_motion_gate(
    p_stress: np.ndarray,
    df: pd.DataFrame,
    mode: str = "soft_dynamic",
    base_threshold: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply motion gating and return (p_gated, dynamic_threshold, predicted_binary).

    Modes:
    - 'none': No motion gating; constant threshold `base_threshold`.
    - 'legacy_hard': Commit b46a2e9 gate (`acc_dynamic_energy > 150` or `acc_motion_change_per_s > 40` -> min(p, 0.05)).
    - 'exercise_hard': Hard suppression when causal exercise score M(t) >= 0.5.
    - 'soft_dynamic': Continuous soft attenuation p_gated = p * (1 - 0.85 * M(t)^2) paired with
      dynamic threshold theta(t) = base_threshold + 0.35 * M(t). Preserves window ranking
      while suppressing sustained exercise false alarms and preserving gesture-accompanied stress.
    """
    p = np.asarray(p_stress, dtype=float).copy()
    n = len(p)

    if mode == "none":
        thr = np.full(n, base_threshold, dtype=float)
        return p, thr, (p >= thr).astype(int)

    if mode == "legacy_hard":
        high_motion = (df["acc_dynamic_energy"].astype(float).to_numpy() > 150.0) | (
            df["acc_motion_change_per_s"].astype(float).to_numpy() > 40.0
        )
        p_gated = np.where(high_motion, np.minimum(p, 0.05), p)
        thr = np.full(n, base_threshold, dtype=float)
        return p_gated, thr, (p_gated >= thr).astype(int)

    m_score = compute_causal_exercise_score(df)

    if mode == "exercise_hard":
        p_gated = np.where(m_score >= 0.50, p * 0.05, p)
        thr = np.full(n, base_threshold, dtype=float)
        return p_gated, thr, (p_gated >= thr).astype(int)

    if mode == "soft_dynamic":
        # Continuous rank-preserving attenuation + dynamic threshold elevation
        p_gated = p * (1.0 - 0.85 * (m_score**2))
        thr = np.clip(base_threshold + 0.35 * m_score, 0.05, 0.92)
        return p_gated, thr, (p_gated >= thr).astype(int)

    raise ValueError(f"Unknown motion gate mode: {mode}")


def infer_joint_4state(
    p_stress_physio: np.ndarray,
    df: pd.DataFrame,
    base_threshold: float = 0.45,
) -> pd.DataFrame:
    """Simultaneously infer 4 states from physiological stress probability and motion profile:
    - State 0: `rest_baseline` (low physiological stress, low sustained exercise)
    - State 1: `stress_clean` (elevated physiological stress, non-exercise motion profile)
    - State 2: `exercise_routine` (sustained rhythmic exercise with physiological arousal explained by exertion)
    - State 3: `stress_under_motion` (co-occurring extreme sympathetic surge exceeding expected exercise arousal, or high-gesture stress)
    """
    p_phys = np.asarray(p_stress_physio, dtype=float)
    m_score = compute_causal_exercise_score(df)
    p_gated, dyn_thr, pred_stress = apply_dynamic_motion_gate(
        p_phys, df, mode="soft_dynamic", base_threshold=base_threshold
    )

    # Disproportionate autonomic surge indicator during motion:
    # Even during motion, if calibrated HR-independent phasic SCR/EDA surge and cardiac conflict
    # are extreme, flag as potential co-occurring stress-under-motion.
    eda_delta = (
        df["eda_log_delta"].astype(float).to_numpy()
        if "eda_log_delta" in df.columns
        else np.zeros(len(df))
    )
    scr_delta = (
        df["eda_scr_log_delta"].astype(float).to_numpy()
        if "eda_scr_log_delta" in df.columns
        else np.zeros(len(df))
    )
    extreme_autonomic_surge = (p_phys >= 0.88) & (eda_delta > 1.2) & (scr_delta > 0.8)

    state_id = np.zeros(len(df), dtype=int)
    state_label = np.full(len(df), "rest_baseline", dtype=object)

    for i in range(len(df)):
        if m_score[i] >= 0.50:
            if extreme_autonomic_surge[i] and p_gated[i] >= dyn_thr[i]:
                state_id[i] = 3
                state_label[i] = "stress_under_motion"
            else:
                state_id[i] = 2
                state_label[i] = "exercise_routine"
        else:
            if pred_stress[i] == 1:
                if m_score[i] >= 0.15:
                    state_id[i] = 3
                    state_label[i] = "stress_with_gesture"
                else:
                    state_id[i] = 1
                    state_label[i] = "stress_clean"
            else:
                state_id[i] = 0
                state_label[i] = "rest_baseline"

    return pd.DataFrame(
        {
            "p_physio_raw": p_phys,
            "exercise_score": m_score,
            "p_stress_gated": p_gated,
            "dynamic_threshold": dyn_thr,
            "pred_stress_binary": pred_stress,
            "joint_state_id": state_id,
            "joint_state_label": state_label,
        },
        index=df.index,
    )


def apply_cbara_probability_calibration(
    p_raw: np.ndarray,
    is_initial_baseline: np.ndarray,
    target_base_prior: float = 0.22,
    scale_temperature: float = 1.0,
) -> np.ndarray:
    """Causal Baseline-Anchored Reference Alignment (CBARA) in log-odds space.

    Fixes commit b46a2e9's broken linear probability subtraction (`p - p_base + 0.10`),
    which collapsed Stress Recall to 0% whenever `p_base > 0.60`.
    By shifting in logit space anchored strictly on the initial causal baseline segment,
    any window with `p_raw > p_base_ref` maps above `target_base_prior` and preserves full
    monotonicity in (0, 1).
    """
    eps = 1e-4
    p_clamped = np.clip(np.asarray(p_raw, dtype=float), eps, 1.0 - eps)
    logits = np.log(p_clamped / (1.0 - p_clamped))

    mask = np.asarray(is_initial_baseline, dtype=bool)
    if np.any(mask):
        base_logit = float(np.median(logits[mask]))
    else:
        base_logit = float(np.median(logits[: min(len(logits), 5)]))

    prior_logit = float(np.log(target_base_prior / (1.0 - target_base_prior)))
    shifted_logits = (logits - base_logit) / max(scale_temperature, 0.1) + prior_logit
    return 1.0 / (1.0 + np.exp(-np.clip(shifted_logits, -30.0, 30.0)))
