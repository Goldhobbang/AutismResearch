"""Study B: Impact of Initial Baseline State on Personal Calibration Robustness & Failure.

Analyzes initial baseline diagnostic features, compares 5 calibration estimators
(Cold-Start, Mean, Median, Low-Motion-Only, Trimmed-Mean, Fallback Policy),
and conducts stress tests (Setup-period vs True Baseline, First-Half vs Second-Half).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.advanced import (
    CALIBRATED_PHYSIO_FEATURES,
    compute_log_base_features,
)


def compute_baseline_diagnostics(
    sdf: pd.DataFrame,
    label_col: str = "label",
    start_col: str = "start_s",
    end_col: str = "end_s",
) -> dict[str, float]:
    """Compute diagnostic metrics from the initial baseline window ONLY.

    Zero future information: only uses baseline windows from the recording.
    """
    raw_obs = compute_log_base_features(sdf)
    b_mask = (sdf[label_col] == "baseline").to_numpy()
    if not np.any(b_mask):
        return {}

    b_idx = sdf.index[b_mask]
    b_raw = raw_obs.loc[b_idx]
    b_sdf = sdf.loc[b_idx]

    eda_mean = float(b_sdf["eda_mean"].mean())
    eda_std = float(b_sdf["eda_mean"].std()) if len(b_sdf) > 1 else 0.0
    eda_slope = float(b_sdf["eda_slope"].mean()) if "eda_slope" in b_sdf else 0.0

    hr_mean = float(b_raw["hr_bpm_raw"].mean())
    hr_std = float(b_raw["hr_bpm_raw"].std()) if len(b_raw) > 1 else 0.0

    # Motion diagnostics
    jerk_mean = float(b_sdf["acc_motion_change_per_s"].mean()) if "acc_motion_change_per_s" in b_sdf else 0.0
    unchanged_mean = float(b_sdf["acc_unchanged_fraction"].mean()) if "acc_unchanged_fraction" in b_sdf else 0.5
    dynamic_energy_mean = float(b_sdf["acc_dynamic_energy"].mean()) if "acc_dynamic_energy" in b_sdf else 0.0

    # Trend: first half vs second half difference
    n = len(b_idx)
    if n >= 2:
        half = n // 2
        eda_half1 = float(b_sdf.iloc[:half]["eda_mean"].mean())
        eda_half2 = float(b_sdf.iloc[half:]["eda_mean"].mean())
        eda_drift = eda_half2 - eda_half1
        hr_half1 = float(b_raw.iloc[:half]["hr_bpm_raw"].mean())
        hr_half2 = float(b_raw.iloc[half:]["hr_bpm_raw"].mean())
        hr_drift = hr_half2 - hr_half1
    else:
        eda_drift = 0.0
        hr_drift = 0.0

    # Signal quality
    valid_fraction = float(np.mean(b_sdf["qc_common_valid"])) if "qc_common_valid" in b_sdf else 1.0

    return {
        "base_eda_mean": eda_mean,
        "base_eda_std": eda_std,
        "base_eda_slope": eda_slope,
        "base_hr_mean": hr_mean,
        "base_hr_std": hr_std,
        "base_jerk_mean": jerk_mean,
        "base_unchanged_mean": unchanged_mean,
        "base_dynamic_energy": dynamic_energy_mean,
        "base_eda_drift_half2_minus_half1": eda_drift,
        "base_hr_drift_half2_minus_half1": hr_drift,
        "base_valid_fraction": valid_fraction,
        "n_baseline_windows": n,
    }


def calibrate_with_method(
    df: pd.DataFrame,
    method: str = "mean",
    max_duration_s: float = 300.0,
    subject_col: str = "subject",
    label_col: str = "label",
    start_col: str = "start_s",
    end_col: str = "end_s",
    fallback_thresholds: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Calibrate dataframe using specified estimator method.

    Methods:
    - 'cold_start': no calibration (raw features)
    - 'mean': standard sample mean of baseline windows in [t0, t0+T)
    - 'median': coordinate-wise median of baseline windows in [t0, t0+T)
    - 'low_motion': mean of baseline windows with acc_motion_change_per_s <= 35.0
    - 'trimmed_mean': 20% trimmed mean of baseline windows
    - 'adaptive_fallback': if baseline is flagged as hyper-aroused or unstable, fallback to cold-start
    """
    raw_obs = compute_log_base_features(df)
    raw_cols = list(raw_obs.columns)
    work = df.copy()

    for col in CALIBRATED_PHYSIO_FEATURES:
        work[col] = 0.0

    meta = {}
    for sid, sdf in work.groupby(subject_col, sort=False):
        idx = sdf.index
        vals = raw_obs.loc[idx, raw_cols].to_numpy(dtype=float)
        b_mask = (sdf[label_col] == "baseline").to_numpy()

        if method == "cold_start" or not np.any(b_mask):
            work.loc[idx, CALIBRATED_PHYSIO_FEATURES] = vals
            meta[sid] = {"method": method, "calibrated": False, "reason": "cold_start_or_no_baseline"}
            continue

        b_starts = sdf.loc[b_mask, start_col].to_numpy(dtype=float)
        t_base_start = float(np.min(b_starts))
        t_cutoff = t_base_start + max_duration_s
        cal_win_mask = b_mask & (sdf[end_col].to_numpy(dtype=float) <= t_cutoff + 1e-4)

        if not np.any(cal_win_mask):
            first_idx = sdf.index[b_mask][0]
            cal_vals = raw_obs.loc[[first_idx], raw_cols].to_numpy(dtype=float)
            cal_sdf = sdf.loc[[first_idx]]
        else:
            cal_vals = raw_obs.loc[sdf.index[cal_win_mask], raw_cols].to_numpy(dtype=float)
            cal_sdf = sdf.loc[sdf.index[cal_win_mask]]

        diag = compute_baseline_diagnostics(sdf)

        # Check fallback condition if adaptive_fallback
        fallback_triggered = False
        fallback_reason = ""
        if method == "adaptive_fallback" and fallback_thresholds is not None:
            if diag.get("base_eda_mean", 0.0) > fallback_thresholds.get("max_eda_mean", 15.0):
                fallback_triggered = True
                fallback_reason = "extreme_initial_eda"
            elif diag.get("base_jerk_mean", 0.0) > fallback_thresholds.get("max_jerk", 45.0):
                fallback_triggered = True
                fallback_reason = "high_initial_motion"

        if fallback_triggered:
            work.loc[idx, CALIBRATED_PHYSIO_FEATURES] = vals
            meta[sid] = {"method": method, "calibrated": False, "reason": fallback_reason, **diag}
            continue

        # Compute reference vector based on method
        if method in ("mean", "adaptive_fallback"):
            ref = np.mean(cal_vals, axis=0)
        elif method == "median":
            ref = np.median(cal_vals, axis=0)
        elif method == "low_motion":
            if "acc_motion_change_per_s" in cal_sdf:
                motion = cal_sdf["acc_motion_change_per_s"].to_numpy(dtype=float)
                quiet_mask = motion <= 35.0
                if np.any(quiet_mask):
                    ref = np.mean(cal_vals[quiet_mask], axis=0)
                else:
                    ref = np.median(cal_vals, axis=0)
            else:
                ref = np.mean(cal_vals, axis=0)
        elif method == "trimmed_mean":
            if len(cal_vals) >= 4:
                ref = trim_mean(cal_vals, proportiontocut=0.20, axis=0)
            else:
                ref = np.median(cal_vals, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        deltas = vals - ref
        for j, c in enumerate(CALIBRATED_PHYSIO_FEATURES):
            work.loc[idx, c] = deltas[:, j]

        meta[sid] = {
            "method": method,
            "calibrated": True,
            "reason": "calibrated_successfully",
            "t_base_start": t_base_start,
            "t_cal_end": float(cal_sdf[end_col].max()),
            **diag,
        }

    return work, meta


def run_baseline_estimator_comparison(
    w_df: pd.DataFrame,
    p_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare calibration estimators across WESAD and PhysioNet on identical evaluation sets."""
    methods = ["cold_start", "mean", "median", "low_motion", "trimmed_mean", "adaptive_fallback"]
    fallback_thr = {"max_eda_mean": 15.0, "max_jerk": 45.0}

    # Model: LR_Calib_Physio
    pipe_lr = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", RobustScaler()),
            ("clf", LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )

    per_subj_rows = []
    summary_rows = []

    # 1. WESAD 15-subject LOSO
    w_subjects = sorted(w_df["subject"].unique())
    # Fix evaluation set: windows strictly starting >= 300s (so 300s calibration window is excluded from evaluation)
    for m in methods:
        w_cal, w_meta = calibrate_with_method(w_df, method=m, max_duration_s=300, fallback_thresholds=fallback_thr)

        for test_s in w_subjects:
            tr = w_cal[w_cal["subject"] != test_s]
            te = w_cal[w_cal["subject"] == test_s]

            y_tr = (tr["label"] == "stress").astype(int).to_numpy()
            est = clone(pipe_lr)
            est.fit(tr[CALIBRATED_PHYSIO_FEATURES], y_tr)

            te_eval = te[
                (te["label"] == "stress")
                | ((te["label"] == "baseline") & (te["start_s"] >= 300))
            ].copy()

            if len(te_eval) == 0 or te_eval["label"].nunique() < 2:
                continue

            y_te = (te_eval["label"] == "stress").astype(int).to_numpy()
            p_te = est.predict_proba(te_eval[CALIBRATED_PHYSIO_FEATURES])[:, 1]
            pred_te = (p_te >= 0.50).astype(int)

            auroc = roc_auc_score(y_te, p_te)
            ap = average_precision_score(y_te, p_te)
            rec = recall_score(y_te, pred_te, zero_division=0)
            prec = precision_score(y_te, pred_te, zero_division=0)
            f1_s = f1_score(y_te, pred_te, zero_division=0)
            base_fpr = float(np.mean(pred_te[y_te == 0] == 1)) if np.any(y_te == 0) else 0.0

            res = {
                "dataset": "WESAD",
                "method": m,
                "subject": test_s,
                "auroc": auroc,
                "ap": ap,
                "stress_recall": rec,
                "precision": prec,
                "stress_f1": f1_s,
                "baseline_fpr": base_fpr,
                **w_meta[test_s],
            }
            per_subj_rows.append(res)

    # 2. PhysioNet 17-subject External Transfer
    # Fit model on WESAD standard mean calibration
    w_cal_std, _ = calibrate_with_method(w_df, method="mean", max_duration_s=300)
    y_w = (w_cal_std["label"] == "stress").astype(int).to_numpy()
    fitted_lr = clone(pipe_lr)
    fitted_lr.fit(w_cal_std[CALIBRATED_PHYSIO_FEATURES], y_w)

    p_st = p_df[p_df["label"].isin(["baseline", "stress"])].copy().sort_values(["subject", "start_s"]).reset_index(drop=True)

    for m in methods:
        p_cal, p_meta = calibrate_with_method(p_st, method=m, max_duration_s=90, fallback_thresholds=fallback_thr)

        for sid, sdf in p_cal.groupby("subject"):
            t_base_start = p_meta[sid].get("t_base_start", sdf["start_s"].min())
            # Evaluation set: windows starting >= t_base_start + 90s, plus stress
            te_eval = sdf[
                (sdf["label"] == "stress")
                | ((sdf["label"] == "baseline") & (sdf["start_s"] >= t_base_start + 90))
            ].copy()

            if len(te_eval) == 0 or te_eval["label"].nunique() < 2:
                continue

            y_te = (te_eval["label"] == "stress").astype(int).to_numpy()
            p_te = fitted_lr.predict_proba(te_eval[CALIBRATED_PHYSIO_FEATURES])[:, 1]
            pred_te = (p_te >= 0.45).astype(int)

            auroc = roc_auc_score(y_te, p_te)
            ap = average_precision_score(y_te, p_te)
            rec = recall_score(y_te, pred_te, zero_division=0)
            prec = precision_score(y_te, pred_te, zero_division=0)
            f1_s = f1_score(y_te, pred_te, zero_division=0)
            base_fpr = float(np.mean(pred_te[y_te == 0] == 1)) if np.any(y_te == 0) else 0.0

            res = {
                "dataset": "PhysioNet",
                "method": m,
                "subject": sid,
                "auroc": auroc,
                "ap": ap,
                "stress_recall": rec,
                "precision": prec,
                "stress_f1": f1_s,
                "baseline_fpr": base_fpr,
                **p_meta[sid],
            }
            per_subj_rows.append(res)

    ps_df = pd.DataFrame(per_subj_rows)

    # Compute summary aggregates
    for (dset, meth), grp in ps_df.groupby(["dataset", "method"], sort=False):
        summary_rows.append(
            {
                "dataset": dset,
                "method": meth,
                "n_subjects": len(grp),
                "mean_auroc": float(grp["auroc"].mean()),
                "median_auroc": float(grp["auroc"].median()),
                "worst_auroc": float(grp["auroc"].min()),
                "worst_subject": str(grp.loc[grp["auroc"].idxmin(), "subject"]),
                "mean_ap": float(grp["ap"].mean()),
                "mean_stress_recall": float(grp["stress_recall"].mean()),
                "worst_recall": float(grp["stress_recall"].min()),
                "mean_precision": float(grp["precision"].mean()),
                "mean_stress_f1": float(grp["stress_f1"].mean()),
                "mean_baseline_fpr": float(grp["baseline_fpr"].mean()),
                "fallback_count": int(np.sum(~grp["calibrated"])) if "calibrated" in grp else 0,
            }
        )

    sum_df = pd.DataFrame(summary_rows)
    return ps_df, sum_df


def run_setup_period_stress_test(
    p_tag: pd.DataFrame,
    w_calibrated_model: Pipeline,
    out_dir: Path,
) -> pd.DataFrame:
    """Stress Test: Compare calibrating on Setup period [0, t0) vs True Baseline [t0, t1).

    Simulates the failure mode in legacy b46a2e9 where watch strapping motion was used as baseline!
    """
    raw_root = ROOT / "data/raw/physionet_v1/Wearable_Dataset/STRESS"
    segs = pd.read_csv(ROOT / "data/processed/physionet/verified_segments_tag_aligned.csv")

    from src.loaders.physionet_e4 import load_e4_csv, compute_physionet_window_features

    results = []
    for sid in p_tag["subject"].unique():
        sdir = raw_root / sid
        if not (sdir / "EDA.csv").exists():
            continue

        _, fs_eda, eda = load_e4_csv(sdir / "EDA.csv")
        _, fs_bvp, bvp = load_e4_csv(sdir / "BVP.csv")
        _, fs_acc, acc = load_e4_csv(sdir / "ACC.csv")

        # Get setup segment and baseline segment
        setup_seg = segs[(segs["subject"] == sid) & (segs["task"] == "Pre_Protocol_Setup")]
        base_seg = segs[(segs["subject"] == sid) & (segs["task"] == "Baseline")]
        if setup_seg.empty or base_seg.empty:
            continue

        t_setup_end = float(setup_seg.iloc[0]["end_s"])
        t_base_start = float(base_seg.iloc[0]["start_s"])
        t_base_end = float(base_seg.iloc[0]["end_s"])

        # Compute setup reference (first 60s of setup: [0, 60))
        if t_setup_end >= 60:
            setup_feats = compute_physionet_window_features(
                eda[: int(60 * fs_eda)],
                bvp[: int(60 * fs_bvp)],
                acc[: int(60 * fs_acc)],
            )
            setup_df = pd.DataFrame([setup_feats])
            setup_ref = compute_log_base_features(setup_df).iloc[0].to_numpy(dtype=float)
        else:
            continue

        # Compute true baseline reference (first 60s of true baseline: [t_base_start, t_base_start+60))
        base_feats = compute_physionet_window_features(
            eda[int(t_base_start * fs_eda) : int((t_base_start + 60) * fs_eda)],
            bvp[int(t_base_start * fs_bvp) : int((t_base_start + 60) * fs_bvp)],
            acc[int(t_base_start * fs_acc) : int((t_base_start + 60) * fs_acc)],
        )
        base_df = pd.DataFrame([base_feats])
        true_ref = compute_log_base_features(base_df).iloc[0].to_numpy(dtype=float)

        # Evaluate on the true evaluation windows of this subject (tag_aligned windows >= t_base_start + 60)
        s_eval = p_tag[(p_tag["subject"] == sid) & (
            (p_tag["label"] == "stress") | ((p_tag["label"] == "baseline") & (p_tag["start_s"] >= t_base_start + 60))
        )].copy()

        if len(s_eval) == 0 or s_eval["label"].nunique() < 2:
            continue

        y_te = (s_eval["label"] == "stress").astype(int).to_numpy()
        s_raw_obs = compute_log_base_features(s_eval)
        raw_mat = s_raw_obs.to_numpy(dtype=float)

        # 1. Prediction using True Baseline calibration
        delta_true = raw_mat - true_ref
        df_true = pd.DataFrame(delta_true, columns=CALIBRATED_PHYSIO_FEATURES)
        p_true = w_calibrated_model.predict_proba(df_true)[:, 1]
        auc_true = roc_auc_score(y_te, p_true)
        rec_true = recall_score(y_te, (p_true >= 0.45).astype(int), zero_division=0)
        fpr_true = float(np.mean((p_true[y_te == 0] >= 0.45).astype(int))) if np.any(y_te == 0) else 0.0

        # 2. Prediction using Corrupted Setup calibration (Watch strapping motion used as baseline!)
        delta_setup = raw_mat - setup_ref
        df_setup = pd.DataFrame(delta_setup, columns=CALIBRATED_PHYSIO_FEATURES)
        p_setup = w_calibrated_model.predict_proba(df_setup)[:, 1]
        auc_setup = roc_auc_score(y_te, p_setup)
        rec_setup = recall_score(y_te, (p_setup >= 0.45).astype(int), zero_division=0)
        fpr_setup = float(np.mean((p_setup[y_te == 0] >= 0.45).astype(int))) if np.any(y_te == 0) else 0.0

        results.append({
            "subject": sid,
            "auroc_true_baseline_calib": auc_true,
            "recall_true_baseline_calib": rec_true,
            "fpr_true_baseline_calib": fpr_true,
            "auroc_setup_motion_calib": auc_setup,
            "recall_setup_motion_calib": rec_setup,
            "fpr_setup_motion_calib": fpr_setup,
            "auroc_drop_due_to_setup_motion": auc_true - auc_setup,
            "recall_drop_due_to_setup_motion": rec_true - rec_setup,
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "setup_motion_stress_test.csv", index=False)
    return res_df


def main():
    run_id = f"run_{int(time.time())}"
    out_dir = ROOT / "results" / "baseline_robustness" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting Study B: Initial Baseline Robustness & Failure Analysis (run_id: {run_id})...")

    # Load datasets
    w_raw = pd.read_csv(ROOT / "data/processed/wesad_features.csv")
    w_bin = w_raw[(w_raw["y"].isin([0, 1])) & (w_raw["qc_common_valid"] == True)].copy()
    w_bin["label"] = np.where(w_bin["y"] == 1, "stress", "baseline")
    w_bin = w_bin.sort_values(["subject", "start_s"]).reset_index(drop=True)

    p_tag = pd.read_csv(ROOT / "results/phase4/physionet_windows_tag_aligned_18sub.csv")

    # 1. Compare estimators
    print("Running estimator comparison (Cold-start, Mean, Median, Low-motion, Trimmed-mean, Fallback)...")
    ps_df, sum_df = run_baseline_estimator_comparison(w_bin, p_tag, out_dir)

    ps_df.to_csv(out_dir / "baseline_estimators_per_subject.csv", index=False)
    sum_df.to_csv(out_dir / "baseline_estimators_summary.csv", index=False)

    # 2. Setup-period stress test
    print("Running Setup-period motion corruption stress test on PhysioNet...")
    # Model fitted on WESAD mean calibration
    w_cal_std, _ = calibrate_with_method(w_bin, method="mean", max_duration_s=300)
    y_w = (w_cal_std["label"] == "stress").astype(int).to_numpy()
    pipe_lr = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", RobustScaler()),
            ("clf", LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )
    pipe_lr.fit(w_cal_std[CALIBRATED_PHYSIO_FEATURES], y_w)

    stress_test_df = run_setup_period_stress_test(p_tag, pipe_lr, out_dir)

    # Latest aliases
    latest_dir = ROOT / "results" / "baseline_robustness" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    ps_df.to_csv(latest_dir / "baseline_estimators_per_subject.csv", index=False)
    sum_df.to_csv(latest_dir / "baseline_estimators_summary.csv", index=False)
    stress_test_df.to_csv(latest_dir / "setup_motion_stress_test.csv", index=False)

    print("\n=== STUDY B: ESTIMATOR COMPARISON SUMMARY ===")
    print(sum_df[["dataset", "method", "mean_auroc", "median_auroc", "worst_auroc", "worst_subject", "mean_stress_recall", "worst_recall", "mean_baseline_fpr"]].to_string(index=False))

    print("\n=== STUDY B: SETUP MOTION STRESS TEST SUMMARY ===")
    print(f"Mean AUROC with True Baseline: {stress_test_df['auroc_true_baseline_calib'].mean():.4f}")
    print(f"Mean AUROC with Setup Motion:   {stress_test_df['auroc_setup_motion_calib'].mean():.4f}")
    print(f"Average AUROC degradation:     {stress_test_df['auroc_drop_due_to_setup_motion'].mean():.4f}")
    print(f"Average Recall degradation:    {stress_test_df['recall_drop_due_to_setup_motion'].mean():.4f}")


if __name__ == "__main__":
    main()
