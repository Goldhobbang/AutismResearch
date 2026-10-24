"""Study A: Rigorous personal calibration duration comparison on WESAD and PhysioNet.

Compares durations under strictly fair common evaluation window sets and operational sets.
No test-label leakage: calibration vector is strictly computed from [t0, t0+T).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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


def bootstrap_ci_diff(diffs: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return (float(arr[0]) if len(arr) else np.nan, float(arr[0]) if len(arr) else np.nan)
    rng = np.random.default_rng(seed)
    boots = np.mean(rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def calibrate_by_duration(
    df: pd.DataFrame,
    duration_s: float,
    subject_col: str = "subject",
    label_col: str = "label",
    start_col: str = "start_s",
    end_col: str = "end_s",
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Calibrate dataframe using only baseline windows within [t_base_start, t_base_start + duration_s).

    For duration_s == 0: uncalibrated raw features (deltas = raw_obs - cohort_dummy_median).
    For duration_s > 0: personal delta = raw_obs - personal_baseline_mu.
    Returns calibrated DataFrame and metadata per subject.
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

        if duration_s == 0 or not np.any(b_mask):
            # Uncalibrated: use 0.0 delta or dummy zero baseline
            ref = np.zeros(vals.shape[1], dtype=float)
            work.loc[idx, CALIBRATED_PHYSIO_FEATURES] = vals
            meta[sid] = {
                "duration_s": 0,
                "n_cal_windows": 0,
                "cal_start_s": float(sdf.loc[idx, start_col].min()),
                "cal_end_s": float(sdf.loc[idx, start_col].min()),
            }
            continue

        b_starts = sdf.loc[b_mask, start_col].to_numpy(dtype=float)
        b_ends = sdf.loc[b_mask, end_col].to_numpy(dtype=float)
        t_base_start = float(np.min(b_starts))
        t_cutoff = t_base_start + duration_s

        # Strict containment: window raw signal must be entirely within [t_base_start, t_cutoff]
        cal_win_mask = b_mask & (sdf[end_col].to_numpy(dtype=float) <= t_cutoff + 1e-4)

        if np.any(cal_win_mask):
            cal_vals = raw_obs.loc[sdf.index[cal_win_mask], raw_cols].to_numpy(dtype=float)
            ref = np.mean(cal_vals, axis=0)
            n_win = int(np.sum(cal_win_mask))
            cal_actual_end = float(np.max(sdf.loc[cal_win_mask, end_col]))
        else:
            # Fallback to earliest baseline window if duration is smaller than window size
            first_b_idx = sdf.index[b_mask][0]
            ref = raw_obs.loc[first_b_idx, raw_cols].to_numpy(dtype=float)
            n_win = 1
            cal_actual_end = float(sdf.loc[first_b_idx, end_col])

        deltas = vals - ref
        for j, c in enumerate(CALIBRATED_PHYSIO_FEATURES):
            work.loc[idx, c] = deltas[:, j]

        meta[sid] = {
            "duration_s": duration_s,
            "n_cal_windows": n_win,
            "cal_start_s": t_base_start,
            "cal_end_s": cal_actual_end,
        }

    return work, meta


def run_wesad_duration_study(
    w_df: pd.DataFrame,
    durations: list[int] = [0, 60, 180, 300, 600],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run WESAD 15-subject LOSO evaluation across calibration durations."""
    subjects = sorted(w_df["subject"].unique())

    models = {
        "LR_Calib_Physio": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", RobustScaler()),
                ("clf", LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, random_state=42)),
            ]
        ),
        "RF_Calib_Physio": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42)),
            ]
        ),
    }

    # Evaluate multiple common sets + operational set
    eval_sets = [
        ("Common_Set_10m", 600),
        ("Common_Set_5m", 300),
        ("Common_Set_3m", 180),
        ("Operational_PostCal", -1),  # starts immediately after calibration end
    ]

    per_subj_rows = []
    summary_rows = []

    for set_name, offset_s in eval_sets:
        # Determine valid durations for this evaluation set
        valid_durations = [d for d in durations if (offset_s == -1 or d <= offset_s)]

        for m_name, pipe_tmpl in models.items():
            dur_results = {}
            for dur in valid_durations:
                # 1. Calibrate entire dataset using duration T
                cal_df, meta = calibrate_by_duration(w_df, dur)

                subj_metrics = []
                for test_s in subjects:
                    tr = cal_df[cal_df["subject"] != test_s]
                    te = cal_df[cal_df["subject"] == test_s]

                    y_tr = (tr["label"] == "stress").astype(int).to_numpy()
                    est = clone(pipe_tmpl)
                    est.fit(tr[CALIBRATED_PHYSIO_FEATURES], y_tr)

                    # Filter test evaluation windows
                    t_base_start = meta[test_s]["cal_start_s"]
                    if offset_s == -1:
                        # Operational: start strictly after this condition's calibration end
                        eval_start = meta[test_s]["cal_end_s"]
                    else:
                        eval_start = t_base_start + offset_s

                    # Test set: baseline windows strictly starting >= eval_start, plus all stress windows
                    te_eval = te[
                        (te["label"] == "stress")
                        | ((te["label"] == "baseline") & (te["start_s"] >= eval_start))
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
                    f1_stress = f1_score(y_te, pred_te, zero_division=0)
                    f1_macro = f1_score(y_te, pred_te, average="macro", zero_division=0)
                    base_fpr = float(np.mean(pred_te[y_te == 0] == 1)) if np.any(y_te == 0) else 0.0

                    # Latency & monitoring coverage
                    total_dur = float(te["end_s"].max() - te["start_s"].min())
                    eval_dur = float(len(te_eval) * 15.0)  # approximate window duration
                    coverage_pct = (eval_dur / max(total_dur, 1.0)) * 100.0
                    wait_latency_s = float(meta[test_s]["cal_end_s"] - meta[test_s]["cal_start_s"])

                    res = {
                        "dataset": "WESAD",
                        "eval_protocol": set_name,
                        "model": m_name,
                        "duration_s": dur,
                        "subject": test_s,
                        "n_eval_windows": len(te_eval),
                        "n_stress_windows": int(np.sum(y_te == 1)),
                        "n_base_windows": int(np.sum(y_te == 0)),
                        "auroc": auroc,
                        "ap": ap,
                        "stress_recall": rec,
                        "precision": prec,
                        "stress_f1": f1_stress,
                        "macro_f1": f1_macro,
                        "baseline_fpr": base_fpr,
                        "wait_latency_s": wait_latency_s,
                        "coverage_pct": coverage_pct,
                    }
                    subj_metrics.append(res)
                    per_subj_rows.append(res)

                dur_results[dur] = pd.DataFrame(subj_metrics)

            # Summarize across durations for this eval set and model
            baseline_df = dur_results[0]
            for dur in valid_durations:
                cur_df = dur_results[dur]
                if len(cur_df) == 0:
                    continue

                # Paired differences vs duration 0
                paired = cur_df.merge(baseline_df, on="subject", suffixes=("", "_base0"))
                f1_diff = (paired["stress_f1"] - paired["stress_f1_base0"]).to_numpy()
                auc_diff = (paired["auroc"] - paired["auroc_base0"]).to_numpy()
                fpr_diff = (paired["baseline_fpr"] - paired["baseline_fpr_base0"]).to_numpy()

                ci_f1_low, ci_f1_high = bootstrap_ci_diff(f1_diff)
                ci_auc_low, ci_auc_high = bootstrap_ci_diff(auc_diff)

                summary_rows.append(
                    {
                        "dataset": "WESAD",
                        "eval_protocol": set_name,
                        "model": m_name,
                        "duration_s": dur,
                        "n_subjects": len(cur_df),
                        "mean_auroc": float(cur_df["auroc"].mean()),
                        "median_auroc": float(cur_df["auroc"].median()),
                        "worst_auroc": float(cur_df["auroc"].min()),
                        "mean_ap": float(cur_df["ap"].mean()),
                        "mean_stress_recall": float(cur_df["stress_recall"].mean()),
                        "worst_recall": float(cur_df["stress_recall"].min()),
                        "mean_precision": float(cur_df["precision"].mean()),
                        "mean_stress_f1": float(cur_df["stress_f1"].mean()),
                        "mean_macro_f1": float(cur_df["macro_f1"].mean()),
                        "mean_baseline_fpr": float(cur_df["baseline_fpr"].mean()),
                        "mean_wait_latency_s": float(cur_df["wait_latency_s"].mean()),
                        "mean_coverage_pct": float(cur_df["coverage_pct"].mean()),
                        "paired_f1_diff_vs_0m": float(np.mean(f1_diff)),
                        "paired_f1_diff_ci_low": ci_f1_low,
                        "paired_f1_diff_ci_high": ci_f1_high,
                        "paired_auc_diff_vs_0m": float(np.mean(auc_diff)),
                        "paired_auc_diff_ci_low": ci_auc_low,
                        "paired_auc_diff_ci_high": ci_auc_high,
                        "paired_fpr_diff_vs_0m": float(np.mean(fpr_diff)),
                    }
                )

    return pd.DataFrame(per_subj_rows), pd.DataFrame(summary_rows)


def run_physionet_duration_study(
    p_tag: pd.DataFrame,
    w_calibrated_models: dict[str, Pipeline],
    durations: list[int] = [0, 60, 90, 120],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run PhysioNet 17-subject external transfer across calibration durations."""
    p_st = p_tag[p_tag["label"].isin(["baseline", "stress"])].copy().sort_values(["subject", "start_s"]).reset_index(drop=True)

    eval_sets = [
        ("Common_Set_120s", 120),
        ("Common_Set_90s", 90),
        ("Common_Set_60s", 60),
        ("Operational_PostCal", -1),
    ]

    per_subj_rows = []
    summary_rows = []

    for set_name, offset_s in eval_sets:
        valid_durations = [d for d in durations if (offset_s == -1 or d <= offset_s)]

        for m_name, fitted_pipe in w_calibrated_models.items():
            dur_results = {}
            for dur in valid_durations:
                cal_df, meta = calibrate_by_duration(p_st, dur)

                subj_metrics = []
                for sid, sdf in cal_df.groupby("subject"):
                    t_base_start = meta[sid]["cal_start_s"]
                    if offset_s == -1:
                        eval_start = meta[sid]["cal_end_s"]
                    else:
                        eval_start = t_base_start + offset_s

                    te_eval = sdf[
                        (sdf["label"] == "stress")
                        | ((sdf["label"] == "baseline") & (sdf["start_s"] >= eval_start))
                    ].copy()

                    if len(te_eval) == 0 or te_eval["label"].nunique() < 2:
                        continue

                    y_te = (te_eval["label"] == "stress").astype(int).to_numpy()
                    p_te = fitted_pipe.predict_proba(te_eval[CALIBRATED_PHYSIO_FEATURES])[:, 1]
                    pred_te = (p_te >= 0.45).astype(int)

                    auroc = roc_auc_score(y_te, p_te)
                    ap = average_precision_score(y_te, p_te)
                    rec = recall_score(y_te, pred_te, zero_division=0)
                    prec = precision_score(y_te, pred_te, zero_division=0)
                    f1_stress = f1_score(y_te, pred_te, zero_division=0)
                    f1_macro = f1_score(y_te, pred_te, average="macro", zero_division=0)
                    base_fpr = float(np.mean(pred_te[y_te == 0] == 1)) if np.any(y_te == 0) else 0.0

                    wait_latency_s = float(meta[sid]["cal_end_s"] - meta[sid]["cal_start_s"])
                    total_dur = float(sdf["end_s"].max() - sdf["start_s"].min())
                    coverage_pct = (float(len(te_eval) * 30.0) / max(total_dur, 1.0)) * 100.0

                    res = {
                        "dataset": "PhysioNet",
                        "eval_protocol": set_name,
                        "model": m_name,
                        "duration_s": dur,
                        "subject": sid,
                        "n_eval_windows": len(te_eval),
                        "n_stress_windows": int(np.sum(y_te == 1)),
                        "n_base_windows": int(np.sum(y_te == 0)),
                        "auroc": auroc,
                        "ap": ap,
                        "stress_recall": rec,
                        "precision": prec,
                        "stress_f1": f1_stress,
                        "macro_f1": f1_macro,
                        "baseline_fpr": base_fpr,
                        "wait_latency_s": wait_latency_s,
                        "coverage_pct": coverage_pct,
                    }
                    subj_metrics.append(res)
                    per_subj_rows.append(res)

                dur_results[dur] = pd.DataFrame(subj_metrics)

            baseline_df = dur_results[0]
            for dur in valid_durations:
                cur_df = dur_results[dur]
                if len(cur_df) == 0:
                    continue

                paired = cur_df.merge(baseline_df, on="subject", suffixes=("", "_base0"))
                f1_diff = (paired["stress_f1"] - paired["stress_f1_base0"]).to_numpy()
                auc_diff = (paired["auroc"] - paired["auroc_base0"]).to_numpy()
                fpr_diff = (paired["baseline_fpr"] - paired["baseline_fpr_base0"]).to_numpy()

                ci_f1_low, ci_f1_high = bootstrap_ci_diff(f1_diff)
                ci_auc_low, ci_auc_high = bootstrap_ci_diff(auc_diff)

                summary_rows.append(
                    {
                        "dataset": "PhysioNet",
                        "eval_protocol": set_name,
                        "model": m_name,
                        "duration_s": dur,
                        "n_subjects": len(cur_df),
                        "mean_auroc": float(cur_df["auroc"].mean()),
                        "median_auroc": float(cur_df["auroc"].median()),
                        "worst_auroc": float(cur_df["auroc"].min()),
                        "mean_ap": float(cur_df["ap"].mean()),
                        "mean_stress_recall": float(cur_df["stress_recall"].mean()),
                        "worst_recall": float(cur_df["stress_recall"].min()),
                        "mean_precision": float(cur_df["precision"].mean()),
                        "mean_stress_f1": float(cur_df["stress_f1"].mean()),
                        "mean_macro_f1": float(cur_df["macro_f1"].mean()),
                        "mean_baseline_fpr": float(cur_df["baseline_fpr"].mean()),
                        "mean_wait_latency_s": float(cur_df["wait_latency_s"].mean()),
                        "mean_coverage_pct": float(cur_df["coverage_pct"].mean()),
                        "paired_f1_diff_vs_0m": float(np.mean(f1_diff)),
                        "paired_f1_diff_ci_low": ci_f1_low,
                        "paired_f1_diff_ci_high": ci_f1_high,
                        "paired_auc_diff_vs_0m": float(np.mean(auc_diff)),
                        "paired_auc_diff_ci_low": ci_auc_low,
                        "paired_auc_diff_ci_high": ci_auc_high,
                        "paired_fpr_diff_vs_0m": float(np.mean(fpr_diff)),
                    }
                )

    return pd.DataFrame(per_subj_rows), pd.DataFrame(summary_rows)


def main():
    run_id = f"run_{int(time.time())}"
    out_dir = ROOT / "results" / "calibration_duration" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting Study A: Calibration Duration Comparison (run_id: {run_id})...")

    # 1. Load WESAD
    print("Loading WESAD features...")
    w_raw = pd.read_csv(ROOT / "data/processed/wesad_features.csv")
    w_bin = w_raw[(w_raw["y"].isin([0, 1])) & (w_raw["qc_common_valid"] == True)].copy()
    w_bin["label"] = np.where(w_bin["y"] == 1, "stress", "baseline")
    w_bin = w_bin.sort_values(["subject", "start_s"]).reset_index(drop=True)

    print("Running WESAD calibration duration experiments...")
    w_subj_df, w_sum_df = run_wesad_duration_study(w_bin)

    # 2. Fit WESAD models on standard 300s calibration for PhysioNet evaluation
    w_cal300, _ = calibrate_by_duration(w_bin, duration_s=300)
    y_w = (w_cal300["label"] == "stress").astype(int).to_numpy()

    fitted_pipes = {}
    pipe_lr = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", RobustScaler()),
            ("clf", LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )
    pipe_lr.fit(w_cal300[CALIBRATED_PHYSIO_FEATURES], y_w)
    fitted_pipes["LR_Calib_Physio"] = pipe_lr

    pipe_rf = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42)),
        ]
    )
    pipe_rf.fit(w_cal300[CALIBRATED_PHYSIO_FEATURES], y_w)
    fitted_pipes["RF_Calib_Physio"] = pipe_rf

    # 3. Load PhysioNet Tag-Aligned
    print("Loading PhysioNet tag-aligned windows...")
    p_tag = pd.read_csv(ROOT / "results/phase4/physionet_windows_tag_aligned_18sub.csv")

    print("Running PhysioNet calibration duration experiments...")
    p_subj_df, p_sum_df = run_physionet_duration_study(p_tag, fitted_pipes)

    # Save outputs
    all_subj = pd.concat([w_subj_df, p_subj_df], ignore_index=True)
    all_sum = pd.concat([w_sum_df, p_sum_df], ignore_index=True)

    all_subj.to_csv(out_dir / "calibration_duration_per_subject.csv", index=False)
    all_sum.to_csv(out_dir / "calibration_duration_summary.csv", index=False)

    # Also save latest alias
    latest_dir = ROOT / "results" / "calibration_duration" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    all_subj.to_csv(latest_dir / "calibration_duration_per_subject.csv", index=False)
    all_sum.to_csv(latest_dir / "calibration_duration_summary.csv", index=False)

    print("\n=== STUDY A: SUMMARY OF CALIBRATION DURATION RESULTS ===")
    print(all_sum[["dataset", "eval_protocol", "model", "duration_s", "mean_auroc", "worst_auroc", "mean_stress_f1", "mean_baseline_fpr", "mean_wait_latency_s", "paired_f1_diff_vs_0m"]].to_string(index=False))


if __name__ == "__main__":
    main()
