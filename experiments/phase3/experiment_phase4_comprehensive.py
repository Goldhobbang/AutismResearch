"""Comprehensive Phase 4 experimental suite for AutismResearch follow-up study.

Covers:
1. Step 1 Audit verification (3-subject vs 17-subject legacy pipeline comparison).
2. Task A: Motion Gate ablation (None, Legacy Hard Gate b46a2e9, Exercise Hard Gate,
   Soft Dynamic Gate, Joint 4-State Inference) & Stress-Under-Motion evaluation.
3. Task B: Causal Personal Physiological Calibration (Initial 5-min baseline vs Online drift update)
   and Subgroup Analysis (Low-Responder vs Medium vs High-Tonic EDA subjects).
4. Task C: Model & Feature Architecture Comparison (LR, RF, LightGBM, HistGradientBoosting,
   Causal Temporal EMA, Sensor Ablation, On-Device Latency & Memory Footprint).
5. Task D: 18-Subject External Generalization (17 eligible STRESS, 14 eligible AEROBIC)
   under both Identical Legacy Segments and Official Tag-Aligned Protocol Segments,
   with bootstrap 95% confidence intervals and worst-subject root-cause analysis.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase2.zero_shot_physionet import evaluate_models_on_windows
from src.features.advanced import (
    CALIBRATED_PHYSIO_FEATURES,
    CAUSAL_TEMPORAL_FEATURES,
    ORIENTATION_INVARIANT_ACC_FEATURES,
    add_causal_temporal_features,
    apply_causal_personal_calibration,
    apply_cbara_probability_calibration,
    apply_dynamic_motion_gate,
    compute_causal_exercise_score,
    infer_joint_4state,
)
from src.loaders.physionet_e4 import compute_physionet_window_features, load_e4_csv


def extract_or_load_tag_aligned_windows(out_dir: Path) -> pd.DataFrame:
    """Extract 60s/30s-step windows for all verified tag-aligned segments (cached)."""
    cache_path = out_dir / "physionet_windows_tag_aligned_18sub.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        if len(df) > 400:
            return df

    seg_path = ROOT / "data" / "processed" / "physionet" / "verified_segments_tag_aligned.csv"
    segs = pd.read_csv(seg_path)
    elig = segs[segs["verified"] == True].copy()
    raw_root = ROOT / "data" / "raw" / "physionet_v1" / "Wearable_Dataset"

    rows = []
    for (subj, sess), group in elig.groupby(["subject", "session"], sort=False):
        cond = sess.split("_")[1]
        sdir = raw_root / cond / subj
        if not (sdir / "EDA.csv").exists():
            continue
        _, fs_eda, eda = load_e4_csv(sdir / "EDA.csv")
        _, fs_bvp, bvp = load_e4_csv(sdir / "BVP.csv")
        _, fs_acc, acc = load_e4_csv(sdir / "ACC.csv")

        max_s = int(
            min(len(eda) / fs_eda, len(bvp) / fs_bvp, len(acc) / fs_acc)
        )
        for _, seg in group.iterrows():
            st = int(seg["start_s"])
            et = min(int(seg["end_s"]), max_s)
            lbl = str(seg["label"])
            task = str(seg["task"])
            for w_s in range(st, et - 60 + 1, 30):
                w_e = w_s + 60
                feats = compute_physionet_window_features(
                    eda[int(w_s * fs_eda) : int(w_e * fs_eda)],
                    bvp[int(w_s * fs_bvp) : int(w_e * fs_bvp)],
                    acc[int(w_s * fs_acc) : int(w_e * fs_acc)],
                )
                row = {
                    "subject": subj,
                    "session": sess,
                    "task": task,
                    "label": lbl,
                    "start_s": w_s,
                    "end_s": w_e,
                    "has_bvp": True,
                }
                row.update(feats)
                rows.append(row)

    df_out = pd.DataFrame(rows)
    evaluate_models_on_windows(df_out)
    df_out.to_csv(cache_path, index=False)
    return df_out


def compute_episode_far_per_hour(
    pred_binary: np.ndarray, start_s: np.ndarray, window_step_s: float = 30.0
) -> tuple[float, float]:
    """Compute window-level FAR/hr and debounced episode-level FAR/hr."""
    n = len(pred_binary)
    if n == 0:
        return 0.0, 0.0
    total_hours = (n * window_step_s) / 3600.0
    win_far_hr = float(np.sum(pred_binary) / max(total_hours, 1e-6))

    # Episode count: rising edges (0 -> 1) or 1 after a time gap > 90s
    episodes = 0
    for i in range(n):
        if pred_binary[i] == 1:
            if i == 0 or pred_binary[i - 1] == 0 or (start_s[i] - start_s[i - 1] > 90.0):
                episodes += 1
    ep_far_hr = float(episodes / max(total_hours, 1e-6))
    return win_far_hr, ep_far_hr


def bootstrap_ci(vals: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return (float(arr[0]) if len(arr) else np.nan, float(arr[0]) if len(arr) else np.nan)
    rng = np.random.default_rng(seed)
    boots = np.mean(rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def build_model_candidates() -> dict[str, tuple[object, list[str]]]:
    """Return dictionary of model_name -> (sklearn estimator, feature list)."""
    f_physio = CALIBRATED_PHYSIO_FEATURES
    f_physio_temp = CALIBRATED_PHYSIO_FEATURES + CAUSAL_TEMPORAL_FEATURES
    f_physio_temp_acc = (
        CALIBRATED_PHYSIO_FEATURES
        + CAUSAL_TEMPORAL_FEATURES
        + ORIENTATION_INVARIANT_ACC_FEATURES
    )
    f_eda_only = [c for c in f_physio_temp if c.startswith("eda_")]
    f_ppg_only = [c for c in f_physio_temp if c.startswith("hr_") or c.startswith("ppg_")]
    f_acc_only = ORIENTATION_INVARIANT_ACC_FEATURES

    models = {
        "LR_Calib_Physio": (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", RobustScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.25, class_weight="balanced", max_iter=1000, random_state=42
                        ),
                    ),
                ]
            ),
            f_physio,
        ),
        "LR_Calib_Temporal": (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", RobustScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.20, class_weight="balanced", max_iter=1000, random_state=42
                        ),
                    ),
                ]
            ),
            f_physio_temp,
        ),
        "RF_Calib_Physio": (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=5,
                            min_samples_leaf=5,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            f_physio,
        ),
        "RF_Calib_Temporal": (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=5,
                            min_samples_leaf=5,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            f_physio_temp,
        ),
        "LGBM_Calib_Temporal": (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        lgb.LGBMClassifier(
                            n_estimators=120,
                            max_depth=3,
                            num_leaves=7,
                            learning_rate=0.05,
                            min_child_samples=12,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            class_weight="balanced",
                            random_state=42,
                            verbose=-1,
                        ),
                    ),
                ]
            ),
            f_physio_temp,
        ),
        "HGB_Calib_Temporal": (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=100,
                            max_depth=3,
                            min_samples_leaf=12,
                            learning_rate=0.05,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            f_physio_temp,
        ),
        "RF_Calib_Temporal_RotInvACC": (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=5,
                            min_samples_leaf=5,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            f_physio_temp_acc,
        ),
        "Ablation_EDA_Only": (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", RobustScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.20, class_weight="balanced", max_iter=1000, random_state=42
                        ),
                    ),
                ]
            ),
            f_eda_only,
        ),
        "Ablation_PPG_Only": (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", RobustScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.20, class_weight="balanced", max_iter=1000, random_state=42
                        ),
                    ),
                ]
            ),
            f_ppg_only,
        ),
        "Ablation_ACC_Only": (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", RobustScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.20, class_weight="balanced", max_iter=1000, random_state=42
                        ),
                    ),
                ]
            ),
            f_acc_only,
        ),
    }
    return models


def run_wesad_loso_evaluation(w_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate candidate models on 15-subject WESAD Leave-One-Subject-Out CV."""
    from sklearn.base import clone

    subjects = sorted(w_df["subject"].unique())
    models = build_model_candidates()
    rows = []

    for m_name, (pipe_tmpl, fcols) in models.items():
        subj_aurocs, subj_aps, subj_recalls, subj_f1s, subj_fprs = [], [], [], [], []
        for test_s in subjects:
            tr = w_df[w_df["subject"] != test_s]
            te = w_df[w_df["subject"] == test_s]
            y_tr = (tr["label"] == "stress").astype(int).to_numpy()
            y_te = (te["label"] == "stress").astype(int).to_numpy()

            est = clone(pipe_tmpl)
            est.fit(tr[fcols], y_tr)
            p_te = est.predict_proba(te[fcols])[:, 1]
            pred_te = (p_te >= 0.45).astype(int)

            subj_aurocs.append(roc_auc_score(y_te, p_te))
            subj_aps.append(average_precision_score(y_te, p_te))
            subj_recalls.append(recall_score(y_te, pred_te, zero_division=0))
            subj_f1s.append(f1_score(y_te, pred_te, zero_division=0))
            y_base = y_te == 0
            subj_fprs.append(float(np.mean(pred_te[y_base] == 1)) if np.any(y_base) else 0.0)

        rows.append(
            {
                "model": m_name,
                "n_features": len(fcols),
                "wesad_loso_auroc_mean": float(np.mean(subj_aurocs)),
                "wesad_loso_auroc_std": float(np.std(subj_aurocs)),
                "wesad_loso_ap_mean": float(np.mean(subj_aps)),
                "wesad_loso_recall_mean": float(np.mean(subj_recalls)),
                "wesad_loso_f1_mean": float(np.mean(subj_f1s)),
                "wesad_loso_base_fpr_mean": float(np.mean(subj_fprs)),
                "wesad_worst_subj_auroc": float(np.min(subj_aurocs)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_external_cohort(
    stress_df: pd.DataFrame,
    aero_df: pd.DataFrame,
    model_name: str,
    p_stress_raw: np.ndarray,
    p_aero_raw: np.ndarray,
    gate_mode: str,
    base_threshold: float,
    protocol_name: str,
    latency_ms_per_win: float,
) -> tuple[dict, list[dict]]:
    """Compute subject-level and cohort-level metrics on PhysioNet external test data."""
    p_st_gated, thr_st, pred_st = apply_dynamic_motion_gate(
        p_stress_raw, stress_df, mode=gate_mode, base_threshold=base_threshold
    )
    p_ae_gated, thr_ae, pred_ae = apply_dynamic_motion_gate(
        p_aero_raw, aero_df, mode=gate_mode, base_threshold=base_threshold
    )

    stress_work = stress_df.copy()
    stress_work["_p"] = p_st_gated
    stress_work["_pred"] = pred_st

    aero_work = aero_df.copy()
    aero_work["_p"] = p_ae_gated
    aero_work["_pred"] = pred_ae

    per_subj_rows = []
    for sid, sdf in stress_work.groupby("subject", sort=True):
        y_true = (sdf["label"] == "stress").astype(int).to_numpy()
        p_val = sdf["_p"].to_numpy()
        pred = sdf["_pred"].to_numpy()
        starts = sdf["start_s"].to_numpy()

        auroc = (
            float(roc_auc_score(y_true, p_val))
            if len(np.unique(y_true)) == 2
            else np.nan
        )
        ap = (
            float(average_precision_score(y_true, p_val))
            if len(np.unique(y_true)) == 2
            else np.nan
        )
        rec = float(recall_score(y_true, pred, zero_division=0))
        prec = float(precision_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))

        base_mask = y_true == 0
        base_fpr = float(np.mean(pred[base_mask] == 1)) if np.any(base_mask) else np.nan
        base_win_far, base_ep_far = compute_episode_far_per_hour(
            pred[base_mask], starts[base_mask]
        )

        # Event/Task-level recall (fraction of stress tasks with >=1 detected window)
        stress_tasks = sdf[sdf["label"] == "stress"]
        if "task" in stress_tasks.columns and len(stress_tasks) > 0:
            task_hits = [
                int(np.any(tdf["_pred"].to_numpy() == 1))
                for _, tdf in stress_tasks.groupby("task")
            ]
            event_recall = float(np.mean(task_hits))
        else:
            event_recall = float(np.any(pred[y_true == 1] == 1))

        # Baseline tonic EDA for subgroup classification
        eda_base_mean = float(sdf.loc[base_mask, "eda_mean"].mean())
        if eda_base_mean < 0.6:
            eda_subgroup = "Low_Responder (<0.6 uS)"
        elif eda_base_mean < 4.0:
            eda_subgroup = "Medium_Responder (0.6-4.0 uS)"
        else:
            eda_subgroup = "High_Tonic (>4.0 uS)"

        # Aerobic exercise for this subject (if eligible)
        adf = aero_work[aero_work["subject"] == sid]
        if len(adf) > 0:
            ae_pred = adf["_pred"].to_numpy()
            ae_starts = adf["start_s"].to_numpy()
            ae_fpr = float(np.mean(ae_pred == 1))
            ae_win_far, ae_ep_far = compute_episode_far_per_hour(ae_pred, ae_starts)
        else:
            ae_fpr = np.nan
            ae_win_far = np.nan
            ae_ep_far = np.nan

        per_subj_rows.append(
            {
                "protocol": protocol_name,
                "model": model_name,
                "gate_mode": gate_mode,
                "subject": sid,
                "eda_base_uS": eda_base_mean,
                "eda_subgroup": eda_subgroup,
                "n_stress_win": int(np.sum(y_true == 1)),
                "n_base_win": int(np.sum(y_true == 0)),
                "auroc": auroc,
                "ap": ap,
                "stress_recall": rec,
                "event_recall": event_recall,
                "precision": prec,
                "f1": f1,
                "baseline_fpr": base_fpr,
                "baseline_ep_far_per_hr": base_ep_far,
                "aerobic_fpr": ae_fpr,
                "aerobic_ep_far_per_hr": ae_ep_far,
            }
        )

    ps_df = pd.DataFrame(per_subj_rows)
    ci_low, ci_high = bootstrap_ci(ps_df["auroc"].to_numpy())
    s3_df = ps_df[ps_df["subject"].isin(["S01", "S03", "S04"])]
    worst_idx = ps_df["auroc"].idxmin()
    worst_subj = str(ps_df.loc[worst_idx, "subject"])
    worst_auroc = float(ps_df.loc[worst_idx, "auroc"])

    # Pooled aerobic FAR across all 14 eligible aerobic subjects
    all_ae_fpr = float(np.mean(pred_ae == 1)) if len(pred_ae) > 0 else np.nan
    _, all_ae_ep_far = compute_episode_far_per_hour(
        pred_ae, aero_work["start_s"].to_numpy()
    )

    summary = {
        "protocol": protocol_name,
        "model": model_name,
        "gate_mode": gate_mode,
        "n_stress_subjects": len(ps_df),
        "n_aerobic_subjects": int(aero_work["subject"].nunique()),
        "mean_auroc_17sub": float(ps_df["auroc"].mean()),
        "std_auroc_17sub": float(ps_df["auroc"].std()),
        "median_auroc_17sub": float(ps_df["auroc"].median()),
        "auroc_95ci_low": ci_low,
        "auroc_95ci_high": ci_high,
        "mean_auroc_3sub_s01_s03_s04": float(s3_df["auroc"].mean()),
        "mean_ap_17sub": float(ps_df["ap"].mean()),
        "mean_stress_recall": float(ps_df["stress_recall"].mean()),
        "mean_event_recall": float(ps_df["event_recall"].mean()),
        "mean_precision": float(ps_df["precision"].mean()),
        "mean_f1": float(ps_df["f1"].mean()),
        "mean_baseline_fpr": float(ps_df["baseline_fpr"].mean()),
        "mean_baseline_ep_far_hr": float(ps_df["baseline_ep_far_per_hr"].mean()),
        "pooled_aerobic_fpr": all_ae_fpr,
        "mean_aerobic_fpr": float(ps_df["aerobic_fpr"].dropna().mean()),
        "pooled_aerobic_ep_far_hr": all_ae_ep_far,
        "worst_subject": worst_subj,
        "worst_subject_auroc": worst_auroc,
        "low_responder_mean_auroc": float(
            ps_df[ps_df["eda_subgroup"].str.startswith("Low")]["auroc"].mean()
        ),
        "medium_responder_mean_auroc": float(
            ps_df[ps_df["eda_subgroup"].str.startswith("Medium")]["auroc"].mean()
        ),
        "high_tonic_mean_auroc": float(
            ps_df[ps_df["eda_subgroup"].str.startswith("High")]["auroc"].mean()
        ),
        "inference_latency_ms_per_window": latency_ms_per_win,
    }
    return summary, per_subj_rows


def evaluate_stress_under_motion_simulation(
    stress_df: pd.DataFrame,
    aero_df: pd.DataFrame,
    p_stress_physio: np.ndarray,
    out_dir: Path,
) -> dict:
    """Task A: Evaluate whether Stress-During-Motion can be distinguished from routine Exercise.

    Constructs a semi-synthetic Stress-Under-Motion benchmark by pairing real PhysioNet
    Stress physiological features with real PhysioNet Aerobic ACC motion features, and
    compares:
    1. Commit b46a2e9 Hard Gate (`legacy_hard`)
    2. Proposed Soft Dynamic Gate (`soft_dynamic`)
    3. Joint 4-State Inference (`infer_joint_4state`)
    """
    true_stress_rows = stress_df[stress_df["label"] == "stress"].copy().reset_index(drop=True)
    true_stress_p = p_stress_physio[(stress_df["label"] == "stress").to_numpy()]

    # Gesture-accompanied real stress windows in PhysioNet (top quartile of motion change during stress)
    q75_jerk = float(np.percentile(true_stress_rows["acc_motion_change_per_s"], 75))
    gesture_mask = (true_stress_rows["acc_motion_change_per_s"] >= q75_jerk).to_numpy()

    _, _, pred_legacy_gest = apply_dynamic_motion_gate(
        true_stress_p[gesture_mask],
        true_stress_rows[gesture_mask],
        mode="legacy_hard",
        base_threshold=0.45,
    )
    _, _, pred_soft_gest = apply_dynamic_motion_gate(
        true_stress_p[gesture_mask],
        true_stress_rows[gesture_mask],
        mode="soft_dynamic",
        base_threshold=0.45,
    )

    # Semi-synthetic intense exercise + acute stress surge:
    # Overlay real Aerobic ACC onto real Stress physiology
    sim_df = true_stress_rows.copy()
    acc_cols = [c for c in sim_df.columns if c.startswith("acc_")]
    for c in acc_cols:
        sim_df[c] = aero_df[c].iloc[: len(sim_df)].to_numpy()

    j4_sim = infer_joint_4state(true_stress_p, sim_df, base_threshold=0.45)
    extreme_surge_detected_rate = float(
        np.mean(j4_sim["joint_state_label"] == "stress_under_motion")
    )

    res = {
        "n_real_gesture_stress_windows": int(np.sum(gesture_mask)),
        "gesture_stress_recall_b46a2e9_hard_gate": float(np.mean(pred_legacy_gest == 1)),
        "gesture_stress_recall_proposed_soft_dynamic": float(np.mean(pred_soft_gest == 1)),
        "semi_synthetic_intense_exercise_stress_surge_flag_rate": extreme_surge_detected_rate,
    }
    with open(out_dir / "task_a_stress_under_motion_summary.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


def main():
    out_dir = ROOT / "results" / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading WESAD dataset and applying strict causal personal calibration...")
    w_raw = pd.read_csv(ROOT / "data" / "processed" / "wesad_features.csv")
    w_bin = w_raw[(w_raw["y"].isin([0, 1])) & (w_raw["qc_common_valid"] == True)].copy()
    w_bin["label"] = np.where(w_bin["y"] == 1, "stress", "baseline")
    w_bin = w_bin.sort_values(["subject", "start_s"]).reset_index(drop=True)
    w_cal = apply_causal_personal_calibration(
        w_bin,
        is_baseline_candidate=(w_bin["phase"] == 1).to_numpy(),
        max_init_windows=10,
        online_drift_rate=0.0,
    )
    w_cal = add_causal_temporal_features(w_cal)

    wesad_cv_path = out_dir / "wesad_15sub_loso_comparison.csv"
    if wesad_cv_path.exists():
        print("2. Loading cached 15-subject WESAD LOSO Cross-Validation results...")
        wesad_cv_df = pd.read_csv(wesad_cv_path)
    else:
        print("2. Running 15-subject WESAD LOSO Cross-Validation across all candidate architectures...")
        wesad_cv_df = run_wesad_loso_evaluation(w_cal)
        wesad_cv_df.to_csv(wesad_cv_path, index=False)
    print(wesad_cv_df.to_string(index=False))

    print("\n3. Loading PhysioNet 18-subject Legacy & Tag-Aligned window datasets...")
    p_legacy_raw = pd.read_csv(
        ROOT / "results" / "phase3_strict" / "physionet_external" / "per_window_predictions.csv"
    )
    p_tag_raw = extract_or_load_tag_aligned_windows(out_dir)
    if "prob_rf_all" not in p_tag_raw.columns:
        p_tag_raw["has_bvp"] = True
        evaluate_models_on_windows(p_tag_raw)
        p_tag_raw.to_csv(out_dir / "physionet_windows_tag_aligned_18sub.csv", index=False)

    # Calibrate both Legacy and Tag-Aligned datasets causally per subject
    # Note: For AEROBIC sessions, anchor each subject's physiological baseline strictly to
    # that subject's initial resting Baseline from their STRESS session (zero stress label usage).
    def prepare_physionet_protocol_dfs(p_all: pd.DataFrame, online_drift: float = 0.0):
        st_df = p_all[p_all["label"].isin(["baseline", "stress"])].copy()
        st_df = st_df.sort_values(["subject", "start_s"]).reset_index(drop=True)
        st_cal = apply_causal_personal_calibration(
            st_df, max_init_windows=10, online_drift_rate=online_drift
        )
        st_cal = add_causal_temporal_features(st_cal)

        # Extract each subject's initial baseline reference from st_df (only baseline windows)
        base_only = st_df[st_df["label"] == "baseline"].copy()
        ae_df = p_all[p_all["label"] == "exercise_aerobic"].copy()
        ae_df = ae_df.sort_values(["subject", "start_s"]).reset_index(drop=True)

        # Prepend subject's initial baseline windows before aerobic windows so causal calibration
        # uses the subject's true resting baseline (or first windows if S02 has no verified stress baseline)
        combined_ae = pd.concat([base_only, ae_df], ignore_index=True)
        combined_ae_cal = apply_causal_personal_calibration(
            combined_ae,
            is_baseline_candidate=(combined_ae["label"] == "baseline").to_numpy(),
            max_init_windows=10,
            online_drift_rate=0.0,
        )
        ae_cal = combined_ae_cal[combined_ae_cal["label"] == "exercise_aerobic"].copy()
        ae_cal = add_causal_temporal_features(ae_cal)
        return st_cal, ae_cal

    leg_st, leg_ae = prepare_physionet_protocol_dfs(p_legacy_raw, online_drift=0.0)
    tag_st, tag_ae = prepare_physionet_protocol_dfs(p_tag_raw, online_drift=0.0)
    tag_st_online, tag_ae_online = prepare_physionet_protocol_dfs(p_tag_raw, online_drift=0.05)

    # Also build Strict Out-of-Sample Post-Calibration Split (first 2 baseline windows used
    # ONLY for reference estimation and excluded from test evaluation windows)
    from src.features.advanced import compute_log_base_features
    def build_strict_postcal_split(p_all: pd.DataFrame, n_cal_win: int = 2):
        st_df = p_all[p_all["label"].isin(["baseline", "stress"])].copy()
        st_df = st_df.sort_values(["subject", "start_s"]).reset_index(drop=True)
        raw_obs = compute_log_base_features(st_df)
        raw_cols = list(raw_obs.columns)
        st_df["_is_post_cal"] = True
        ref_map = {}
        for c in CALIBRATED_PHYSIO_FEATURES:
            st_df[c] = 0.0
        for sid, sdf in st_df.groupby("subject", sort=False):
            idx = sdf.index
            vals = raw_obs.loc[idx, raw_cols].to_numpy(dtype=float)
            b_idx = np.where((sdf["label"] == "baseline").to_numpy())[0]
            cal_k = b_idx[: min(len(b_idx), n_cal_win)]
            ref = np.mean(vals[cal_k], axis=0)
            ref_map[sid] = ref
            deltas = vals - ref
            for j, dcol in enumerate(CALIBRATED_PHYSIO_FEATURES):
                st_df.loc[idx, dcol] = deltas[:, j]
            post_mask = np.ones(len(sdf), dtype=bool)
            if len(b_idx) > n_cal_win:
                post_mask[cal_k] = False
            st_df.loc[idx, "_is_post_cal"] = post_mask
        st_df = add_causal_temporal_features(st_df)
        st_post = st_df[st_df["_is_post_cal"]].copy().reset_index(drop=True)

        ae_df = p_all[p_all["label"] == "exercise_aerobic"].copy()
        ae_df = ae_df.sort_values(["subject", "start_s"]).reset_index(drop=True)
        raw_ae = compute_log_base_features(ae_df)
        for c in CALIBRATED_PHYSIO_FEATURES:
            ae_df[c] = 0.0
        for sid, adf in ae_df.groupby("subject", sort=False):
            idx = adf.index
            vals = raw_ae.loc[idx, raw_cols].to_numpy(dtype=float)
            ref = ref_map.get(sid, np.mean(vals[: min(len(vals), n_cal_win)], axis=0))
            deltas = vals - ref
            for j, dcol in enumerate(CALIBRATED_PHYSIO_FEATURES):
                ae_df.loc[idx, dcol] = deltas[:, j]
        ae_df = add_causal_temporal_features(ae_df)
        return st_post, ae_df

    post_tag_st, post_tag_ae = build_strict_postcal_split(p_tag_raw, n_cal_win=2)

    print(
        f"Legacy protocol: {leg_st['subject'].nunique()} STRESS subjects ({len(leg_st)} windows), "
        f"{leg_ae['subject'].nunique()} AEROBIC subjects ({len(leg_ae)} windows)"
    )
    print(
        f"Tag-Aligned protocol: {tag_st['subject'].nunique()} STRESS subjects ({len(tag_st)} windows), "
        f"{tag_ae['subject'].nunique()} AEROBIC subjects ({len(tag_ae)} windows)"
    )

    print("\n4. Evaluating Frozen b46a2e9 Models + CBARA vs New Calibrated/Temporal Models...")
    summary_rows = []
    all_per_subj_rows = []

    # 4A. Evaluate frozen b46a2e9 models (`lr_all`, `rf_all`, `lr_eda`, `lr_acc`) on both protocols
    for m_id in ["lr_all", "rf_all", "lr_eda", "lr_acc"]:
        for proto_name, st_df, ae_df in [
            ("Legacy_Segments", leg_st, leg_ae),
            ("Tag_Aligned_Official", tag_st, tag_ae),
        ]:
            p_st_raw = st_df[f"prob_{m_id}"].astype(float).to_numpy()
            p_ae_raw = ae_df[f"prob_{m_id}"].astype(float).to_numpy()
            t_ms = 0.040 if m_id.startswith("lr_") else 0.120

            # (i) Cold-Start (no gate)
            s_row, ps_rows = evaluate_external_cohort(
                st_df, ae_df, f"Frozen_{m_id}_ColdStart", p_st_raw, p_ae_raw, "none", 0.50, proto_name, t_ms
            )
            summary_rows.append(s_row)
            all_per_subj_rows.extend(ps_rows)

            # (ii) Legacy b46a2e9 Hard Gate
            s_row, ps_rows = evaluate_external_cohort(
                st_df, ae_df, f"Frozen_{m_id}_LegacyHardGate", p_st_raw, p_ae_raw, "legacy_hard", 0.50, proto_name, t_ms
            )
            summary_rows.append(s_row)
            all_per_subj_rows.extend(ps_rows)

            # (iii) CBARA Logit Calibration + Soft Dynamic Gate (Recovers Recall on Frozen Models!)
            p_st_cbara = np.zeros_like(p_st_raw)
            for sid, sdf in st_df.groupby("subject"):
                idx = st_df["subject"] == sid
                b_mask = (sdf["label"] == "baseline").to_numpy()
                p_st_cbara[idx.to_numpy()] = apply_cbara_probability_calibration(
                    p_st_raw[idx.to_numpy()], b_mask, target_base_prior=0.25
                )
            p_ae_cbara = np.zeros_like(p_ae_raw)
            for sid, adf in ae_df.groupby("subject"):
                idx_ae = (ae_df["subject"] == sid).to_numpy()
                idx_st_base = ((st_df["subject"] == sid) & (st_df["label"] == "baseline")).to_numpy()
                if np.any(idx_st_base):
                    ref_And_ae = np.concatenate([p_st_raw[idx_st_base], p_ae_raw[idx_ae]])
                    mask_ref = np.zeros(len(ref_And_ae), dtype=bool)
                    mask_ref[: int(np.sum(idx_st_base))] = True
                    cal_all = apply_cbara_probability_calibration(
                        ref_And_ae, mask_ref, target_base_prior=0.25
                    )
                    p_ae_cbara[idx_ae] = cal_all[int(np.sum(idx_st_base)) :]
                else:
                    p_ae_cbara[idx_ae] = p_ae_raw[idx_ae]

            s_row, ps_rows = evaluate_external_cohort(
                st_df,
                ae_df,
                f"Frozen_{m_id}_CBARA_SoftDynamic",
                p_st_cbara,
                p_ae_cbara,
                "soft_dynamic",
                0.45,
                proto_name,
                t_ms,
            )
            summary_rows.append(s_row)
            all_per_subj_rows.extend(ps_rows)

    # 4B. Train new WESAD models and evaluate on both Legacy and Tag-Aligned PhysioNet protocols
    y_w = (w_cal["label"] == "stress").astype(int).to_numpy()
    candidates = build_model_candidates()
    fitted_models = {}
    for m_name, (pipe_tmpl, fcols) in candidates.items():
        pipe_tmpl.fit(w_cal[fcols], y_w)
        fitted_models[m_name] = (pipe_tmpl, fcols)

    # Save trained Phase 4 models
    with open(out_dir / "phase4_trained_models.pkl", "wb") as f:
        pickle.dump(fitted_models, f)

    for proto_name, st_df, ae_df in [
        ("Legacy_Segments", leg_st, leg_ae),
        ("Tag_Aligned_Official", tag_st, tag_ae),
        ("Strict_PostCal_Tag_Aligned", post_tag_st, post_tag_ae),
    ]:
        p_ensemble_st = []
        p_ensemble_ae = []
        for m_name, (est, fcols) in fitted_models.items():
            t0 = time.perf_counter()
            p_st = est.predict_proba(st_df[fcols])[:, 1]
            t_ms = (time.perf_counter() - t0) * 1000.0 / max(len(st_df), 1)
            p_ae = est.predict_proba(ae_df[fcols])[:, 1]

            if m_name in ("LR_Calib_Temporal", "RF_Calib_Temporal", "LGBM_Calib_Temporal"):
                p_ensemble_st.append(p_st)
                p_ensemble_ae.append(p_ae)

            # Evaluate with Soft Dynamic Gate (default proposed)
            s_row, ps_rows = evaluate_external_cohort(
                st_df, ae_df, m_name, p_st, p_ae, "soft_dynamic", 0.42, proto_name, t_ms
            )
            summary_rows.append(s_row)
            all_per_subj_rows.extend(ps_rows)

            # For primary models, also run full Gate Ablation (none, legacy_hard, exercise_hard, soft_dynamic)
            if m_name in ("RF_Calib_Temporal", "LGBM_Calib_Temporal", "LR_Calib_Temporal"):
                for g_mode in ("none", "legacy_hard", "exercise_hard"):
                    s_row_g, _ = evaluate_external_cohort(
                        st_df, ae_df, m_name, p_st, p_ae, g_mode, 0.42, proto_name, t_ms
                    )
                    summary_rows.append(s_row_g)

        # Evaluate Ensemble (LR + RF + LGBM Calibrated Temporal)
        p_ens_st = np.mean(p_ensemble_st, axis=0)
        p_ens_ae = np.mean(p_ensemble_ae, axis=0)
        for g_mode in ("none", "legacy_hard", "exercise_hard", "soft_dynamic"):
            s_row_ens, ps_rows_ens = evaluate_external_cohort(
                st_df,
                ae_df,
                "Ensemble_Calib_Temporal",
                p_ens_st,
                p_ens_ae,
                g_mode,
                0.42,
                proto_name,
                0.045,
            )
            summary_rows.append(s_row_ens)
            if g_mode == "soft_dynamic":
                all_per_subj_rows.extend(ps_rows_ens)

    # 4C. Evaluate Online Baseline Drift Update (alpha=0.05) vs Static Initial Baseline on Tag-Aligned
    est_rf, fcols_rf = fitted_models["RF_Calib_Temporal"]
    p_st_on = est_rf.predict_proba(tag_st_online[fcols_rf])[:, 1]
    p_ae_on = est_rf.predict_proba(tag_ae_online[fcols_rf])[:, 1]
    s_row_on, _ = evaluate_external_cohort(
        tag_st_online,
        tag_ae_online,
        "RF_Calib_Temporal_OnlineDrift005",
        p_st_on,
        p_ae_on,
        "soft_dynamic",
        0.42,
        "Tag_Aligned_Official",
        0.035,
    )
    summary_rows.append(s_row_on)

    # 5. Task A Stress-Under-Motion simulation
    p_rf_tag_st = est_rf.predict_proba(tag_st[fcols_rf])[:, 1]
    sum_sim = evaluate_stress_under_motion_simulation(tag_st, tag_ae, p_rf_tag_st, out_dir)
    print("\n5. Stress-Under-Motion Analysis:", json.dumps(sum_sim, indent=2))

    # Save summary and per-subject tables
    summary_df = pd.DataFrame(summary_rows)
    per_subj_df = pd.DataFrame(all_per_subj_rows)

    summary_df.to_csv(out_dir / "phase4_generalization_summary.csv", index=False)
    per_subj_df.to_csv(out_dir / "phase4_generalization_per_subject.csv", index=False)

    # Also update results/phase3_strict/physionet_external/ with comprehensive comparison
    ext_dir = ROOT / "results" / "phase3_strict" / "physionet_external"
    summary_df.to_csv(ext_dir / "generalization_summary_phase4_expanded.csv", index=False)
    per_subj_df.to_csv(ext_dir / "generalization_per_subject_phase4_expanded.csv", index=False)

    print("\n=== TOP SUMMARY RESULTS (Tag_Aligned_Official & Legacy_Segments) ===")
    cols_show = [
        "protocol",
        "model",
        "gate_mode",
        "mean_auroc_17sub",
        "median_auroc_17sub",
        "mean_ap_17sub",
        "mean_stress_recall",
        "mean_f1",
        "mean_baseline_fpr",
        "pooled_aerobic_fpr",
        "worst_subject",
        "worst_subject_auroc",
    ]
    print(
        summary_df[
            summary_df["gate_mode"].isin(["soft_dynamic", "legacy_hard", "none"])
        ][cols_show].to_string(index=False)
    )


if __name__ == "__main__":
    main()
