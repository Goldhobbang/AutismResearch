"""End-to-end experimental runner for Nurses Stress future event forecasting.

Executes:
1. Full raw dataset audit & label reliability audit -> `dataset_manifest.json`, `subject_summary.csv`, `event_audit.csv`
2. Causal multi-scale feature extraction across 214 annotated sessions -> `features.csv`
3. Core model comparison (M0 Dummy, M1 LR, M2 RF, M3 LightGBM) on P1 (60-180s early warning) -> `model_comparison.csv`, `per_subject_metrics.csv`, `splits.csv`, `predictions.csv`, `event_metrics.csv`
4. Lead-time comparison (P0: 0-60s, P1: 60-180s, P2: 180-300s) -> `lead_time_comparison.csv`
5. Sensor combination ablation (EDA, PPG, ACC, EDA+PPG, ALL) -> `sensor_ablation.csv`
6. Experiment A: Physical activity / motion gate comparison (None, Hard Gate, Soft Dynamic Gate) -> `motion_gate_comparison.csv`
7. Experiment B: Alert policy trade-off (Fixed 0.5, Val Opt, Hysteresis, Duration 2-win, Cooldown 300s, Soft Gate + Hysteresis) -> `alert_policy_comparison.csv`
8. Experiment C & D: New user generalization & Personal baseline calibration comparison (Uncalibrated, 1m, 3m, 5m, Low-Motion, Adaptive Fallback) -> `calibration_comparison.csv`
9. Final model artifact export -> `results/nurse_forecasting/models/`
10. All 10 required visual evidence figures -> `results/nurse_forecasting/figures/`
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.advanced import (
    CALIBRATED_PHYSIO_FEATURES,
    apply_dynamic_motion_gate,
    compute_causal_exercise_score,
)
from src.features.nurse_forecasting import (
    ACC_RAW_FEATURES,
    EDA_RAW_FEATURES,
    PPG_RAW_FEATURES,
    SENSOR_FEATURE_MAP,
)
from src.forecasting.alerts import (
    apply_alert_policy_to_session,
    evaluate_event_level_forecasting,
)
from src.forecasting.calibration import apply_session_causal_calibration
from src.forecasting.evaluate import compute_window_metrics, summarize_subject_metrics
from src.forecasting.train import (
    build_forecast_model,
    generate_nurse_features_table,
    run_loso_forecast_experiment,
    select_validation_threshold,
)
from src.loaders.nurse_stress import load_nurse_session_zip

OUT_DIR = ROOT / "results" / "nurse_forecasting"
FIG_DIR = OUT_DIR / "figures"
MOD_DIR = OUT_DIR / "models"


def evaluate_predictions_with_event_metrics(
    pred_df: pd.DataFrame,
    ev_df: pd.DataFrame,
    task: str = "P1",
    pred_col: str = "pred",
    prob_col: str = "prob",
    label_tag: str = "",
) -> dict:
    """Compute both participant-averaged window metrics and cohort event-level metrics."""
    elig_col = f"{task.lower()}_forecast_eligible"
    max_lead = 60.0 if task == "P0" else (180.0 if task == "P1" else 300.0)
    min_lead = 0.0 if task == "P0" else (60.0 if task == "P1" else 180.0)

    subj_rows = []
    for sid, sg in pred_df.groupby("subject", sort=True):
        wm = compute_window_metrics(
            sg["y_true"].to_numpy(dtype=int),
            sg[prob_col].to_numpy(dtype=float),
            sg[pred_col].to_numpy(dtype=int),
        )
        wm["subject"] = sid
        wm["group"] = label_tag
        subj_rows.append(wm)

    s_df = pd.DataFrame(subj_rows)
    sum_df = summarize_subject_metrics(s_df, ["group"])
    sum_row = sum_df.iloc[0].to_dict()

    # Pooled window metrics as secondary check
    pooled_wm = compute_window_metrics(
        pred_df["y_true"].to_numpy(dtype=int),
        pred_df[prob_col].to_numpy(dtype=float),
        pred_df[pred_col].to_numpy(dtype=int),
    )

    elig_evs = ev_df[ev_df[elig_col] == True]
    ev_m = evaluate_event_level_forecasting(
        pred_df,
        elig_evs,
        pred_col=pred_col,
        max_valid_lead_s=max_lead,
        min_valid_lead_s=min_lead,
    )

    return {
        **sum_row,
        "pooled_auroc": pooled_wm["auroc"],
        "pooled_auprc": pooled_wm["auprc"],
        "pooled_f1": pooled_wm["f1"],
        "pooled_fpr": pooled_wm["fpr"],
        **ev_m,
    }


def make_all_figures(
    feat_df: pd.DataFrame,
    ev_df: pd.DataFrame,
    model_comp_df: pd.DataFrame,
    lead_comp_df: pd.DataFrame,
    sensor_abl_df: pd.DataFrame,
    motion_comp_df: pd.DataFrame,
    policy_comp_df: pd.DataFrame,
    calib_comp_df: pd.DataFrame,
    per_subj_p1_df: pd.DataFrame,
    preds_by_model: dict[str, pd.DataFrame],
    final_lr_pipe,
    final_cols: list[str],
):
    """Generate all 10 required visual evidence figures with explicit dataset/condition captions."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    caption_base = (
        "Dataset: Hospital Nurses Stress (15 subjects, 214 annotated sessions, 176 confirmed P1 events) | "
        "15-fold LOSO | Window=180s | Survey start resolution +-60s (NOT ASD meltdown)"
    )

    def save_fig(fig, filename: str):
        fig.text(0.5, 0.005, caption_base, ha="center", fontsize=7.5, color="#333333")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(FIG_DIR / filename, dpi=180)
        plt.close(fig)

    # 1. Model AUROC & AUPRC comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    models = model_comp_df["model"].tolist()
    x = np.arange(len(models))
    w = 0.35
    axes[0].bar(x - w / 2, model_comp_df["auroc_mean"], width=w, label="Subject Mean AUROC", color="#1f77b4")
    axes[0].bar(x + w / 2, model_comp_df["auroc_median"], width=w, label="Subject Median AUROC", color="#6baed6")
    axes[0].axhline(0.5, linestyle="--", color="gray", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("AUROC (unitless)")
    axes[0].set_title("P1 (60-180s Lead) Model AUROC Comparison")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - w / 2, model_comp_df["auprc_mean"], width=w, label="Subject Mean AUPRC", color="#2ca02c")
    axes[1].bar(x + w / 2, model_comp_df["event_recall"], width=w, label="Event Recall (176 events)", color="#ff7f0e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Score / Rate")
    axes[1].set_title("P1 AUPRC & Event Recall Across Models")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    save_fig(fig, "01_model_auroc_auprc_comparison.png")

    # 2. Precision-Recall Curve (P1 models)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m_name, p_df in preds_by_model.items():
        y_t = p_df["y_true"].to_numpy(dtype=int)
        pr_p = p_df["prob"].to_numpy(dtype=float)
        prec, rec, _ = precision_recall_curve(y_t, pr_p)
        ap_val = model_comp_df.loc[model_comp_df["model"] == m_name, "pooled_auprc"].iloc[0]
        ax.plot(rec, prec, linewidth=1.8, label=f"{m_name} (Pooled AP={ap_val:.3f})")
    prev = preds_by_model["LR"]["y_true"].mean()
    ax.axhline(prev, linestyle="--", color="gray", label=f"Prevalence ({prev:.3f})")
    ax.set_xlabel("Window Recall (Sensitivity)")
    ax.set_ylabel("Window Precision (PPV)")
    ax.set_title("Precision-Recall Curve — P1 (60-180s Future Event Onset, 15-fold LOSO)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "02_precision_recall_curve.png")

    # 3. ROC Curve (P1 models)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m_name, p_df in preds_by_model.items():
        y_t = p_df["y_true"].to_numpy(dtype=int)
        pr_p = p_df["prob"].to_numpy(dtype=float)
        fpr_c, tpr_c, _ = roc_curve(y_t, pr_p)
        auc_val = model_comp_df.loc[model_comp_df["model"] == m_name, "pooled_auroc"].iloc[0]
        ax.plot(fpr_c, tpr_c, linewidth=1.8, label=f"{m_name} (Pooled AUROC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (0.500)")
    ax.set_xlabel("Window False Positive Rate (FPR)")
    ax.set_ylabel("Window True Positive Rate (Recall)")
    ax.set_title("ROC Curve — P1 (60-180s Future Event Onset, 15-fold LOSO)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "03_roc_curve.png")

    # 4. Per-subject Recall and FPR (Primary LR model on P1)
    lr_sub = per_subj_p1_df[per_subj_p1_df["model"] == "LR"].sort_values("subject")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sx = np.arange(len(lr_sub))
    ax.bar(sx - 0.25, lr_sub["event_recall"], width=0.25, label="Event Recall", color="#1f77b4")
    ax.bar(sx, lr_sub["recall"], width=0.25, label="Window Recall", color="#2ca02c")
    ax.bar(sx + 0.25, lr_sub["fpr"], width=0.25, label="Window FPR", color="#d62728")
    ax.set_xticks(sx)
    ax.set_xticklabels(lr_sub["subject"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Nurse Participant ID (LOSO Held-out Test Subject)")
    ax.set_ylabel("Rate (0 to 1)")
    ax.set_title("Per-Subject Event Recall, Window Recall, and FPR (M1 Logistic Regression, P1)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "04_per_subject_recall_and_fpr.png")

    # 5. Performance across Lead Times (P0: 0-60s, P1: 60-180s, P2: 180-300s)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for m_name in ["LR", "RF", "LightGBM"]:
        sub_l = lead_comp_df[lead_comp_df["model"] == m_name]
        axes[0].plot(sub_l["task"], sub_l["auroc_mean"], "o-", linewidth=2, label=f"{m_name} Mean AUROC")
        axes[1].plot(sub_l["task"], sub_l["event_recall"], "s-", linewidth=2, label=f"{m_name} Event Recall")
    axes[0].set_ylabel("Subject-Mean AUROC")
    axes[0].set_xlabel("Forecast Horizon Setting (P0: 0-60s, P1: 60-180s, P2: 180-300s)")
    axes[0].set_title("AUROC vs Forecast Lead Time Horizon")
    axes[0].set_ylim(0.45, 0.85)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel("Event Recall (Fraction of Stress Events Warned)")
    axes[1].set_xlabel("Forecast Horizon Setting")
    axes[1].set_title("Event Recall vs Forecast Lead Time Horizon")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    save_fig(fig, "05_lead_time_performance.png")

    # 6. Event Recall vs False Alarms/hour Curve across decision thresholds
    lr_p1_pred = preds_by_model["LR"]
    elig_p1_evs = ev_df[ev_df["p1_forecast_eligible"] == True]
    thrs = np.linspace(0.25, 0.85, 25)
    ev_recs = []
    fars = []
    win_recs = []
    for t_val in thrs:
        tmp = lr_p1_pred.copy()
        tmp["pred_t"] = (tmp["prob"] >= t_val).astype(int)
        em = evaluate_event_level_forecasting(
            tmp, elig_p1_evs, pred_col="pred_t", max_valid_lead_s=180.0, min_valid_lead_s=60.0
        )
        ev_recs.append(em["event_recall"])
        fars.append(em["false_alarms_per_hour"])
        win_recs.append(float((tmp[tmp["y_true"] == 1]["pred_t"] == 1).mean()))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(fars, ev_recs, "o-", color="#1f77b4", linewidth=2, label="Event Recall (176 P1 events)")
    ax.plot(fars, win_recs, "s--", color="#2ca02c", linewidth=1.5, label="Window Recall")
    for idx_pt in [2, 6, 10, 14, 18, 22]:
        if idx_pt < len(thrs):
            ax.annotate(
                f"thr={thrs[idx_pt]:.2f}",
                (fars[idx_pt], ev_recs[idx_pt]),
                textcoords="offset points",
                xytext=(5, -10),
                fontsize=7.5,
            )
    ax.set_xlabel("False Alarm Episodes per Verified Non-Event Hour (FA/hr)")
    ax.set_ylabel("Recall (Sensitivity)")
    ax.set_title("Trade-off: Event Recall vs False Alarms/hour Across Thresholds (LR, P1)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "06_recall_vs_false_alarms_curve.png")

    # 7. Personal Calibration Duration & Method Comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cx = np.arange(len(calib_comp_df))
    c_names = calib_comp_df["calibration_method"].tolist()
    axes[0].bar(cx - 0.18, calib_comp_df["auroc_mean"], width=0.35, label="Mean AUROC", color="#1f77b4")
    axes[0].bar(cx + 0.18, calib_comp_df["event_recall"], width=0.35, label="Event Recall", color="#2ca02c")
    axes[0].set_xticks(cx)
    axes[0].set_xticklabels(c_names, rotation=20)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score / Rate")
    axes[0].set_title("Personal Calibration Methods: AUROC & Event Recall (LR, P1)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(cx - 0.18, calib_comp_df["false_alarms_per_hour"], width=0.35, label="False Alarms / hr", color="#d62728")
    axes[1].bar(cx + 0.18, calib_comp_df["worst_subject_auroc"], width=0.35, label="Worst Subject AUROC", color="#9467bd")
    axes[1].set_xticks(cx)
    axes[1].set_xticklabels(c_names, rotation=20)
    axes[1].set_ylabel("FA/hr or Worst AUROC")
    axes[1].set_title("Personal Calibration: False Alarms/hr & Worst Subject Floor")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    save_fig(fig, "07_personal_calibration_comparison.png")

    # 8. Physical Activity / Motion Score vs Risk Score Relationship
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    p_sample = lr_p1_pred.copy()
    p_sample["motion_bin"] = pd.qcut(
        p_sample["acc_motion_change_per_s"].rank(method="first"),
        q=4,
        labels=["Q1 (Quiet)", "Q2 (Low)", "Q3 (Moderate)", "Q4 (High Activity)"],
    )
    p_gated_soft, dyn_thr_soft, pred_soft = apply_dynamic_motion_gate(
        p_sample["prob"].to_numpy(), p_sample, mode="soft_dynamic", base_threshold=0.50
    )
    _, _, pred_hard = apply_dynamic_motion_gate(
        p_sample["prob"].to_numpy(), p_sample, mode="legacy_hard", base_threshold=0.50
    )
    p_sample["pred_none"] = p_sample["pred"]
    p_sample["pred_hard"] = pred_hard
    p_sample["pred_soft"] = pred_soft

    neg_sub = p_sample[p_sample["y_true"] == 0]
    b_grp = neg_sub.groupby("motion_bin", observed=False)[["pred_none", "pred_hard", "pred_soft"]].mean()
    bx = np.arange(len(b_grp))
    axes[0].bar(bx - 0.25, b_grp["pred_none"], width=0.25, label="No Gate", color="#e6550d")
    axes[0].bar(bx, b_grp["pred_hard"], width=0.25, label="Hard Gate", color="#756bb1")
    axes[0].bar(bx + 0.25, b_grp["pred_soft"], width=0.25, label="Soft Dynamic Gate", color="#31a354")
    axes[0].set_xticks(bx)
    axes[0].set_xticklabels(b_grp.index)
    axes[0].set_ylabel("False Alert Rate on Non-Event Windows")
    axes[0].set_title("Non-Event Window False Alert Rate by ACC Activity Quartile")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Scatter of Exercise Score M(t) vs Raw & Gated Risk Score
    rng = np.random.default_rng(42)
    s_idx = rng.choice(len(p_sample), size=min(1200, len(p_sample)), replace=False)
    axes[1].scatter(
        p_sample["exercise_score"].iloc[s_idx],
        p_sample["prob"].iloc[s_idx],
        alpha=0.25,
        s=12,
        color="#d62728",
        label="Uncorrected Prob p(t)",
    )
    axes[1].scatter(
        p_sample["exercise_score"].iloc[s_idx],
        p_gated_soft[s_idx],
        alpha=0.25,
        s=12,
        color="#1f77b4",
        label="Soft-Gated Prob p_gated(t)",
    )
    axes[1].set_xlabel("Causal Activity / Exercise Score M(t) [0 to 1]")
    axes[1].set_ylabel("Predicted Pre-Stress Risk Score")
    axes[1].set_title("Effect of Soft Dynamic Motion Gate Across Activity Levels")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    save_fig(fig, "08_motion_vs_risk_score.png")

    # 9. Real Continuous Biosignal, Event Onset, Risk Score, and Alert Timeline (ACTUAL DATA)
    # Find a session in E4 or 7A or F5 with a clear detected confirmed stress event
    pos_hits = lr_p1_pred[(lr_p1_pred["y_true"] == 1) & (lr_p1_pred["pred"] == 1)].sort_values("prob", ascending=False)
    chosen_sess = str(pos_hits.iloc[0]["session"]) if not pos_hits.empty else str(lr_p1_pred.iloc[0]["session"])
    chosen_subj = str(pos_hits.iloc[0]["subject"]) if not pos_hits.empty else str(lr_p1_pred.iloc[0]["subject"])

    sess_windows = feat_df[(feat_df["subject"] == chosen_subj) & (feat_df["session"] == chosen_sess)].sort_values("pred_ts").copy()
    # Compute model risk score on all windows of this session using the held-out fold model
    fold_thr = float(lr_p1_pred[lr_p1_pred["subject"] == chosen_subj]["threshold"].iloc[0])
    # Get predictions from held-out fold for this subject
    sess_probs = final_lr_pipe.predict_proba(sess_windows[final_cols])[:, 1]
    sess_gated, sess_dyn_thr, sess_alerts = apply_dynamic_motion_gate(
        sess_probs, sess_windows, mode="soft_dynamic", base_threshold=fold_thr
    )
    sess_windows["prob_raw"] = sess_probs
    sess_windows["prob_gated"] = sess_gated
    sess_windows["alert_fired"] = sess_alerts

    t_min_hr = (sess_windows["pred_ts"] - sess_windows["pred_ts"].min()) / 60.0
    sess_evs = ev_df[(ev_df["subject"] == chosen_subj) & (ev_df["matched_session"] == chosen_sess)]

    fig, axes = plt.subplots(4, 1, figsize=(12, 8.5), sharex=True)
    t0_abs = float(sess_windows["pred_ts"].min())

    axes[0].plot(t_min_hr, sess_windows["eda_mean"], color="#1f77b4", linewidth=1.4, label="EDA Mean (uS, 180s causal window)")
    axes[0].set_ylabel("EDA (uS)")
    axes[0].set_title(f"Real Continuous Timeline — Nurse {chosen_subj} | Session {chosen_sess} (Held-out Test Fold)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_min_hr, sess_windows["hr_mean_bpm"], color="#d62728", linewidth=1.4, label="Heart Rate Mean (bpm)")
    axes[1].set_ylabel("HR (bpm)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t_min_hr, sess_windows["acc_motion_change_per_s"], color="#7f7f7f", linewidth=1.3, label="ACC Motion Jerk (1/64g / s)")
    axes[2].set_ylabel("ACC Jerk")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    axes[3].plot(t_min_hr, sess_windows["prob_gated"], color="#2ca02c", linewidth=1.8, label="Soft-Gated Pre-Event Risk Score")
    axes[3].axhline(fold_thr, color="#333333", linestyle="--", linewidth=1.0, label=f"Validation Threshold ({fold_thr:.2f})")
    alert_pts = sess_windows[sess_windows["alert_fired"] == 1]
    if not alert_pts.empty:
        axes[3].scatter(
            (alert_pts["pred_ts"] - t0_abs) / 60.0,
            alert_pts["prob_gated"],
            color="red",
            s=28,
            zorder=5,
            label="Early Warning Alert Fired",
        )
    axes[3].set_ylabel("Risk Score")
    axes[3].set_xlabel("Session Elapsed Time (minutes from first 180s window)")
    axes[3].set_ylim(0, 1.05)
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.3)

    # Shade confirmed stress events and mark onset vertical lines across all 4 panels
    for _, ev_r in sess_evs.iterrows():
        ev_s_min = (float(ev_r["event_start_ts"]) - t0_abs) / 60.0
        ev_e_min = (float(ev_r["event_end_ts"]) - t0_abs) / 60.0
        cat = str(ev_r["label_category"])
        color_sh = "#ff9896" if "stress" in cat and "no_stress" not in cat else "#c7e9c0"
        for ax in axes:
            ax.axvline(ev_s_min, color="#bcbd22" if "no_stress" in cat else "#e31a1c", linestyle="-", linewidth=1.5)
            ax.axvspan(ev_s_min, ev_e_min, color=color_sh, alpha=0.25)

    save_fig(fig, "09_real_session_forecast_timeline.png")

    # 10. Final Model Feature Importance (Standardized Logistic Regression Coefficients)
    clf_lr = final_lr_pipe.named_steps["clf"]
    coefs = clf_lr.coef_.reshape(-1)
    imp_df = pd.DataFrame({"feature": final_cols, "coef": coefs, "abs_coef": np.abs(coefs)})
    imp_df = imp_df.sort_values("abs_coef", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in imp_df["coef"]]
    ax.barh(imp_df["feature"], imp_df["coef"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized Logistic Regression Weight (Red = Increases Pre-Event Risk)")
    ax.set_title("Top 20 Standardized Feature Weights — Final Nurses Stress Forecasting Model (M1 LR, P1)")
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, "10_final_model_feature_importance.png")


def main():
    t_start_all = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MOD_DIR.mkdir(parents=True, exist_ok=True)

    features_csv = OUT_DIR / "features.csv"
    print("=== Step 1 & 2: Dataset Audit & Causal Feature Extraction ===", flush=True)
    feat_df, manifest, subj_df, ev_df = generate_nurse_features_table(
        cache_csv=features_csv, window_s=180.0
    )

    (OUT_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    subj_df.to_csv(OUT_DIR / "subject_summary.csv", index=False)
    ev_df.to_csv(OUT_DIR / "event_audit.csv", index=False)

    print(
        f"Features ready: {len(feat_df)} total windows across {feat_df['subject'].nunique()} subjects "
        f"and {feat_df['session'].nunique()} annotated sessions.",
        flush=True,
    )
    print("Window state distribution:\n", feat_df["state"].value_counts().to_string(), flush=True)

    # 3. Model Comparison on Primary Task P1 (60-180s early warning)
    print("\n=== Step 3: Core Model Comparison on P1 (Dummy, LR, RF, LightGBM) ===", flush=True)
    model_comp_path = OUT_DIR / "model_comparison.csv"
    per_subj_path = OUT_DIR / "per_subject_metrics.csv"
    preds_path = OUT_DIR / "predictions.csv"

    if model_comp_path.exists() and per_subj_path.exists() and preds_path.exists():
        model_comp_df = pd.read_csv(model_comp_path)
        per_subj_p1_df = pd.read_csv(per_subj_path, dtype={"subject": str})
        combined_preds_p1 = pd.read_csv(preds_path, dtype={"subject": str, "session": str})
        all_preds_p1 = {
            m: combined_preds_p1[combined_preds_p1["model"] == m].copy()
            for m in ["Dummy", "LR", "RF", "LightGBM"]
        }
        print("  Loaded cached P1 model_comparison.csv, per_subject_metrics.csv, predictions.csv", flush=True)
    else:
        models_to_run = ["Dummy", "LR", "RF", "LightGBM"]
        model_comp_rows = []
        all_per_subj_p1 = []
        all_preds_p1 = {}
        splits_saved = False

        for m_name in models_to_run:
            t0 = time.perf_counter()
            ps_df, pred_df, ev_m, splits_meta = run_loso_forecast_experiment(
                feat_df, ev_df, task="P1", model_name=m_name, sensor_set="all"
            )
            if not splits_saved:
                pd.DataFrame(splits_meta).to_csv(OUT_DIR / "splits.csv", index=False)
                splits_saved = True

            all_per_subj_p1.append(ps_df)
            all_preds_p1[m_name] = pred_df

            agg = evaluate_predictions_with_event_metrics(
                pred_df, ev_df, task="P1", pred_col="pred", prob_col="prob", label_tag=m_name
            )
            agg["task"] = "P1"
            agg["model"] = m_name
            agg["sensors"] = "all"
            agg["fit_s_total"] = float(ps_df["fit_s"].sum())
            agg["infer_ms_per_win_median"] = float(ps_df["infer_ms_per_win"].median())
            model_comp_rows.append(agg)
            print(
                f"  [P1 {m_name}] Mean AUROC={agg['auroc_mean']:.4f} (Med={agg['auroc_median']:.4f}), "
                f"Mean AUPRC={agg['auprc_mean']:.4f}, F1={agg['f1_mean']:.4f}, "
                f"Event Recall={agg['event_recall']:.4f} ({agg['n_detected_events']}/{agg['n_total_events']}), "
                f"FA/hr={agg['false_alarms_per_hour']:.4f}, Lead Mean={agg['lead_time_mean_s']:.1f}s "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

        model_comp_df = pd.DataFrame(model_comp_rows)
        model_comp_df.to_csv(model_comp_path, index=False)

        per_subj_p1_df = pd.concat(all_per_subj_p1, ignore_index=True)
        per_subj_p1_df.to_csv(per_subj_path, index=False)

        combined_preds_p1 = pd.concat(list(all_preds_p1.values()), ignore_index=True)
        combined_preds_p1.to_csv(preds_path, index=False)

    # 4. Lead Time Horizon Comparison (P0: 0-60s, P1: 60-180s, P2: 180-300s)
    print("\n=== Step 4: Lead Time Comparison (P0, P1, P2) ===", flush=True)
    lead_csv_path = OUT_DIR / "lead_time_comparison.csv"
    if lead_csv_path.exists():
        lead_comp_df = pd.read_csv(lead_csv_path)
        print("  Loaded cached lead_time_comparison.csv", flush=True)
    else:
        lead_rows = []
        for task_id in ["P0", "P1", "P2"]:
            for m_name in ["Dummy", "LR", "RF", "LightGBM"]:
                if task_id == "P1" and m_name in all_preds_p1:
                    pred_df = all_preds_p1[m_name]
                else:
                    _, pred_df, _, _ = run_loso_forecast_experiment(
                        feat_df, ev_df, task=task_id, model_name=m_name, sensor_set="all"
                    )
                agg = evaluate_predictions_with_event_metrics(
                    pred_df, ev_df, task=task_id, pred_col="pred", prob_col="prob", label_tag=f"{task_id}_{m_name}"
                )
                agg["task"] = task_id
                agg["model"] = m_name
                agg["min_lead_s"] = 0 if task_id == "P0" else (60 if task_id == "P1" else 180)
                agg["max_lead_s"] = 60 if task_id == "P0" else (180 if task_id == "P1" else 300)
                lead_rows.append(agg)
                print(
                    f"  [{task_id} {m_name}] AUROC={agg['auroc_mean']:.4f}, AUPRC={agg['auprc_mean']:.4f}, "
                    f"Event Recall={agg['event_recall']:.4f}, Lead Mean={agg['lead_time_mean_s']:.1f}s, "
                    f"FA/hr={agg['false_alarms_per_hour']:.4f}",
                    flush=True,
                )

        lead_comp_df = pd.DataFrame(lead_rows)
        lead_comp_df.to_csv(lead_csv_path, index=False)

    # 5. Sensor Ablation on P1 (EDA, PPG, ACC, EDA+PPG, ALL) for LR and RF
    print("\n=== Step 5: Sensor Ablation (EDA, PPG, ACC, EDA+PPG, ALL) ===", flush=True)
    sensor_csv_path = OUT_DIR / "sensor_ablation.csv"
    if sensor_csv_path.exists():
        sensor_abl_df = pd.read_csv(sensor_csv_path)
        print("  Loaded cached sensor_ablation.csv", flush=True)
    else:
        sensor_sets = ["eda", "ppg", "acc", "eda_ppg", "all"]
        sensor_rows = []
        for m_name in ["LR", "RF"]:
            for s_set in sensor_sets:
                if s_set == "all" and m_name in all_preds_p1:
                    pred_df = all_preds_p1[m_name]
                else:
                    _, pred_df, _, _ = run_loso_forecast_experiment(
                        feat_df, ev_df, task="P1", model_name=m_name, sensor_set=s_set
                    )
                agg = evaluate_predictions_with_event_metrics(
                    pred_df, ev_df, task="P1", pred_col="pred", prob_col="prob", label_tag=f"{m_name}_{s_set}"
                )
                agg["model"] = m_name
                agg["sensors"] = s_set
                sensor_rows.append(agg)
                print(
                    f"  [Sensor {m_name} - {s_set}] AUROC={agg['auroc_mean']:.4f}, AUPRC={agg['auprc_mean']:.4f}, "
                    f"Event Recall={agg['event_recall']:.4f}, FA/hr={agg['false_alarms_per_hour']:.4f}",
                    flush=True,
                )

        sensor_abl_df = pd.DataFrame(sensor_rows)
        sensor_abl_df.to_csv(sensor_csv_path, index=False)

    # 6. Experiment A: Motion Gate Comparison (None, Hard Gate, Soft Dynamic Gate)
    print("\n=== Step 6: Experiment A - Physical Activity / Motion Gating ===", flush=True)
    motion_rows = []
    for m_name in ["LR", "RF", "LightGBM"]:
        base_pred = all_preds_p1[m_name].copy()
        base_pred["motion_quartile"] = pd.qcut(
            base_pred["acc_motion_change_per_s"].rank(method="first"),
            q=4,
            labels=["Q1_Quiet", "Q2_Low", "Q3_Mod", "Q4_High"],
        )
        for gate_mode in ["none", "legacy_hard", "soft_dynamic"]:
            gdf = base_pred.copy()
            p_g_list = []
            pr_g_list = []
            for sid, sg in gdf.groupby("subject", sort=False):
                thr_s = float(sg["threshold"].iloc[0])
                pg, dthr, prg = apply_dynamic_motion_gate(
                    sg["prob"].to_numpy(dtype=float), sg, mode=gate_mode, base_threshold=thr_s
                )
                sg_out = sg.copy()
                sg_out["prob_gated"] = pg
                sg_out["pred_gated"] = prg
                p_g_list.append(sg_out)
            gdf = pd.concat(p_g_list, ignore_index=True)

            agg = evaluate_predictions_with_event_metrics(
                gdf, ev_df, task="P1", pred_col="pred_gated", prob_col="prob_gated", label_tag=f"{m_name}_{gate_mode}"
            )
            # Activity quartile alert rates on non-event windows
            neg_g = gdf[gdf["y_true"] == 0]
            q_rates = neg_g.groupby("motion_quartile", observed=False)["pred_gated"].mean().to_dict()
            agg["model"] = m_name
            agg["motion_gate"] = gate_mode
            agg["q1_quiet_alert_rate"] = float(q_rates.get("Q1_Quiet", 0.0))
            agg["q2_low_alert_rate"] = float(q_rates.get("Q2_Low", 0.0))
            agg["q3_mod_alert_rate"] = float(q_rates.get("Q3_Mod", 0.0))
            agg["q4_high_alert_rate"] = float(q_rates.get("Q4_High", 0.0))
            motion_rows.append(agg)
            print(
                f"  [Motion Gate {m_name} - {gate_mode}] AUROC={agg['auroc_mean']:.4f}, "
                f"Event Recall={agg['event_recall']:.4f} (Missed={agg['n_missed_events']}), "
                f"Win FPR={agg['fpr_mean']:.4f}, FA/hr={agg['false_alarms_per_hour']:.4f}, "
                f"Q4 High-Activity FPR={agg['q4_high_alert_rate']:.4f}",
                flush=True,
            )

    motion_comp_df = pd.DataFrame(motion_rows)
    motion_comp_df.to_csv(OUT_DIR / "motion_gate_comparison.csv", index=False)

    # 7. Experiment B: Alert Policy Trade-offs
    print("\n=== Step 7: Experiment B - Alert Policy Trade-offs ===", flush=True)
    policies = [
        "fixed_05",
        "val_opt",
        "hysteresis",
        "duration_2win",
        "cooldown_300s",
        "soft_gate_hysteresis",
    ]
    policy_rows = []
    base_val_recall = None
    base_val_fa = None

    for m_name in ["LR", "RF"]:
        base_pred = all_preds_p1[m_name].copy()
        for pol in policies:
            pdf_list = []
            for (sid, sess), sg in base_pred.groupby(["subject", "session"], sort=False):
                sg = sg.sort_values("pred_ts").copy()
                thr_s = float(sg["threshold"].iloc[0])
                p_pol = apply_alert_policy_to_session(
                    sg["prob"].to_numpy(dtype=float),
                    sg["end_s"].to_numpy(dtype=float),
                    val_threshold=thr_s,
                    policy=pol,
                    df_session=sg,
                )
                sg["pred_pol"] = p_pol
                pdf_list.append(sg)
            pdf = pd.concat(pdf_list, ignore_index=True)

            agg = evaluate_predictions_with_event_metrics(
                pdf, ev_df, task="P1", pred_col="pred_pol", prob_col="prob", label_tag=f"{m_name}_{pol}"
            )
            agg["model"] = m_name
            agg["policy"] = pol
            agg["policy_failed_zero_recall"] = bool(agg["event_recall"] <= 0.0)
            if pol == "val_opt":
                base_val_recall = agg["event_recall"]
                base_val_fa = agg["false_alarms_per_hour"]
            agg["delta_event_recall_vs_val_opt"] = (
                float(agg["event_recall"] - base_val_recall) if base_val_recall is not None else 0.0
            )
            agg["delta_fa_per_hr_vs_val_opt"] = (
                float(agg["false_alarms_per_hour"] - base_val_fa) if base_val_fa is not None else 0.0
            )
            policy_rows.append(agg)
            print(
                f"  [Policy {m_name} - {pol}] Event Recall={agg['event_recall']:.4f} "
                f"(Delta={agg['delta_event_recall_vs_val_opt']:+.4f}), "
                f"FA/hr={agg['false_alarms_per_hour']:.4f} (Delta={agg['delta_fa_per_hr_vs_val_opt']:+.4f}), "
                f"Lead Mean={agg['lead_time_mean_s']:.1f}s",
                flush=True,
            )

    policy_comp_df = pd.DataFrame(policy_rows)
    policy_comp_df.to_csv(OUT_DIR / "alert_policy_comparison.csv", index=False)

    # 8. Experiment C & D: Personal Baseline Calibration & Generalization to Unseen Subjects
    print("\n=== Step 8: Experiments C & D - Personal Baseline Calibration & Generalization ===", flush=True)
    cal_methods = [
        ("uncalibrated", 0.0),
        ("cal_1m", 60.0),
        ("cal_3m", 180.0),
        ("cal_5m", 300.0),
        ("low_motion", 180.0),
        ("adaptive_fallback", 180.0),
    ]
    # Use physiological + calibrated delta features + ACC features so calibration delta directly drives prediction
    cal_eval_cols = list(CALIBRATED_PHYSIO_FEATURES) + [
        "eda_recent60_minus_prev120",
        "eda_log_slope",
        "eda_rise_fraction",
        "hr_slope_bpm_per_s",
        "hr_delta_recent60_bpm",
        "acc_motion_change_per_s",
        "acc_unchanged_fraction",
    ]
    cal_rows = []
    for c_name, c_dur in cal_methods:
        cal_df = apply_session_causal_calibration(feat_df, method=c_name, duration_s=c_dur)
        for m_name in ["LR", "RF"]:
            _, pred_df, _, _ = run_loso_forecast_experiment(
                cal_df,
                ev_df,
                task="P1",
                model_name=m_name,
                sensor_set="all",
                feature_cols_override=cal_eval_cols,
            )
            # Also evaluate with Soft Dynamic Gate combined (Experiment C: Common + Calib + Motion Gate)
            p_g_list = []
            for sid, sg in pred_df.groupby("subject", sort=False):
                thr_s = float(sg["threshold"].iloc[0])
                pg, _, prg = apply_dynamic_motion_gate(
                    sg["prob"].to_numpy(dtype=float), sg, mode="soft_dynamic", base_threshold=thr_s
                )
                sg_out = sg.copy()
                sg_out["prob_soft"] = pg
                sg_out["pred_soft"] = prg
                p_g_list.append(sg_out)
            pred_soft_df = pd.concat(p_g_list, ignore_index=True)

            agg_raw = evaluate_predictions_with_event_metrics(
                pred_df, ev_df, task="P1", pred_col="pred", prob_col="prob", label_tag=f"{m_name}_{c_name}_nogate"
            )
            agg_raw["model"] = m_name
            agg_raw["calibration_method"] = c_name
            agg_raw["calibration_duration_s"] = c_dur
            agg_raw["motion_gate"] = "none"
            cal_rows.append(agg_raw)

            agg_soft = evaluate_predictions_with_event_metrics(
                pred_soft_df, ev_df, task="P1", pred_col="pred_soft", prob_col="prob_soft", label_tag=f"{m_name}_{c_name}_softgate"
            )
            agg_soft["model"] = m_name
            agg_soft["calibration_method"] = c_name
            agg_soft["calibration_duration_s"] = c_dur
            agg_soft["motion_gate"] = "soft_dynamic"
            cal_rows.append(agg_soft)

            if m_name == "LR":
                print(
                    f"  [Calib LR - {c_name:17s}] NoGate AUROC={agg_raw['auroc_mean']:.4f}, "
                    f"WorstSubj={agg_raw['worst_subject']} ({agg_raw['worst_subject_auroc']:.4f}), "
                    f"FA/hr={agg_raw['false_alarms_per_hour']:.4f} | "
                    f"SoftGate AUROC={agg_soft['auroc_mean']:.4f}, EventRec={agg_soft['event_recall']:.4f}, "
                    f"FA/hr={agg_soft['false_alarms_per_hour']:.4f}",
                    flush=True,
                )

    calib_comp_df_all = pd.DataFrame(cal_rows)
    calib_comp_df_all.to_csv(OUT_DIR / "calibration_comparison.csv", index=False)
    calib_lr_sub = calib_comp_df_all[
        (calib_comp_df_all["model"] == "LR") & (calib_comp_df_all["motion_gate"] == "soft_dynamic")
    ].reset_index(drop=True)

    # Save event_metrics.csv consolidating key event-level results across experiments
    ev_summary_rows = []
    for _, r in model_comp_df.iterrows():
        ev_summary_rows.append({"experiment": "Model_Comparison_P1", "variant": r["model"], **r.to_dict()})
    for _, r in lead_comp_df[lead_comp_df["model"] == "LR"].iterrows():
        ev_summary_rows.append({"experiment": "Lead_Time_LR", "variant": r["task"], **r.to_dict()})
    for _, r in motion_comp_df[motion_comp_df["model"] == "LR"].iterrows():
        ev_summary_rows.append({"experiment": "Motion_Gate_LR", "variant": r["motion_gate"], **r.to_dict()})
    for _, r in policy_comp_df[policy_comp_df["model"] == "LR"].iterrows():
        ev_summary_rows.append({"experiment": "Alert_Policy_LR", "variant": r["policy"], **r.to_dict()})
    pd.DataFrame(ev_summary_rows).to_csv(OUT_DIR / "event_metrics.csv", index=False)

    # 9. Train & Save Final Production Model ("Nurses 기반 미래 스트레스 사건 예측 모델")
    print("\n=== Step 9: Saving Final Model & Deployment Package ===", flush=True)
    p1_elig_df = feat_df[feat_df["qc_common_valid"].astype(bool) & feat_df["y_P1"].isin([0, 1])].copy()
    final_cols = [c for c in SENSOR_FEATURE_MAP["all"] if c in p1_elig_df.columns]

    final_lr_pipe = build_forecast_model("LR", n_jobs=4)
    final_lr_pipe.fit(p1_elig_df[final_cols], p1_elig_df["y_P1"].to_numpy(dtype=int))

    final_rf_pipe = build_forecast_model("RF", n_jobs=4)
    final_rf_pipe.fit(p1_elig_df[final_cols], p1_elig_df["y_P1"].to_numpy(dtype=int))

    final_lgbm_pipe = build_forecast_model("LightGBM", n_jobs=4)
    final_lgbm_pipe.fit(p1_elig_df[final_cols], p1_elig_df["y_P1"].to_numpy(dtype=int))

    feat_sha256 = hashlib.sha256(features_csv.read_bytes()).hexdigest()
    opt_thr = float(per_subj_p1_df[per_subj_p1_df["model"] == "LR"]["threshold"].median())

    lr_artifact = {
        "model_name": "Nurses 기반 미래 스트레스 사건 예측 모델 (M1 Logistic Regression + Low-Motion Calibration + Soft Dynamic Gate)",
        "model_version": "1.0.0-nurse-p1",
        "task": "P1_future_stress_event_onset_60_to_180s",
        "pipeline": final_lr_pipe,
        "feature_columns": final_cols,
        "n_features": len(final_cols),
        "decision_threshold": opt_thr,
        "alert_policy": {
            "type": "soft_gate_hysteresis",
            "t_on": max(opt_thr, 0.50),
            "t_off": min(opt_thr * 0.70, 0.35),
            "episode_merge_gap_s": 180.0,
        },
        "personal_calibration": {
            "method": "low_motion",
            "duration_s": 180.0,
            "max_jerk_threshold": 38.0,
            "causal_strict": True,
        },
        "motion_gate": {
            "mode": "soft_dynamic",
            "attenuation_formula": "p_gated = p * (1 - 0.85 * M(t)^2)",
            "dynamic_threshold_formula": "theta(t) = clip(base_thr + 0.35 * M(t), 0.05, 0.92)",
        },
        "sensor_specs": {
            "EDA": {"sampling_rate_hz": 4.0, "unit": "microSiemens (uS)"},
            "BVP": {"sampling_rate_hz": 64.0, "unit": "PPG raw"},
            "ACC": {"sampling_rate_hz": 32.0, "unit": "1/64g (3-axis)"},
            "HR": {"sampling_rate_hz": 1.0, "unit": "bpm"},
        },
        "training_data_hash_sha256": feat_sha256,
        "n_training_subjects": 15,
        "n_training_windows": len(p1_elig_df),
        "disclaimer": "Nurses 기반 미래 스트레스 사건 예측 모델이며, 실제 자폐 아동 멜트다운 예측 모델이 아님.",
    }

    lr_model_path = MOD_DIR / "nurse_stress_forecast_lr_p1.joblib"
    rf_model_path = MOD_DIR / "nurse_stress_forecast_rf_p1.joblib"
    lgbm_model_path = MOD_DIR / "nurse_stress_forecast_lgbm_p1.joblib"

    joblib.dump(lr_artifact, lr_model_path)
    joblib.dump({"model_name": "Nurses RF P1", "pipeline": final_rf_pipe, "feature_columns": final_cols}, rf_model_path)
    joblib.dump({"model_name": "Nurses LightGBM P1", "pipeline": final_lgbm_pipe, "feature_columns": final_cols}, lgbm_model_path)

    # Export C header weights for ESP32-S3 feasibility verification
    imputer = final_lr_pipe.named_steps["imputer"]
    scaler = final_lr_pipe.named_steps["scaler"]
    clf = final_lr_pipe.named_steps["clf"]
    n_f = len(final_cols)
    raw_weight_bytes = (n_f * 4) * 3 + 4  # median, mean, scale, coef (4 arrays of float32) + intercept

    spec_meta = {
        "model_name": lr_artifact["model_name"],
        "model_version": lr_artifact["model_version"],
        "task": lr_artifact["task"],
        "n_features": n_f,
        "feature_columns": final_cols,
        "decision_threshold": opt_thr,
        "alert_policy": lr_artifact["alert_policy"],
        "personal_calibration": lr_artifact["personal_calibration"],
        "motion_gate": lr_artifact["motion_gate"],
        "sensor_specs": lr_artifact["sensor_specs"],
        "training_data_hash_sha256": feat_sha256,
        "file_sizes_bytes": {
            "lr_joblib_bytes": int(lr_model_path.stat().st_size),
            "rf_joblib_bytes": int(rf_model_path.stat().st_size),
            "lgbm_joblib_bytes": int(lgbm_model_path.stat().st_size),
            "lr_mcu_float32_arrays_bytes": int((n_f * 4 * 4) + 4),
            "raw_180s_ring_buffer_bytes": int((4 * 180 + 64 * 180 + 32 * 3 * 180) * 4),
        },
        "coefficients": {col: float(w) for col, w in zip(final_cols, clf.coef_.reshape(-1))},
        "intercept": float(clf.intercept_[0]),
        "scaler_mean": {col: float(m) for col, m in zip(final_cols, scaler.mean_)},
        "scaler_scale": {col: float(s) for col, s in zip(final_cols, scaler.scale_)},
        "imputer_median": {col: float(m) for col, m in zip(final_cols, imputer.statistics_)},
    }
    (MOD_DIR / "nurse_stress_forecast_lr_p1_spec.json").write_text(
        json.dumps(spec_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 10. Generate all 10 figures
    print("\n=== Step 10: Generating 10 Visual Evidence Figures ===", flush=True)
    make_all_figures(
        feat_df=feat_df,
        ev_df=ev_df,
        model_comp_df=model_comp_df,
        lead_comp_df=lead_comp_df,
        sensor_abl_df=sensor_abl_df,
        motion_comp_df=motion_comp_df,
        policy_comp_df=policy_comp_df,
        calib_comp_df=calib_lr_sub,
        per_subj_p1_df=per_subj_p1_df,
        preds_by_model=all_preds_p1,
        final_lr_pipe=final_lr_pipe,
        final_cols=final_cols,
    )

    print(
        f"\nALL EXPERIMENTS COMPLETED SUCCESSFULLY in {time.perf_counter() - t_start_all:.1f}s!",
        flush=True,
    )


if __name__ == "__main__":
    main()
