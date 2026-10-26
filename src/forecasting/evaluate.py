"""Window-level and event-level evaluation utilities for Nurses Stress forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_window_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    """Compute comprehensive window-level classification metrics."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    yh = np.asarray(pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    has_pos = (y == 1).any()
    has_neg = (y == 0).any()
    has_pred_pos = (yh == 1).any()

    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan
    rec = float(recall_score(y, yh, zero_division=0)) if has_pos else np.nan
    fnr = float(1.0 - rec) if np.isfinite(rec) else np.nan

    return {
        "n_windows": int(len(y)),
        "pos_windows": int((y == 1).sum()),
        "neg_windows": int((y == 0).sum()),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "auroc": float(roc_auc_score(y, p)) if both else np.nan,
        "auprc": float(average_precision_score(y, p)) if both else np.nan,
        "accuracy": float(accuracy_score(y, yh)) if len(y) else np.nan,
        "balanced_acc": float(balanced_accuracy_score(y, yh)) if both else np.nan,
        "precision": float(precision_score(y, yh, zero_division=0)) if has_pred_pos else np.nan,
        "recall": rec,
        "specificity": spec,
        "f1": float(f1_score(y, yh, zero_division=0)) if has_pos else np.nan,
        "fpr": fpr,
        "fnr": fnr,
        "brier": float(brier_score_loss(y, np.clip(p, 0.0, 1.0))) if len(y) else np.nan,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def summarize_subject_metrics(per_subj_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Summarize subject-level metrics with mean, median, std, min, max, and worst subject."""
    metric_cols = [
        "auroc",
        "auprc",
        "accuracy",
        "balanced_acc",
        "precision",
        "recall",
        "specificity",
        "f1",
        "fpr",
        "fnr",
        "brier",
    ]
    rows = []
    for keys, g in per_subj_df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_subjects"] = int(g["subject"].nunique())
        row["total_windows"] = int(g["n_windows"].sum())
        row["total_pos_windows"] = int(g["pos_windows"].sum())
        row["total_neg_windows"] = int(g["neg_windows"].sum())
        row["tp_sum"] = int(g["tp"].sum())
        row["fp_sum"] = int(g["fp"].sum())
        row["tn_sum"] = int(g["tn"].sum())
        row["fn_sum"] = int(g["fn"].sum())

        for mc in metric_cols:
            vals = g[mc].dropna().to_numpy(dtype=float)
            row[f"{mc}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{mc}_median"] = float(np.median(vals)) if len(vals) else np.nan
            row[f"{mc}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{mc}_min"] = float(np.min(vals)) if len(vals) else np.nan
            row[f"{mc}_max"] = float(np.max(vals)) if len(vals) else np.nan

        valid_auc = g.dropna(subset=["auroc"])
        if not valid_auc.empty:
            worst_idx = valid_auc["auroc"].idxmin()
            row["worst_subject"] = str(valid_auc.loc[worst_idx, "subject"])
            row["worst_subject_auroc"] = float(valid_auc.loc[worst_idx, "auroc"])
        else:
            row["worst_subject"] = "N/A"
            row["worst_subject_auroc"] = np.nan

        rows.append(row)
    return pd.DataFrame(rows)
