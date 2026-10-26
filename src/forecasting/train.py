"""LOSO model training and evaluation pipeline for Nurses Stress future event forecasting.

Implements:
- M0: DummyClassifier (prior baseline)
- M1: LogisticRegression (StandardScaler + class_weight='balanced', C=0.25)
- M2: RandomForestClassifier (n_estimators=200, max_depth=6, min_samples_leaf=8, class_weight='balanced_subsample')
- M3: LGBMClassifier (n_estimators=120, max_depth=4, num_leaves=15, learning_rate=0.05, class_weight='balanced')

Strict leakage prevention:
- 15-fold Leave-One-Subject-Out (LOSO) outer split.
- Inner GroupShuffleSplit on training subjects only (never touches test subject) to select optimal decision threshold.
- Imputation and scaling are fit strictly inside each fold's training set.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.advanced import apply_dynamic_motion_gate, compute_causal_exercise_score
from src.features.nurse_forecasting import (
    SENSOR_FEATURE_MAP,
    extract_nurse_window_features,
)
from src.forecasting.alerts import (
    apply_alert_policy_to_session,
    evaluate_event_level_forecasting,
)
from src.forecasting.calibration import apply_session_causal_calibration
from src.forecasting.evaluate import compute_window_metrics, summarize_subject_metrics
from src.forecasting.labels import build_session_prediction_grid
from src.loaders.nurse_stress import (
    audit_nurse_dataset,
    load_nurse_session_zip,
    resolve_nurse_root,
)


def build_forecast_model(model_name: str, n_jobs: int = 4) -> Pipeline:
    """Build scikit-learn Pipeline with median imputation and scaling."""
    if model_name == "Dummy":
        clf = DummyClassifier(strategy="prior")
    elif model_name == "LR":
        clf = LogisticRegression(C=0.25, class_weight="balanced", max_iter=2000, random_state=42)
    elif model_name == "RF":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=n_jobs,
        )
    elif model_name == "LightGBM":
        clf = LGBMClassifier(
            n_estimators=120,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=n_jobs,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def select_validation_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Select decision threshold maximizing F1 on inner validation subjects only."""
    y = np.asarray(y_val, dtype=int)
    p = np.asarray(p_val, dtype=float)
    thresholds = np.linspace(0.20, 0.85, 66)
    best_t = 0.50
    best_f1 = -1.0
    for t in thresholds:
        score = f1_score(y, (p >= t).astype(int), zero_division=0)
        if score >= best_f1:
            best_f1 = float(score)
            best_t = float(t)
    return best_t


def generate_nurse_features_table(
    root_dir: str | Path | None = None,
    cache_csv: str | Path | None = None,
    window_s: float = 180.0,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Extract or load cached causal window features across all 214 annotated Nurse sessions."""
    manifest, subj_df, ev_df = audit_nurse_dataset(root_dir)
    if cache_csv is not None and Path(cache_csv).exists():
        feat_df = pd.read_csv(cache_csv, dtype={"subject": str, "session": str})
        return feat_df, manifest, subj_df, ev_df

    sessions_meta = [s for s in manifest["sessions"] if s["has_survey_annotation"]]
    all_rows = []
    t0_wall = time.perf_counter()

    for idx_s, sm in enumerate(sessions_meta):
        sid = str(sm["subject"])
        sess_id = str(sm["session"])
        zpath = sm["zip_path"]
        sub_evs = ev_df[ev_df["subject"] == sid]

        sdata = load_nurse_session_zip(zpath, include_temp=False)
        grid = build_session_prediction_grid(
            sdata["start_ts"],
            sdata["end_ts"],
            sub_evs,
            window_s=window_s,
            event_Zone_step_s=30.0,
            background_step_s=60.0,
        )
        if not grid:
            continue

        for g in grid:
            tic = time.perf_counter()
            feats = extract_nurse_window_features(sdata, g["pred_rel_s"], window_s=window_s)
            elapsed_ms = (time.perf_counter() - tic) * 1000.0
            row = {
                "dataset": "Nurse_Stress",
                "subject": sid,
                "session": sess_id,
                "pred_ts": g["pred_ts"],
                "pred_rel_s": g["pred_rel_s"],
                "start_s": g["start_s"],
                "end_s": g["end_s"],
                "state": g["state"],
                "target_event_id": g["target_event_id"],
                "lead_time_to_onset_s": g["lead_time_to_onset_s"],
                "y_P0": g["y_P0"],
                "y_P1": g["y_P1"],
                "y_P2": g["y_P2"],
                "feature_extract_ms": elapsed_ms,
                **feats,
            }
            all_rows.append(row)

        if (idx_s + 1) % 25 == 0 or (idx_s + 1) == len(sessions_meta):
            print(
                f"[Feature Extraction] {idx_s + 1}/{len(sessions_meta)} sessions done "
                f"({len(all_rows)} windows, {time.perf_counter() - t0_wall:.1f}s)",
                flush=True,
            )

    feat_df = pd.DataFrame(all_rows)
    feat_df = feat_df.sort_values(["subject", "session", "pred_ts"]).reset_index(drop=True)
    # Apply default causal personal calibration ('low_motion' 180s)
    feat_df = apply_session_causal_calibration(feat_df, method="low_motion", duration_s=180.0)
    feat_df["exercise_score"] = compute_causal_exercise_score(feat_df)

    if cache_csv is not None:
        Path(cache_csv).parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(cache_csv, index=False)

    return feat_df, manifest, subj_df, ev_df


def run_loso_forecast_experiment(
    df: pd.DataFrame,
    ev_df: pd.DataFrame,
    task: str = "P1",
    model_name: str = "LR",
    sensor_set: str = "all",
    feature_cols_override: list[str] | None = None,
    n_jobs: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    """Run 15-fold LOSO cross-validation for a given forecast task (P0, P1, or P2), model, and sensor set."""
    y_col = f"y_{task}"
    elig_col = f"{task.lower()}_forecast_eligible"

    # Eligible windows for this task: qc_common_valid == True and y_col in {0, 1}
    qc_ok = df["qc_common_valid"].astype(bool)
    task_mask = qc_ok & df[y_col].isin([0, 1])
    work = df[task_mask].copy().reset_index(drop=True)

    cols = (
        feature_cols_override
        if feature_cols_override is not None
        else [c for c in SENSOR_FEATURE_MAP[sensor_set] if c in work.columns]
    )
    if not cols:
        raise ValueError(f"No valid feature columns for sensor_set={sensor_set}")

    subjects = sorted(work["subject"].unique())
    logo = LeaveOneGroupOut()

    per_subj_rows = []
    pred_dfs = []
    splits_meta = []

    for fold, (dev_idx, test_idx) in enumerate(logo.split(work, work[y_col], work["subject"])):
        dev_df = work.iloc[dev_idx]
        test_df = work.iloc[test_idx].copy()
        test_subj = str(test_df["subject"].iloc[0])

        # Inner group split on dev subjects for threshold selection
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42 + fold)
        tr_rel, val_rel = next(gss.split(dev_df, dev_df[y_col], dev_df["subject"]))
        tr_df = dev_df.iloc[tr_rel]
        val_df = dev_df.iloc[val_rel]

        tr_subjs = sorted(tr_df["subject"].unique().tolist())
        val_subjs = sorted(val_df["subject"].unique().tolist())
        assert test_subj not in tr_subjs and test_subj not in val_subjs
        assert not (set(tr_subjs) & set(val_subjs))

        splits_meta.append(
            {
                "task": task,
                "fold": fold,
                "test_subject": test_subj,
                "n_train_subjects": len(tr_subjs),
                "n_val_subjects": len(val_subjs),
                "train_subjects": ";".join(tr_subjs),
                "val_subjects": ";".join(val_subjs),
                "n_train_windows": len(tr_df),
                "n_val_windows": len(val_df),
                "n_test_windows": len(test_df),
            }
        )

        pipe = build_forecast_model(model_name, n_jobs=n_jobs)
        t0 = time.perf_counter()
        pipe.fit(tr_df[cols], tr_df[y_col].to_numpy(dtype=int))
        fit_s = time.perf_counter() - t0

        if model_name == "Dummy":
            val_thr = 0.50
        else:
            p_val = pipe.predict_proba(val_df[cols])[:, 1]
            val_thr = select_validation_threshold(val_df[y_col].to_numpy(dtype=int), p_val)

        # Refit on all 14 dev subjects with the selected validation threshold
        pipe_full = build_forecast_model(model_name, n_jobs=n_jobs)
        pipe_full.fit(dev_df[cols], dev_df[y_col].to_numpy(dtype=int))

        t_inf = time.perf_counter()
        p_test = pipe_full.predict_proba(test_df[cols])[:, 1]
        infer_ms_per_win = ((time.perf_counter() - t_inf) * 1000.0) / max(len(test_df), 1)

        if model_name == "Dummy":
            pred_test = np.zeros(len(test_df), dtype=int)
        else:
            pred_test = (p_test >= val_thr).astype(int)

        test_df["task"] = task
        test_df["model"] = model_name
        test_df["sensors"] = sensor_set
        test_df["fold"] = fold
        test_df["y_true"] = test_df[y_col].to_numpy(dtype=int)
        test_df["prob"] = p_test
        test_df["threshold"] = val_thr
        test_df["pred"] = pred_test
        test_df["pred_fixed05"] = (p_test >= 0.50).astype(int)

        w_metrics = compute_window_metrics(test_df["y_true"].to_numpy(), p_test, pred_test)

        # Event-level metrics for this subject
        sub_elig_evs = ev_df[(ev_df["subject"] == test_subj) & (ev_df[elig_col] == True)]
        max_lead = 60.0 if task == "P0" else (180.0 if task == "P1" else 300.0)
        min_lead = 0.0 if task == "P0" else (60.0 if task == "P1" else 180.0)
        ev_metrics = evaluate_event_level_forecasting(
            test_df,
            sub_elig_evs,
            pred_col="pred",
            max_valid_lead_s=max_lead,
            min_valid_lead_s=min_lead,
        )

        per_subj_rows.append(
            {
                "task": task,
                "model": model_name,
                "sensors": sensor_set,
                "fold": fold,
                "subject": test_subj,
                "threshold": val_thr,
                "fit_s": fit_s,
                "infer_ms_per_win": infer_ms_per_win,
                **w_metrics,
                **ev_metrics,
            }
        )
        pred_dfs.append(test_df)

    per_subj_df = pd.DataFrame(per_subj_rows)
    full_pred_df = pd.concat(pred_dfs, ignore_index=True)

    # Cohort-wide event-level evaluation across all 15 subjects
    all_elig_evs = ev_df[ev_df[elig_col] == True]
    max_lead = 60.0 if task == "P0" else (180.0 if task == "P1" else 300.0)
    min_lead = 0.0 if task == "P0" else (60.0 if task == "P1" else 180.0)
    cohort_ev_metrics = evaluate_event_level_forecasting(
        full_pred_df,
        all_elig_evs,
        pred_col="pred",
        max_valid_lead_s=max_lead,
        min_valid_lead_s=min_lead,
    )

    return per_subj_df, full_pred_df, cohort_ev_metrics, splits_meta
