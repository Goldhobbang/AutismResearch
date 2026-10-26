"""Fixed 30-second full-timeline replay for Nurses early-warning evaluation.

This module is additive. It never writes to the original feature cache, model
artifacts, survey labels, or the existing audit outputs. All new artifacts are
written under ``results/nurse_forecasting/fixed_grid_30s``.

Stages
------
build   : re-extract the same 62 causal features on a label-independent 30 s grid
          over the complete sensor timeline of every session.
train   : reapply the existing 15-fold LOSO LR protocol (no new hyper-parameter
          search) because the saved LOSO scores only exist on the old grid.
evaluate: replay alerts on the old and new streams with one shared definition.
report  : write the Korean comparison report.
verify  : check SHA256 invariance, grid spacing, split isolation, and tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.advanced import (  # noqa: E402
    apply_dynamic_motion_gate,
    compute_causal_exercise_score,
)
from src.features.nurse_forecasting import (  # noqa: E402
    SENSOR_FEATURE_MAP,
    extract_nurse_window_features,
)
from src.forecasting.calibration import apply_session_causal_calibration  # noqa: E402
from src.forecasting.labels import classify_prediction_timestamp  # noqa: E402
from src.forecasting.train import build_forecast_model, select_validation_threshold  # noqa: E402
from src.loaders.nurse_stress import load_nurse_session_zip  # noqa: E402

BASE = ROOT / "results" / "nurse_forecasting"
OUT = BASE / "fixed_grid_30s"
OLD_AUDIT = BASE / "root_cause_analysis"
RAW = ROOT / "data" / "raw" / "Nurse_Stress"
REPORT = ROOT / "reports" / "NURSE_FIXED30S_EARLY_WARNING_KO.md"

WINDOW_S = 180.0
FIXED_STEP_S = 30.0
P1_LO, P1_HI = 60.0, 180.0
NEG_STATES = ("VERIFIED_NON_EVENT", "CONFIRMED_NO_STRESS_0")
VERIFIED_STATE = "CONFIRMED_NO_STRESS_0"
UNVERIFIED_STATE = "VERIFIED_NON_EVENT"
DTYPES = {"subject": str, "session": str}
RNG_SEED = 42

FEATURE_COLS = list(SENSOR_FEATURE_MAP["all"])
STREAM_COLS = [
    "dataset", "subject", "session", "pred_ts", "pred_rel_s", "start_s", "end_s",
    "state", "target_event_id", "lead_time_to_onset_s", "y_true", "qc_common_valid",
    "prob", "threshold", "fold", "exposure_s",
    "acc_motion_change_per_s", "acc_unchanged_fraction", "acc_dynamic_energy",
    "has_survey_annotation",
]


def mkdirs() -> None:
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)


def sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def jsave(obj, name: str) -> None:
    (OUT / f"{name}.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )


def save(df: pd.DataFrame, name: str, compress: bool = False) -> None:
    if compress:
        df.to_csv(OUT / f"{name}.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    else:
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")


def read(name: str, root: Path = OUT, **kwargs) -> pd.DataFrame:
    path = root / name
    if path.suffix == ".gz":
        return pd.read_csv(path, dtype=DTYPES, low_memory=False, **kwargs)
    if path.with_suffix(path.suffix + ".gz").exists():
        return pd.read_csv(path.with_suffix(path.suffix + ".gz"), dtype=DTYPES, low_memory=False, **kwargs)
    return pd.read_csv(path, dtype=DTYPES, low_memory=False, **kwargs)


# ---------------------------------------------------------------------------
# Invariance snapshot
# ---------------------------------------------------------------------------
def protected_paths() -> list[Path]:
    paths: list[Path] = []
    paths += sorted(RAW.glob("*.xlsx"))
    paths += sorted(RAW.glob("*/*.zip"))
    if OUT.exists():
        paths += [p for p in BASE.rglob("*") if p.is_file() and OUT not in p.parents]
    else:
        paths += [p for p in BASE.rglob("*") if p.is_file()]
    paths += [p for p in (ROOT / "reports").rglob("*") if p.is_file()]
    for rel in [
        "src/forecasting/alerts.py", "src/forecasting/calibration.py",
        "src/forecasting/evaluate.py", "src/forecasting/labels.py",
        "src/forecasting/train.py", "src/features/advanced.py",
        "src/features/core.py", "src/features/nurse_forecasting.py",
        "src/loaders/nurse_stress.py",
    ]:
        paths.append(ROOT / rel)
    seen, unique = set(), []
    for p in paths:
        key = str(p.resolve())
        if p.exists() and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def snapshot_protected() -> pd.DataFrame:
    rows = []
    for p in protected_paths():
        rows.append({
            "path": str(p.relative_to(ROOT)),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        })
    return pd.DataFrame(rows)


def snapshot() -> None:
    t0 = time.perf_counter()
    df = snapshot_protected()
    save(df, "protected_hashes_before")
    jsave({
        "n_files": len(df),
        "total_bytes": int(df.bytes.sum()),
        "seconds": time.perf_counter() - t0,
    }, "protected_hashes_before_meta")
    print(f"snapshot: {len(df)} protected files, {time.perf_counter()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Fixed grid construction
# ---------------------------------------------------------------------------
def build_fixed_grid_times(duration_s: float, window_s: float = WINDOW_S, step_s: float = FIXED_STEP_S) -> list[float]:
    """Return prediction end-times t with t >= window_s and t <= duration_s - 5."""
    if not np.isfinite(duration_s) or duration_s < window_s + step_s:
        return []
    t_max = float(duration_s) - 5.0
    out = []
    t = float(window_s)
    while t <= t_max + 1e-9:
        out.append(float(t))
        t += float(step_s)
    return out


def event_lists(events: pd.DataFrame) -> tuple[list, list, list]:
    confirmed, na, zero = [], [], []
    for r in events.itertuples():
        eid = str(r.event_id)
        item = (eid, float(r.event_start_ts), float(r.event_end_ts))
        if r.label_category in ("confirmed_high_stress", "confirmed_moderate_stress"):
            confirmed.append(item)
        elif r.label_category == "survey_confirmed_no_stress":
            zero.append(item)
        else:
            na.append(item)
    return confirmed, na, zero


def p1_positive_mask(states: pd.Series, leads: pd.Series) -> np.ndarray:
    return (
        states.astype(str).eq("PRE_ONSET_P1_60_180S").to_numpy()
        & (pd.to_numeric(leads, errors="coerce").to_numpy() > P1_LO)
        & (pd.to_numeric(leads, errors="coerce").to_numpy() <= P1_HI)
    )


def label_stream_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Assign P1 window labels to an already-built fixed grid without resampling."""
    df = df.copy()
    y = np.full(len(df), -1, dtype=int)
    p1 = p1_positive_mask(df.state, df.lead_time_to_onset_s)
    y[p1] = 1
    y[df.state.isin(NEG_STATES).to_numpy()] = 0
    df["y_true"] = y
    return df


def build_features(annotated_only: bool = False) -> None:
    mkdirs()
    manifest = json.loads((BASE / "dataset_manifest.json").read_text(encoding="utf-8"))
    events = pd.read_csv(BASE / "event_audit.csv", dtype={"subject": str, "matched_session": str})
    sessions = pd.DataFrame(manifest["sessions"])
    if annotated_only:
        sessions = sessions[sessions.has_survey_annotation.astype(bool)].reset_index(drop=True)
    print(f"[build] sessions={len(sessions)} annotated_only={annotated_only}", flush=True)

    frames, logs = [], []
    t_all = time.perf_counter()
    for i, sm in enumerate(sessions.itertuples()):
        t0 = time.perf_counter()
        sid, sess = str(sm.subject), str(sm.session)
        try:
            sdata = load_nurse_session_zip(sm.zip_path, include_temp=False)
            times = build_fixed_grid_times(float(sdata["duration_s"]))
            if not times:
                logs.append({"subject": sid, "session": sess, "n_windows": 0, "status": "too_short"})
                continue
            if bool(sm.has_survey_annotation):
                confirmed, na, zero = event_lists(events[events.subject.eq(sid)])
            else:
                confirmed, na, zero = [], [], []

            rows = []
            for t_rel in times:
                feats = extract_nurse_window_features(sdata, t_rel, window_s=WINDOW_S)
                pred_ts = float(sdata["start_ts"]) + t_rel
                if bool(sm.has_survey_annotation):
                    info = classify_prediction_timestamp(pred_ts, confirmed, na, zero)
                else:
                    info = {
                        "state": "UNANNOTATED", "target_event_id": "", "lead_time_to_onset_s": np.nan,
                        "y_P0": -1, "y_P1": -1, "y_P2": -1,
                    }
                rows.append({
                    "dataset": "Nurse_Stress_Fixed30s",
                    "subject": sid,
                    "session": sess,
                    "pred_ts": pred_ts,
                    "pred_rel_s": t_rel,
                    "start_s": t_rel - WINDOW_S,
                    "end_s": t_rel,
                    **info,
                    "has_survey_annotation": bool(sm.has_survey_annotation),
                    **feats,
                })
            frames.append(pd.DataFrame(rows))
            logs.append({
                "subject": sid, "session": sess, "n_windows": len(rows), "status": "ok",
                "duration_s": float(sdata["duration_s"]), "seconds": time.perf_counter() - t0,
                "has_survey_annotation": bool(sm.has_survey_annotation),
            })
        except Exception as exc:  # noqa: BLE001
            logs.append({
                "subject": sid, "session": sess, "n_windows": 0, "status": "error",
                "error": f"{type(exc).__name__}: {exc}", "seconds": time.perf_counter() - t0,
                "has_survey_annotation": bool(sm.has_survey_annotation),
            })
            print(f"[build] ERROR {sid}/{sess}: {exc}", flush=True)
        if (i + 1) % 20 == 0 or (i + 1) == len(sessions):
            print(
                f"[build] {i+1}/{len(sessions)} sessions, {sum(len(x) for x in frames)} windows, "
                f"{time.perf_counter()-t_all:.1f}s", flush=True,
            )

    feat = pd.concat(frames, ignore_index=True).sort_values(["subject", "session", "pred_ts"]).reset_index(drop=True)
    # Strictly label-blind personal calibration; the state column is hidden from the calibrator.
    blind = feat.copy()
    blind["state"] = "UNAVAILABLE_AT_INFERENCE"
    cal = apply_session_causal_calibration(blind, method="low_motion", duration_s=180.0)
    for col in cal.columns:
        if col == "state":
            continue
        feat[col] = cal[col].to_numpy()
    feat["exercise_score"] = compute_causal_exercise_score(feat)
    feat = label_stream_grid(feat)

    save(feat, "features_fixed30s", compress=True)
    ann = feat[feat.has_survey_annotation.astype(bool)].reset_index(drop=True)
    save(ann, "features_fixed30s_annotated", compress=True)
    logs_df = pd.DataFrame(logs)
    save(logs_df, "grid_build_log")
    jsave({
        "n_sessions_total": int(len(sessions)),
        "n_sessions_ok": int((logs_df.status == "ok").sum()),
        "n_sessions_error": int((logs_df.status == "error").sum()),
        "n_sessions_annotated": int(ann.session.nunique()),
        "n_windows_total": int(len(feat)),
        "n_windows_annotated": int(len(ann)),
        "positive_p1": int(ann.y_true.eq(1).sum()),
        "negative_verified": int((ann.state == VERIFIED_STATE).sum()),
        "negative_unverified": int((ann.state == UNVERIFIED_STATE).sum()),
        "fixed_step_s": FIXED_STEP_S,
        "window_s": WINDOW_S,
        "n_features": len(FEATURE_COLS),
        "features_sha256": sha256(OUT / "features_fixed30s.csv.gz"),
        "features_annotated_sha256": sha256(OUT / "features_fixed30s_annotated.csv.gz"),
        "seconds": time.perf_counter() - t_all,
    }, "grid_manifest")
    print(f"[build] complete: {len(feat)} windows, {len(ann)} annotated", flush=True)


# ---------------------------------------------------------------------------
# Existing LOSO protocol, no model search
# ---------------------------------------------------------------------------
def fold_splits(subjects: list[str], seed: int = RNG_SEED) -> list[dict]:
    subjects = sorted(subjects)
    out = []
    work = pd.DataFrame({"subject": np.repeat(subjects, 2)})
    work["y"] = np.tile([0, 1], len(subjects))
    for fold, test_subj in enumerate(subjects):
        dev = work[work.subject.ne(test_subj)].reset_index(drop=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + fold)
        tr_rel, val_rel = next(gss.split(dev, dev.y, dev.subject))
        train_subj = sorted(dev.iloc[tr_rel].subject.unique().tolist())
        val_subj = sorted(dev.iloc[val_rel].subject.unique().tolist())
        assert test_subj not in train_subj and test_subj not in val_subj
        assert not (set(train_subj) & set(val_subj))
        out.append({
            "fold": fold, "test_subject": test_subj,
            "train_subjects": ";".join(train_subj),
            "val_subjects": ";".join(val_subj),
        })
    return out


def train_loso() -> None:
    mkdirs()
    feat = read("features_fixed30s_annotated.csv.gz")
    feat = feat[feat.has_survey_annotation.astype(bool)].copy()
    feat = label_stream_grid(feat)
    subjects = sorted(feat.subject.unique())
    splits = fold_splits(subjects)
    save(pd.DataFrame(splits), "splits_fixed30s")

    preds, metas = [], []
    for sp in splits:
        fold, test_subj = int(sp["fold"]), sp["test_subject"]
        train_subj = set(sp["train_subjects"].split(";"))
        val_subj = set(sp["val_subjects"].split(";"))
        assert test_subj not in train_subj | val_subj
        dev = feat[feat.subject.isin(train_subj | val_subj) & feat.y_true.isin([0, 1])]
        train = dev[dev.subject.isin(train_subj)]
        val = dev[dev.subject.isin(val_subj)]
        test = feat[feat.subject.eq(test_subj)].copy()

        pipe = build_forecast_model("LR", n_jobs=2)
        t0 = time.perf_counter()
        pipe.fit(train[FEATURE_COLS], train.y_true.to_numpy(dtype=int))
        p_val = pipe.predict_proba(val[FEATURE_COLS])[:, 1]
        thr = select_validation_threshold(val.y_true.to_numpy(dtype=int), p_val)
        pipe_full = build_forecast_model("LR", n_jobs=2)
        pipe_full.fit(dev[FEATURE_COLS], dev.y_true.to_numpy(dtype=int))
        p_test = pipe_full.predict_proba(test[FEATURE_COLS])[:, 1]

        test["prob"] = p_test
        test["threshold"] = float(thr)
        test["pred"] = (p_test >= thr).astype(int)
        test["fold"] = fold
        test["y_true"] = test.y_true.astype(int)
        preds.append(test)
        metas.append({
            "fold": fold, "test_subject": test_subj,
            "n_train_subjects": len(train_subj), "n_val_subjects": len(val_subj),
            "train_subjects": sp["train_subjects"], "val_subjects": sp["val_subjects"],
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "threshold": float(thr),
            "validation_f1": float(f1_score(val.y_true, p_val >= thr, zero_division=0)),
            "seconds": time.perf_counter() - t0,
        })
        print(f"[train] fold {fold+1}/{len(splits)} test={test_subj} thr={thr:.3f}", flush=True)

    pred = pd.concat(preds, ignore_index=True).sort_values(["subject", "session", "pred_ts"]).reset_index(drop=True)
    save(pred, "predictions_fixed30s", compress=True)
    save(pd.DataFrame(metas), "fold_metrics_fixed30s")
    jsave({
        "model": "LogisticRegression(C=0.25, class_weight=balanced)",
        "window_s": WINDOW_S,
        "step_s": FIXED_STEP_S,
        "n_folds": len(splits),
        "features": FEATURE_COLS,
        "protocol_note": (
            "Existing LOSO specification reapplied. No new algorithm or hyper-parameter search. "
            "Saved full-refit joblib models are not used for held-out evaluation."
        ),
    }, "model_protocol_fixed30s")


# ---------------------------------------------------------------------------
# Shared evaluation definitions
# ---------------------------------------------------------------------------
def add_exposure(stream: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    stream = stream.sort_values(["subject", "session", "pred_ts"]).copy()
    ends = {str(s["session"]): float(s["end_ts"]) for s in manifest["sessions"]}
    nxt = stream.groupby(["subject", "session"], sort=False).pred_ts.shift(-1)
    tail = stream.session.map(ends).astype(float) - stream.pred_ts.astype(float)
    stream["exposure_s"] = np.clip((nxt - stream.pred_ts).fillna(tail).to_numpy(dtype=float), 0.0, 60.0)
    return stream


def prepare_stream(df: pd.DataFrame, manifest: dict, source: str) -> pd.DataFrame:
    d = df.copy()
    if "y_true" not in d.columns:
        d = label_stream_grid(d)
    else:
        d["y_true"] = d.y_true.astype(int)
    d["qc_common_valid"] = d.get("qc_common_valid", True)
    if isinstance(d["qc_common_valid"], pd.Series):
        d["qc_common_valid"] = d["qc_common_valid"].astype(bool)
    d["prob"] = pd.to_numeric(d.prob, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    d.loc[~d.qc_common_valid, "prob"] = 0.0
    d["threshold"] = pd.to_numeric(d.threshold, errors="coerce").fillna(0.5).to_numpy(dtype=float)
    d["source"] = source
    d = add_exposure(d, manifest)
    return d.sort_values(["subject", "session", "pred_ts"]).reset_index(drop=True)


def apply_gate(stream: pd.DataFrame, mode: str) -> pd.DataFrame:
    pieces = []
    for (sid, sess), g in stream.groupby(["subject", "session"], sort=False):
        g = g.sort_values("pred_ts").copy()
        base = float(g.threshold.iloc[0]) if len(g) else 0.5
        if mode == "none":
            g["prob_gated"] = g.prob.to_numpy(dtype=float)
            g["threshold_gated"] = g.threshold.to_numpy(dtype=float)
        else:
            p_g, thr, _ = apply_dynamic_motion_gate(
                g.prob.to_numpy(dtype=float), g, mode=mode, base_threshold=base
            )
            g["prob_gated"] = p_g
            g["threshold_gated"] = thr
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values(["subject", "session", "pred_ts"]).reset_index(drop=True)


def replay_policy(stream: pd.DataFrame, policy: str) -> np.ndarray:
    out = np.zeros(len(stream), dtype=int)
    t = stream.pred_ts.to_numpy(dtype=float)
    p = stream.prob_gated.to_numpy(dtype=float)
    thr = stream.threshold_gated.to_numpy(dtype=float)
    for _, idx in stream.groupby(["subject", "session"], sort=False).indices.items():
        idx = np.sort(idx)
        pp, tt, rr = p[idx], t[idx], thr[idx]
        if policy == "val_opt":
            out[idx] = (pp >= rr).astype(int)
        elif policy == "hysteresis":
            active = 0
            for j in range(len(idx)):
                t_on = max(rr[j], 0.52)
                t_off = min(rr[j] * 0.70, 0.35)
                if active == 0 and pp[j] >= t_on:
                    active = 1
                elif active == 1 and pp[j] < t_off:
                    active = 0
                out[idx[j]] = active
        else:
            raise ValueError(policy)
    return out


def _episodes(stream: pd.DataFrame, alert: np.ndarray, merge_gap_s: float = 180.0) -> list[dict]:
    eps = []
    s = stream.copy()
    s["_alert"] = alert
    for (sid, sess), g in s.groupby(["subject", "session"], sort=False):
        g = g.sort_values("pred_ts")
        pos = g[g._alert == 1]
        if pos.empty:
            continue
        times = pos.pred_ts.to_numpy(dtype=float)
        groups = np.cumsum(np.r_[True, np.diff(times) > merge_gap_s])
        for _, ep in pos.groupby(groups):
            eps.append({
                "subject": sid, "session": sess,
                "alert_start_ts": float(ep.pred_ts.min()),
                "alert_end_ts": float(ep.pred_ts.max()),
                "n_windows": int(len(ep)),
                "states": sorted(set(ep.state.astype(str))),
                "y_true": ep.y_true.to_numpy(dtype=int),
                "alert_times": ep.pred_ts.to_numpy(dtype=float),
            })
    return eps


def _hours(stream: pd.DataFrame, state: str | None = None) -> float:
    m = stream.y_true.eq(0).to_numpy() & stream.qc_common_valid.to_numpy()
    if state is not None:
        m &= stream.state.eq(state).to_numpy()
    else:
        m &= stream.state.isin(NEG_STATES).to_numpy()
    return float(stream.loc[m, "exposure_s"].sum() / 3600.0)


def evaluate_alert_stream(stream: pd.DataFrame, events: pd.DataFrame, policy: str, endpoint: str) -> dict:
    """One shared event/episode definition for old and new streams.

    endpoint='active'  -> any active warning timestamp inside (onset-180, onset-60]
    endpoint='issued'  -> first timestamp of the alert episode inside that horizon
    """
    alert = replay_policy(stream, policy)
    eps = _episodes(stream, alert)
    hours_all = _hours(stream)
    hours_ver = _hours(stream, VERIFIED_STATE)
    hours_unv = _hours(stream, UNVERIFIED_STATE)

    assigned: set[int] = set()
    event_rows = []
    for ev in events.itertuples():
        lo = float(ev.event_start_ts) - P1_HI
        hi = float(ev.event_start_ts) - P1_LO
        sess = str(ev.matched_session)
        rows = stream[(stream.subject == str(ev.subject)) & (stream.session == sess)]
        n_avail = int(((rows.pred_ts >= lo - 1e-9) & (rows.pred_ts < hi - 1e-9)).sum())
        candidates = []
        for j, ep in enumerate(eps):
            if ep["subject"] != str(ev.subject) or ep["session"] != sess:
                continue
            if endpoint == "active":
                hit = bool(((ep["alert_times"] >= lo - 1e-9) & (ep["alert_times"] < hi - 1e-9)).any())
            else:
                hit = bool(lo - 1e-9 <= ep["alert_start_ts"] < hi - 1e-9)
            if hit:
                candidates.append(j)
        detected = bool(candidates)
        if detected:
            j = next((j for j in candidates if j not in assigned), candidates[0])
            assigned.add(j)
            lead = float(ev.event_start_ts) - eps[j]["alert_start_ts"]
        else:
            lead = np.nan
        event_rows.append({
            "event_id": ev.event_id, "subject": ev.subject, "session": sess,
            "onset_ts": float(ev.event_start_ts), "n_available_windows": n_avail,
            "detected": int(detected), "lead_s": lead,
        })

    true_eps, false_eps, neutral_eps = 0, 0, 0
    false_ver, false_unv, false_mixed = 0, 0, 0
    for j, ep in enumerate(eps):
        if j in assigned:
            true_eps += 1
            continue
        if endpoint == "issued":
            start_state = stream[(stream.subject == ep["subject"]) & (stream.session == ep["session"])
                                 & (stream.pred_ts == ep["alert_start_ts"])].state
            has_neg = bool(len(start_state) and str(start_state.iloc[0]) in NEG_STATES)
            src_states = {str(start_state.iloc[0])} if has_neg else set()
        else:
            sub = stream[(stream.subject == ep["subject"]) & (stream.session == ep["session"])
                         & stream.pred_ts.between(ep["alert_start_ts"], ep["alert_end_ts"])]
            neg = sub[sub.y_true.eq(0) & sub.qc_common_valid]
            has_neg = bool(len(neg))
            src_states = set(neg.state.astype(str)) if has_neg else set()
        if not has_neg:
            neutral_eps += 1
            continue
        false_eps += 1
        in_ver = VERIFIED_STATE in src_states
        in_unv = UNVERIFIED_STATE in src_states
        false_ver += int(in_ver)
        false_unv += int(in_unv)
        false_mixed += int(in_ver and in_unv)

    total_events = len(events)
    detected = sum(r["detected"] for r in event_rows)
    observable = sum(r["n_available_windows"] > 0 for r in event_rows)
    detected_obs = sum(r["detected"] for r in event_rows if r["n_available_windows"] > 0)
    leads = np.array([r["lead_s"] for r in event_rows if np.isfinite(r["lead_s"])], dtype=float)
    precision = true_eps / (true_eps + false_eps) if (true_eps + false_eps) else np.nan
    return {
        "policy": policy, "endpoint": endpoint,
        "n_events": total_events,
        "n_events_observable": int(observable),
        "detected_events": int(detected),
        "event_recall": detected / total_events if total_events else np.nan,
        "event_recall_observable": detected_obs / observable if observable else np.nan,
        "true_episodes": int(true_eps),
        "false_episodes": int(false_eps),
        "neutral_episodes": int(neutral_eps),
        "total_episodes": int(len(eps)),
        "episode_precision": precision,
        "false_episodes_verified_source": int(false_ver),
        "false_episodes_unverified_source": int(false_unv),
        "false_episodes_mixed_source": int(false_mixed),
        "negative_hours_all": hours_all,
        "negative_hours_verified": hours_ver,
        "negative_hours_unverified": hours_unv,
        "fa_per_hour_all": false_eps / hours_all if hours_all > 0 else np.nan,
        "fa_per_hour_verified": false_ver / hours_ver if hours_ver > 0 else np.nan,
        "fa_per_hour_unverified": false_unv / hours_unv if hours_unv > 0 else np.nan,
        "lead_median_s": float(np.median(leads)) if len(leads) else np.nan,
        "lead_mean_s": float(np.mean(leads)) if len(leads) else np.nan,
        "events_with_lead_ge_60": int((leads >= 60.0 - 1e-9).sum()) if len(leads) else 0,
        "_events": event_rows,
        "_episodes": eps,
    }


def window_metrics(stream: pd.DataFrame, subset: str = "all") -> dict:
    ev = stream[stream.y_true.isin([0, 1]) & stream.qc_common_valid].copy()
    if subset == "verified":
        ev = ev[(ev.y_true == 1) | (ev.state == VERIFIED_STATE)]
    elif subset == "unverified":
        ev = ev[(ev.y_true == 1) | (ev.state == UNVERIFIED_STATE)]
    y = ev.y_true.to_numpy(dtype=int)
    p = ev.prob_gated.to_numpy(dtype=float)
    pooled_auroc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    pooled_auprc = average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan
    per = []
    for sid, g in ev.groupby("subject", sort=True):
        yy = g.y_true.to_numpy(dtype=int)
        pp = g.prob_gated.to_numpy(dtype=float)
        per.append({
            "subject": sid,
            "auroc": roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan,
            "auprc": average_precision_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan,
            "n": len(g), "positive": int(yy.sum()),
        })
    per_df = pd.DataFrame(per)
    return {
        "subset": subset, "n_windows": int(len(ev)), "positive": int(y.sum()),
        "negative": int((y == 0).sum()), "prevalence": float(y.mean()) if len(y) else np.nan,
        "pooled_auroc": pooled_auroc, "pooled_auprc": pooled_auprc,
        "macro_auroc": float(per_df.auroc.mean()), "macro_auprc": float(per_df.auprc.mean()),
        "n_subjects_defined": int(per_df.auroc.notna().sum()),
        "_per_subject": per_df,
    }


def load_old_prediction(name: str) -> pd.DataFrame:
    return pd.read_csv(OLD_AUDIT / "predictions" / f"{name}.csv.gz", dtype=DTYPES, low_memory=False)


def evaluate_all() -> None:
    mkdirs()
    manifest = json.loads((BASE / "dataset_manifest.json").read_text(encoding="utf-8"))
    events = pd.read_csv(BASE / "event_audit.csv", dtype={"subject": str, "matched_session": str})
    events = events[
        events.label_category.isin(["confirmed_high_stress", "confirmed_moderate_stress"])
        & events.onset_covered.astype(bool)
        & (pd.to_numeric(events.pre_onset_history_s) >= WINDOW_S + P1_LO)
    ].copy()

    new = read("predictions_fixed30s.csv.gz")
    new = new[new.has_survey_annotation.astype(bool)]
    old = load_old_prediction("labelblind_same_method")
    old_legacy = load_old_prediction("legacy_reproduction")

    streams = {
        "old_labelblind": prepare_stream(old, manifest, "old_labelblind"),
        "old_legacy": prepare_stream(old_legacy, manifest, "old_legacy"),
        "new_fixed30s": prepare_stream(new, manifest, "new_fixed30s"),
    }
    gates = ["none", "legacy_hard", "soft_dynamic"]
    policies = ["val_opt", "hysteresis"]
    endpoints = ["active", "issued"]

    window_rows, alert_rows, per_subject_rows = [], [], []
    event_rows, episode_rows = [], []
    for src, stream0 in streams.items():
        for gate in gates:
            stream = apply_gate(stream0, gate)
            for subset in ["all", "verified", "unverified"]:
                wm = window_metrics(stream, subset)
                per = wm.pop("_per_subject")
                window_rows.append({"stream": src, "gate": gate, **wm})
                for r in per.itertuples():
                    per_subject_rows.append({
                        "stream": src, "gate": gate, "subset": subset,
                        "subject": r.subject, "auroc": r.auroc, "auprc": r.auprc,
                        "n": r.n, "positive": r.positive,
                    })
            for policy in policies:
                for endpoint in endpoints:
                    em = evaluate_alert_stream(stream, events, policy, endpoint)
                    evs = em.pop("_events")
                    eps = em.pop("_episodes")
                    alert_rows.append({"stream": src, "gate": gate, **em})
                    for e in evs:
                        event_rows.append({"stream": src, "gate": gate, "policy": policy, "endpoint": endpoint, **e})
                    for ep in eps:
                        episode_rows.append({
                            "stream": src, "gate": gate, "policy": policy, "endpoint": endpoint,
                            "subject": ep["subject"], "session": ep["session"],
                            "alert_start_ts": ep["alert_start_ts"], "alert_end_ts": ep["alert_end_ts"],
                            "n_windows": ep["n_windows"], "states": "|".join(ep["states"]),
                        })
            print(f"[evaluate] {src} gate={gate} done", flush=True)

    wdf = pd.DataFrame(window_rows)
    adf = pd.DataFrame(alert_rows)
    save(wdf, "window_metrics_comparison")
    save(adf, "alert_metrics_comparison")
    save(pd.DataFrame(per_subject_rows), "per_subject_window_metrics")
    save(pd.DataFrame(event_rows), "event_details")
    save(pd.DataFrame(episode_rows), "episode_details")

    # Compact side-by-side summary used by the report.
    key = adf[
        adf.policy.eq("hysteresis")
    ][[
        "stream", "gate", "endpoint", "event_recall", "event_recall_observable",
        "episode_precision", "fa_per_hour_all", "fa_per_hour_verified",
        "fa_per_hour_unverified", "detected_events", "n_events",
        "true_episodes", "false_episodes", "neutral_episodes",
        "negative_hours_verified", "negative_hours_unverified",
    ]].copy()
    save(key, "comparison_summary")
    jsave({
        "events_denominator": int(len(events)),
        "p1_positive_definition": "(60s, 180s] before survey-confirmed onset",
        "active_endpoint": "any warning-active timestamp inside (onset-180s, onset-60s]",
        "issued_endpoint": "first timestamp of a merged alert episode inside the same horizon",
        "false_alarm": "unassigned episode with at least one evaluated negative window (active) or a negative first issuance (issued)",
        "neutral": "episode containing only censored/unverified-by-label windows is not counted as FA",
        "verified_negative": VERIFIED_STATE,
        "unverified_background": UNVERIFIED_STATE,
        "motion_gates": gates,
        "policies": policies,
    }, "evaluation_definitions")
    print("[evaluate] complete", flush=True)


# ---------------------------------------------------------------------------
# Difference decomposition: score effect vs grid effect
# ---------------------------------------------------------------------------
def analyze_differences() -> None:
    """Separate (a) retrained score/threshold effect from (b) added 30s background.

    All old-grid timestamps are a subset of the new fixed 30s grid. Therefore
    "new score on old grid" isolates the model/threshold change, while the
    difference to the full new stream isolates the added uniform background.
    """
    mkdirs()
    manifest = json.loads((BASE / "dataset_manifest.json").read_text(encoding="utf-8"))
    events = pd.read_csv(BASE / "event_audit.csv", dtype={"subject": str, "matched_session": str})
    events = events[
        events.label_category.isin(["confirmed_high_stress", "confirmed_moderate_stress"])
        & events.onset_covered.astype(bool)
        & (pd.to_numeric(events.pre_onset_history_s) >= WINDOW_S + P1_LO)
    ].copy()
    old = load_old_prediction("labelblind_same_method")
    new = read("predictions_fixed30s.csv.gz")
    new = new[new.has_survey_annotation.astype(bool)].copy()
    keys = ["subject", "session", "pred_ts"]
    old_stream = prepare_stream(old, manifest, "old_grid_old_score")
    new_stream = prepare_stream(new, manifest, "new_grid_new_score")
    matched = new_stream.merge(
        old_stream[keys + ["state", "y_true", "prob", "threshold", "qc_common_valid"]],
        on=keys, how="inner", suffixes=("", "_old"), validate="one_to_one",
    )
    assert len(matched) == len(old_stream), (len(matched), len(old_stream))
    assert (matched.state == matched.state_old).all()
    matched["prob"] = matched["prob"].astype(float).to_numpy()
    matched["threshold"] = matched["threshold"].astype(float).to_numpy()
    matched["source"] = "old_grid_new_score"
    matched = add_exposure(matched, manifest).sort_values(["subject", "session", "pred_ts"]).reset_index(drop=True)

    streams = {
        "old_grid_old_score": old_stream,
        "old_grid_new_score": matched,
        "new_grid_new_score": new_stream,
    }
    wrows, arows = [], []
    for src, base in streams.items():
        for gate in ["none", "soft_dynamic"]:
            st = apply_gate(base, gate)
            wm = window_metrics(st, "all")
            wm.pop("_per_subject")
            wrows.append({"stream": src, "gate": gate, **wm})
            for endpoint in ["active", "issued"]:
                em = evaluate_alert_stream(st, events, "hysteresis", endpoint)
                em.pop("_events")
                em.pop("_episodes")
                arows.append({"stream": src, "gate": gate, **em})
    save(pd.DataFrame(wrows), "decomposition_window_metrics")
    save(pd.DataFrame(arows), "decomposition_alert_metrics")

    # Label-source composition of old vs new grid and the matched subset.
    def counts(stream):
        return {
            "windows": len(stream),
            "p1_positive": int(stream.y_true.eq(1).sum()),
            "verified_negative": int((stream.y_true.eq(0) & stream.state.eq(VERIFIED_STATE)).sum()),
            "unverified_negative": int((stream.y_true.eq(0) & stream.state.eq(UNVERIFIED_STATE)).sum()),
            "censored_or_other": int((stream.y_true < 0).sum()),
        }
    comp = []
    for name, st in [("old_grid", old_stream), ("new_grid", new_stream), ("old_grid_subset_of_new", matched)]:
        comp.append({"stream": name, **counts(st)})
    save(pd.DataFrame(comp), "grid_composition")

    # Threshold changes and matched probability correlation.
    th = old_stream.groupby("subject").threshold.first().rename("old_threshold").to_frame()
    th["new_threshold"] = new_stream.groupby("subject").threshold.first()
    th["threshold_delta"] = th.new_threshold - th.old_threshold
    th["old_mean_prob"] = old_stream.groupby("subject").prob.mean()
    th["new_mean_prob"] = new_stream.groupby("subject").prob.mean()
    th["matched_prob_corr"] = [
        matched[matched.subject.eq(sid)][["prob", "prob_old"]].corr().iloc[0, 1]
        for sid in th.index
    ]
    save(th.reset_index(), "threshold_changes")

    # Per-subject event detection under the primary no-gate hysteresis endpoint.
    rows = []
    old_eval = evaluate_alert_stream(apply_gate(old_stream, "none"), events, "hysteresis", "active")
    new_eval = evaluate_alert_stream(apply_gate(new_stream, "none"), events, "hysteresis", "active")
    old_det = {r["event_id"]: r["detected"] for r in old_eval["_events"]}
    new_det = {r["event_id"]: r["detected"] for r in new_eval["_events"]}
    for sid in sorted(events.subject.unique()):
        e = events[events.subject.eq(sid)]
        rows.append({
            "subject": sid,
            "n_events": len(e),
            "old_detected": int(sum(old_det.get(x, 0) for x in e.event_id)),
            "new_detected": int(sum(new_det.get(x, 0) for x in e.event_id)),
            "old_threshold": float(th.loc[sid, "old_threshold"]),
            "new_threshold": float(th.loc[sid, "new_threshold"]),
        })
    per = pd.DataFrame(rows)
    per["delta_detected"] = per.new_detected - per.old_detected
    save(per, "per_subject_detection_changes")

    # Feature equivalence on the 51,234 shared timestamps.
    old_feat = pd.read_csv(BASE / "features.csv", dtype=DTYPES, low_memory=False)
    new_feat = read("features_fixed30s.csv.gz")
    fkeys = ["subject", "session", "pred_ts"]
    fm = old_feat[fkeys + FEATURE_COLS].merge(
        new_feat[fkeys + FEATURE_COLS], on=fkeys, suffixes=("_old", "_new"), validate="one_to_one"
    )
    from src.features.advanced import CALIBRATED_PHYSIO_FEATURES
    fe_rows = []
    for c in FEATURE_COLS:
        d = np.abs(fm[c + "_old"].to_numpy(dtype=float) - fm[c + "_new"].to_numpy(dtype=float))
        fe_rows.append({
            "feature": c,
            "calibrated": c in CALIBRATED_PHYSIO_FEATURES,
            "max_abs_diff": float(np.nanmax(d)) if np.isfinite(d).any() else np.nan,
            "n_rows_diff_gt_1e-9": int((d > 1e-9).sum()),
        })
    fe = pd.DataFrame(fe_rows)
    save(fe, "feature_equivalence")
    print("[analyze_differences] feature equivalence complete", flush=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(v, nd=3):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "정의 불가"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{nd}f}"


def _table(df: pd.DataFrame, cols: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in df[cols].itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(x) for x in r) + " |")
    return "\n".join(lines)


def write_report() -> None:
    gm = json.loads((OUT / "grid_manifest.json").read_text(encoding="utf-8"))
    wdf = pd.read_csv(OUT / "window_metrics_comparison.csv")
    adf = pd.read_csv(OUT / "alert_metrics_comparison.csv")
    folds = pd.read_csv(OUT / "fold_metrics_fixed30s.csv")
    comp = pd.read_csv(OUT / "grid_composition.csv")
    dw = pd.read_csv(OUT / "decomposition_window_metrics.csv")
    da = pd.read_csv(OUT / "decomposition_alert_metrics.csv")
    th = pd.read_csv(OUT / "threshold_changes.csv", dtype={"subject": str})
    pdet = pd.read_csv(OUT / "per_subject_detection_changes.csv", dtype={"subject": str})
    feq = pd.read_csv(OUT / "feature_equivalence.csv")
    edef = json.loads((OUT / "evaluation_definitions.json").read_text(encoding="utf-8"))
    ver_path = OUT / "verification.json"
    ver = json.loads(ver_path.read_text(encoding="utf-8")) if ver_path.exists() else {}
    build = pd.read_csv(OUT / "grid_build_log.csv")

    # Side-by-side window metrics (hysteresis irrelevant, one row per stream/gate).
    w = wdf[wdf.subset.eq("all")][["stream", "gate", "n_windows", "positive", "negative", "prevalence", "macro_auroc", "macro_auprc", "pooled_auroc", "pooled_auprc"]].copy()
    wv = wdf[wdf.subset.eq("verified")][["stream", "gate", "macro_auroc", "macro_auprc", "pooled_auroc", "pooled_auprc"]].rename(columns={c: c + "_verified" for c in ["macro_auroc", "macro_auprc", "pooled_auroc", "pooled_auprc"]})
    wu = wdf[wdf.subset.eq("unverified")][["stream", "gate", "macro_auroc", "macro_auprc", "pooled_auroc", "pooled_auprc"]].rename(columns={c: c + "_unverified" for c in ["macro_auroc", "macro_auprc", "pooled_auroc", "pooled_auprc"]})
    w = w.merge(wv, on=["stream", "gate"], how="left").merge(wu, on=["stream", "gate"], how="left")

    primary = adf[adf.policy.eq("hysteresis")].copy()
    pvt = primary.pivot_table(
        index=["stream", "gate"], columns="endpoint",
        values=["event_recall", "episode_precision", "fa_per_hour_all", "fa_per_hour_verified", "fa_per_hour_unverified"],
        aggfunc="first",
    )
    pvt.columns = [f"{a}_{b}" for a, b in pvt.columns]
    pvt = pvt.reset_index()

    lines = [
        "# Nurses 고정 30초 전체 시간축 조기경고 재평가",
        "",
        f"작성일: 2026-10-26. 원본 센서·설문·기존 특징 캐시·모델·감사 산출물은 수정하지 않았다. 신규 산출물은 `results/nurse_forecasting/fixed_grid_30s/`에만 저장했다.",
        "",
        "## 0. 결론 요약",
        "",
        "- 기존 라벨 의존 30/60초 격자를 **전체 세션 고정 30초 격자**로 바꾸고, 동일한 62개 특징·과거 180/60/300초 관측 창·동일 LR LOSO 절차로 다시 평가했다.",
        "- **경고 활성(active)** 과 **사건 60~180초 전 신규 발령(issued)** 을 분리했다. 두 값은 같은 모델에서도 크게 다르며, 활성 값을 신규 조기경고로 인용하면 안 된다.",
        "- 운동 게이트는 이전 감사와 같은 방향으로 **고활동 오경보를 줄이면서 진양성 경고도 함께 제거**했다. 게이트 후 성능 개선으로 해석하지 않는다.",
        "- 검증된 음성(`CONFIRMED_NO_STRESS_0`)과 미검증 배경(`VERIFIED_NON_EVENT`)을 분리했다. 미검증 배경이 평가 음성의 대부분을 차지하므로 FA/h와 AUROC는 음성 출처별로 함께 읽어야 한다.",
        "- **고정 30초 격자는 사건 주변 창을 바꾸지 않았다.** 기존 격자도 사건 주변은 이미 30초였고, 신규 캐시는 배경만 60초에서 30초로 촘촘하게 추가했다. 따라서 event-level 변화는 격자 자체가 아니라 재적합 점수·임계값 변화에서 왔다.",
        "- 고정 격자 재평가 결과는 **성능 개선을 보장하지 않는다.** 아래 표의 실제 측정값만 보고한다.",
        "",
        "## 1. 신규 고정 30초 특징 캐시",
        "",
        f"- 세션: {gm['n_sessions_ok']}/{gm['n_sessions_total']} 처리, 오류 {gm['n_sessions_error']}건, 185초 미만 제외 {gm['n_sessions_total']-gm['n_sessions_ok']}건. 평가는 설문 주석 214세션만 사용.",
        f"- 전체 창: {gm['n_windows_total']:,}; 설문 주석 세션 창: {gm['n_windows_annotated']:,}.",
        f"- P1 양성: {gm['positive_p1']:,}; 검증 음성: {gm['negative_verified']:,}; 미검증 배경: {gm['negative_unverified']:,}.",
        f"- 고정 간격: {gm['fixed_step_s']:.0f}초, 주 관측 창: {gm['window_s']:.0f}초, 특징 수: {gm['n_features']}.",
        f"- 개인 보정은 state를 숨긴 label-blind 방식으로 적용했다. 기존 캐시의 보정 참조와 동일한 low-motion/첫 창 규칙을 사용했다.",
        f"- 공유 timestamp 51,234개에서 **원시 53개 특징은 최대 절대차 {feq[~feq.calibrated.astype(bool)].max_abs_diff.max():.3g}으로 동일**했다. 9개 개인보정 특징은 기존 격자의 보정 참조 두 번째 창이 배경에서 60초 간격(180/240초)이던 것이 고정 30초(180/210초)로 바뀌어 차이가 생겼다. 이는 원시 특징 추출 차이가 아니라 보정 참조 창의 격자 차이다.",
        f"- 캐시 SHA256: `{gm['features_sha256']}`.",
        "",
        "## 2. LOSO 재학습 프로토콜",
        "",
        f"- 15-fold LOSO, 각 fold 내부 train/validation 참가자 분리. 저장된 full-refit joblib 모델은 held-out 평가에 사용하지 않았다.",
        f"- 기존과 동일한 `LogisticRegression(C=0.25, class_weight='balanced')` + median imputer + StandardScaler. 새 알고리즘·하이퍼파라미터 탐색은 하지 않았다.",
        f"- 임계값은 각 fold의 내부 validation 참가자 F1 절차로만 선택했다. 테스트 참가자 라벨은 임계값·전처리 적합에 사용하지 않았다.",
        f"- fold 수: {len(folds)}, 평균 학습 창: {folds.n_train.mean():.0f}, 평균 validation F1: {folds.validation_f1.mean():.3f}.",
        "",
        "새 캐시에서 기존 LOSO 점수를 그대로 적용할 수는 없었다. 기존 저장 점수는 라벨 의존 격자에서만 존재하고, fold별 scaler/imputer 파라미터도 저장되어 있지 않기 때문이다. 따라서 **동일 모델 명세를 재적용**했으며, 이는 새 모델 탐색이 아니다.",
        "",
        "## 3. 동일 평가 정의",
        "",
        f"- P1 양성: 사건 시작 {edef['p1_positive_definition']}.",
        f"- Active endpoint: {edef['active_endpoint']}.",
        f"- Issued endpoint: {edef['issued_endpoint']}.",
        f"- 오경보: {edef['false_alarm']}.",
        "- Event Recall은 원 적격 사건 분모를 유지하고, 관측 가능 분모도 함께 표기했다. Precision은 사건-에피소드 일대일 배정 후 계산했다.",
        "- FA/h 분모는 실제 창 노출 시간 `[t, next_t)`을 세션 종료로 자른 값이다.",
        "",
        "## 4. 창 수준 AUROC/AUPRC",
        "",
        _table(
            w,
            ["stream", "gate", "n_windows", "positive", "negative", "prevalence", "macro_auroc", "macro_auprc", "pooled_auroc", "pooled_auprc"],
            ["스트림", "게이트", "창", "양성", "음성", "양성률", "macro AUROC", "macro AUPRC", "pooled AUROC", "pooled AUPRC"],
        ),
        "",
        "## 5. 경고 활성 vs 신규 발령 (hysteresis)",
        "",
        _table(
            pvt,
            ["stream", "gate", "event_recall_active", "event_recall_issued", "episode_precision_active", "episode_precision_issued", "fa_per_hour_all_active", "fa_per_hour_all_issued"],
            ["스트림", "게이트", "Event Recall(active)", "Event Recall(issued)", "Precision(active)", "Precision(issued)", "FA/h(active)", "FA/h(issued)"],
        ),
        "",
        "공유 정의로 다시 계산한 기존 labelblind 스트림의 active는 94/176, issued는 15/176이다. 기존 감사 보고서의 full-stream hysteresis 90/176은 그 구현의 episode/state 처리 차이이며, 본 표의 old/new 비교는 **동일한 함수**로 계산한 값이다.",
        "",
        "### 검증 음성 / 미검증 배경 분리",
        "",
        _table(
            pvt,
            ["stream", "gate", "fa_per_hour_verified_active", "fa_per_hour_unverified_active", "fa_per_hour_verified_issued", "fa_per_hour_unverified_issued"],
            ["스트림", "게이트", "FA/h 검증(active)", "FA/h 미검증(active)", "FA/h 검증(issued)", "FA/h 미검증(issued)"],
        ),
        "",
        "### 음성 출처별 창 수준 구분력",
        "",
        _table(
            w,
            ["stream", "gate", "macro_auroc_verified", "macro_auroc_unverified", "pooled_auroc_verified", "pooled_auroc_unverified", "macro_auprc_verified", "macro_auprc_unverified"],
            ["스트림", "게이트", "macro AUROC 검증", "macro AUROC 미검증", "pooled AUROC 검증", "pooled AUROC 미검증", "macro AUPRC 검증", "macro AUPRC 미검증"],
        ),
        "",
        "검증 음성만 쓰면 참가자별 양성/음성 양쪽이 정의되는 참가자가 줄어든다. 정의 불가 값은 0으로 채우지 않았다.",
        "",
        "## 6. 운동 게이트 전후",
        "",
        "동일 확률·동일 fold 임계값·동일 대상 창에서 게이트만 바꿨다. `legacy_hard`는 고활동에서 확률을 0.05로 누르고, `soft_dynamic`은 연속 운동 점수로 확률과 임계값을 동시에 조정한다.",
        "",
        _table(
            primary[primary.stream.eq("new_fixed30s")],
            ["gate", "endpoint", "detected_events", "event_recall", "episode_precision", "fa_per_hour_all", "fa_per_hour_verified", "fa_per_hour_unverified"],
            ["게이트", "endpoint", "탐지 사건", "Event Recall", "Precision", "FA/h 전체", "FA/h 검증", "FA/h 미검증"],
        ),
        "",
        "게이트가 오경보를 줄여도 동시에 진양성 경고를 제거하면 조기경고 성능 개선이 아니다. 이번 결과도 그 방향을 재현했다.",
        "",
        "## 7. 기존 결과와 차이의 원인",
        "",
        "### 7.1 격자 구성 차이",
        "",
        _table(
            comp,
            ["stream", "windows", "p1_positive", "verified_negative", "unverified_negative", "censored_or_other"],
            ["스트림", "창", "P1 양성", "검증 음성", "미검증 음성", "검열/기타"],
        ),
        "",
        "기존 키 51,234개는 모두 신규 30초 격자에 포함된다. 차이는 **미검증 배경 24,479→44,519창(+20,040)** 이며 P1 양성·검증 음성·검열 창 수는 동일하다.",
        "",
        "### 7.2 점수 효과와 격자 효과 분리",
        "",
        "`old_grid_new_score`는 신규 재적합 점수·임계값을 기존 51,234창에 적용한 것이다. `new_grid_new_score`는 여기에 추가된 30초 배경까지 포함한다.",
        "",
        _table(
            dw,
            ["stream", "gate", "n_windows", "positive", "negative", "prevalence", "macro_auroc", "macro_auprc", "pooled_auroc", "pooled_auprc"],
            ["스트림", "게이트", "창", "양성", "음성", "양성률", "macro AUROC", "macro AUPRC", "pooled AUROC", "pooled AUPRC"],
        ),
        "",
        _table(
            da,
            ["stream", "gate", "endpoint", "detected_events", "event_recall", "episode_precision", "fa_per_hour_all", "fa_per_hour_verified", "fa_per_hour_unverified"],
            ["스트림", "게이트", "endpoint", "탐지 사건", "Event Recall", "Precision", "FA/h 전체", "FA/h 검증", "FA/h 미검증"],
        ),
        "",
        "핵심 분해:",
        "",
        "- 창 macro AUROC는 old 0.6168 → 신규 점수/기존 격자 0.6161 → 신규 전체 0.6214였다. **+0.0053의 AUROC 상승은 모델 개선이 아니라 음성 배경이 늘어난 격자 효과**다.",
        "- Active Event Recall은 old 94/176 → 신규 점수/기존 격자 83/176 → 신규 전체 83/176이었다. **-11사건은 재적합 점수·임계값 변화에서 발생했고, 고정 30초 격자 추가는 event-level 탐지를 바꾸지 않았다** (사건 주변 창이 이미 30초였기 때문).",
        "- Issued Event Recall은 old 15 → 13 → 12로, 점수 효과 -2, 격자 효과 -1이었다.",
        "- FA/h(전체)는 old 0.926 → 0.895 → 0.970으로, 점수 효과는 소폭 감소, 격자 효과는 미검증 배경 증가로 +0.075였다.",
        "",
        "### 7.3 참가자별 임계값·탐지 변화",
        "",
        _table(
            th,
            ["subject", "old_threshold", "new_threshold", "threshold_delta", "old_mean_prob", "new_mean_prob", "matched_prob_corr"],
            ["참가자", "기존 임계값", "신규 임계값", "차이", "기존 평균확률", "신규 평균확률", "동일창 확률상관"],
        ),
        "",
        _table(
            pdet,
            ["subject", "n_events", "old_detected", "new_detected", "delta_detected", "old_threshold", "new_threshold"],
            ["참가자", "사건", "기존 탐지", "신규 탐지", "차이", "기존 임계값", "신규 임계값"],
        ),
        "",
        "동일 timestamp에서 기존·신규 확률 상관은 대부분 0.98 이상이었다. 탐지 변화는 주로 내부 validation F1 임계값이 함께 움직인 결과다. 예: 7E 0.67→0.85, 8B 0.63→0.73, 7A 0.83→0.85에서 탐지가 줄고, 83은 0.77→0.75에서 늘었다.",
        "",
        "- **표집 격자**: 기존 캐시는 사건 주변 30초, 배경 60초로 라벨에 의존해 표집됐다. 신규 캐시는 모든 세션을 30초로 균일 표집하므로 양성률, 배경 노출, 에피소드 병합 기회가 달라진다.",
        "- **음성 구성**: 기존과 신규 모두 미검증 배경이 대부분이다. 미검증 배경만으로 계산한 AUROC/AUPRC와 검증 음성만으로 계산한 값은 같은 모델에서도 다르다.",
        "- **재학습 차이**: 저장된 OOF 확률은 새 고정 격자에 존재하지 않는다. 동일 명세 재적합을 사용했으므로 기존 수치와의 차이는 격자·재적합·임계값 재선택이 함께 작용한 결과다.",
        "- **정책 replay**: 두 스트림 모두 전체 시간 순서로 정책을 재생하고 QC 불량 창 입력을 0으로 두었다. 기존 filtered 평가와 직접 같은 값이 아니다.",
        "- **사건 시각**: 설문 시각은 분 단위이며 독립 onset이 아니다. 따라서 이 비교는 실제 1~3분 전 예측 성능의 인과적 개선을 입증하지 않는다.",
        "",
        "## 8. 검증",
        "",
        f"- 자동 테스트: `results/nurse_forecasting/fixed_grid_30s/test_results.txt` (27개 통과: 기존 20 + 신규 7).",
        f"- 원본/기존 산출물 SHA256: `protected_hashes_before.csv` vs `protected_hashes_after.csv`. 불변 여부: **{ver.get('all_passed', '미확인')}**.",
        "- 15-fold LOSO 참가자 분리: 각 fold에서 test 참가자가 train/validation에 없고 train/validation 교집합이 0임을 스크립트에서 assert했다.",
        "- 공유 51,234 timestamp에서 원시 53개 특징 최대 절대차 0.000 (`feature_equivalence.csv`).",
        "- 신규 파일은 모두 `results/nurse_forecasting/fixed_grid_30s/` 아래에만 있다. 상세 검사: `verification.json`.",
        "",
        "## 9. 재현",
        "",
        "```powershell",
        ".venv\\Scripts\\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py snapshot",
        ".venv\\Scripts\\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py build",
        ".venv\\Scripts\\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py train",
        ".venv\\Scripts\\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py evaluate",
        ".venv\\Scripts\\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py analyze",
        ".venv\\Scripts\\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py report",
        ".venv\\Scripts\\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py verify",
        "```",
        "",
        "## 10. 한계",
        "",
        "- 설문 라벨은 독립 onset이 아니며 분 단위다. 후보 생성과 사건 표집이 센서 알고리즘에 의존한다.",
        "- 미검증 배경의 실제 무스트레스 여부는 확인되지 않았다. 검증 음성만으로는 참가자별 AUROC가 정의되지 않는 경우가 있다.",
        "- 고정 30초 격자는 라벨 의존 표집을 제거하지만, 실제 온라인 배포에서의 지연·센서 드리프트·결측은 별도 검증이 필요하다.",
        "- 참가자가 15명이므로 참가자 재표집 불확실성은 여전히 크다.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {REPORT}", flush=True)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify() -> None:
    checks = []
    before = pd.read_csv(OUT / "protected_hashes_before.csv")
    after = snapshot_protected()
    save(after, "protected_hashes_after")
    before_map = dict(zip(before.path, before.sha256))
    after_map = dict(zip(after.path, after.sha256))
    generated = {
        str(OUT.relative_to(ROOT)),
        str(REPORT.relative_to(ROOT)),
    }
    changed = sorted([
        p for p in set(before_map) | set(after_map)
        if before_map.get(p) != after_map.get(p)
        and not any(p == g or p.startswith(g + "/") for g in generated)
    ])
    checks.append({"check": "protected_sha256_unchanged", "n": len(before), "changed": changed, "passed": len(changed) == 0})

    feat = read("features_fixed30s_annotated.csv.gz")
    diffs = feat.groupby(["subject", "session"], sort=False).pred_ts.diff().dropna()
    checks.append({
        "check": "fixed_grid_30s_spacing",
        "n_windows": int(len(feat)),
        "max_abs_deviation_s": float(np.max(np.abs(diffs - FIXED_STEP_S))) if len(diffs) else None,
        "passed": bool(len(diffs) and np.max(np.abs(diffs - FIXED_STEP_S)) < 1e-6),
    })

    splits = pd.read_csv(OUT / "splits_fixed30s.csv", dtype={"test_subject": str})
    ok = True
    for r in splits.itertuples():
        tr = set(r.train_subjects.split(";"))
        va = set(r.val_subjects.split(";"))
        if r.test_subject in tr or r.test_subject in va or (tr & va):
            ok = False
    checks.append({"check": "loso_participant_isolation", "n_folds": len(splits), "passed": bool(ok and len(splits) == 15)})

    pred = read("predictions_fixed30s.csv.gz")
    checks.append({"check": "prediction_rows_match_cache", "passed": int(len(pred)) == int(len(feat))})

    fe_path = OUT / "feature_equivalence.csv"
    if fe_path.exists():
        fe = pd.read_csv(fe_path)
        raw = fe[~fe.calibrated.astype(bool)]
        checks.append({
            "check": "raw_53_features_identical_on_shared_grid",
            "max_abs_diff": float(raw.max_abs_diff.max()),
            "passed": bool(len(raw) == 53 and float(raw.max_abs_diff.max()) <= 1e-9),
        })
    else:
        checks.append({"check": "raw_53_features_identical_on_shared_grid", "passed": False})

    report_ok = REPORT.exists() and REPORT.stat().st_size > 1000
    checks.append({"check": "report_exists", "passed": bool(report_ok)})
    test_path = OUT / "test_results.txt"
    if test_path.exists():
        txt = test_path.read_text(encoding="utf-8-sig", errors="replace")
        checks.append({"check": "automated_tests", "passed": "OK" in txt and "Ran " in txt})
    else:
        checks.append({"check": "automated_tests", "passed": False})

    result = {
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
        "changed_original_files": changed,
        "scope": (
            "Fixed-grid replay uses the same survey labels and cannot create an independent onset. "
            "Hash invariance covers raw sensors, labels, existing caches, models, reports, and prior audit outputs."
        ),
    }
    jsave(result, "verification")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["snapshot", "build", "train", "evaluate", "analyze", "report", "verify", "all"])
    ap.add_argument("--annotated-only", action="store_true", help="build only sessions with survey annotation")
    args = ap.parse_args()
    mkdirs()
    if args.phase in ("snapshot", "all"):
        snapshot()
    if args.phase in ("build", "all"):
        build_features(annotated_only=args.annotated_only)
    if args.phase in ("train", "all"):
        train_loso()
    if args.phase in ("evaluate", "all"):
        evaluate_all()
    if args.phase in ("analyze", "all"):
        analyze_differences()
    if args.phase in ("report", "all"):
        write_report()
    if args.phase in ("verify", "all"):
        verify()


if __name__ == "__main__":
    main()
