"""Stress-Predict: strictly causal 60 < lead <= 180 s task-onset forecasting.

Run from repository root with .venv/Scripts/python.exe -m
experiments.stress_predict_forecasting.run. No existing experiment is modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from zoneinfo import ZoneInfo

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.features.core import basic, native_slice
from src.forecasting.evaluate import compute_window_metrics

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/Stress-Predict-Dataset"
OUT = ROOT / "results/stress_predict_forecasting"
REPORT = ROOT / "reports/STRESS_PREDICT_FORECASTING_KO.md"
VERSION = "1.0"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
TASKS = [("stroop", "I", "J"), ("interview", "M", "N"), ("hyperventilation", "Q", "R")]
PROTOCOL = dict(version=VERSION, window_s=180, step_s=30,
    positive_lead_s="(60,180]", timestamp_zone="UTC; logs Europe/Dublin",
    marker_log_tolerance_s=120, log_only_boundary_guard_s=60,
    minimum_finite_fraction=.95, maximum_missing_run_s=5,
    threshold=.5, warning_policy="p>=0.5 active; rising edge issues new warning; no label/event resets; retain previous state across sensor gaps",
    model_features="EDA/HR/ACC basic statistics only, no time/ID/order/labels",
    schedule_diagnostic="120-second elapsed-recording bins, train participants only, Laplace prior",
    splits="outer LOSO; inner GroupShuffleSplit 20% validation, seed 42+fold; fixed threshold, no refit on validation",
    rf=dict(n_estimators=200, max_depth=6, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42),
    lr=dict(C=.25, class_weight="balanced", max_iter=2000, random_state=42))


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def obtain():
    if not (RAW / "Raw_data").exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        if __import__("shutil").which("git"):
            subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/italha-d/Stress-Predict-Dataset.git", str(RAW)], check=True)
        else:
            import urllib.request
            target = RAW.parent / "stress_predict_original.zip"
            if not target.exists():
                urllib.request.urlretrieve("https://github.com/italha-d/Stress-Predict-Dataset/archive/refs/heads/main.zip", target)
            with zipfile.ZipFile(target) as z:
                z.extractall(RAW.parent)
            (RAW.parent / "Stress-Predict-Dataset-main").rename(RAW)
    return subprocess.check_output(["git", "-C", str(RAW), "rev-parse", "HEAD"], text=True).strip() if (RAW / ".git").exists() else "zip-main"


def read_logs():
    """Read values from the small metadata workbook without installing Excel packages."""
    with zipfile.ZipFile(RAW / "Processed_data/Time_logs.xlsx") as z:
        shared = ["".join(s.itertext()) for s in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        rows = ET.fromstring(z.read("xl/worksheets/sheet1.xml")).findall("m:sheetData/m:row", NS)
        result = {}
        for row in rows[2:]:
            cells = {}
            for cell in row:
                v = cell.find("m:v", NS)
                if v is not None:
                    key = "".join(filter(str.isalpha, cell.attrib["r"]))
                    cells[key] = shared[int(v.text)] if cell.attrib.get("t") == "s" else float(v.text)
            if "A" in cells:
                result[cells["A"]] = cells
        return result


def log_timestamps(cells, session_mid):
    day = datetime(1899, 12, 30) + timedelta(days=cells["Z"])
    midnight = day.replace(tzinfo=ZoneInfo("Europe/Dublin")).timestamp()
    result = {}
    for k in "BCEFGHIJKLMNOPQRSTUVWX":
        raw = midnight + round(cells[k] * 86400)
        # Workbook mixes 12-hour and 24-hour clock. Resolve AM/PM against the
        # sensor session, never against labels, expected durations or model output.
        result[k] = min((raw - 43200, raw, raw + 43200), key=lambda x: abs(x-session_mid))
    return result


def read_sensor(path, name):
    with path.open() as f:
        h0 = np.asarray([float(x) for x in next(f).strip().split(",")])
        h1 = np.asarray([float(x) for x in next(f).strip().split(",")])
    assert np.all(h0 == h0[0]) and np.all(h1 == h1[0]), "inconsistent channel timestamps/rates"
    fs, t0 = float(h1[0]), float(h0[0])
    assert fs == {"EDA": 4, "HR": 1, "ACC": 32}[name], "unexpected sampling rate"
    assert 1.5e9 < t0 < 1.8e9, "not seconds UNIX timestamp"
    x = pd.read_csv(path, skiprows=2, header=None, skip_blank_lines=False).to_numpy(dtype=float)
    assert x.shape[1] == (3 if name == "ACC" else 1), "wrong channel count"
    bad = ~np.isfinite(x)
    if name == "HR":
        bad |= (x < 30) | (x > 240)
    elif name == "EDA":
        bad |= (x < 0) | (x > 100)
    else:
        x /= 64.0
        bad |= np.abs(x) > 2.1
    x[bad] = np.nan
    return dict(x=x, t0=t0, fs=fs, end=t0+len(x)/fs,
                missing_fraction=float(bad.mean()), rows=len(x), sha256=sha(path))


def nearest_tag(tags, target, tolerance=120):
    distances = np.abs(tags-target)
    ix = np.argsort(distances)
    if not len(ix) or distances[ix[0]] > tolerance:
        return None, "no_marker_within_120s"
    # A close tie is not resolved by assuming task order or duration.
    if len(ix) > 1 and distances[ix[1]] <= tolerance and distances[ix[1]]-distances[ix[0]] < 30:
        return None, "ambiguous_marker_match"
    return int(ix[0]), "matched"


def audit_subject(folder, cells):
    sid = folder.name
    sensors = {k: read_sensor(folder / f"{k}.csv", k) for k in ("EDA", "HR", "ACC")}
    start = min(v["t0"] for v in sensors.values())
    common_start = max(v["t0"] for v in sensors.values())
    end = min(v["end"] for v in sensors.values())
    logs = log_timestamps(cells, (start+end)/2)
    tags_raw = np.loadtxt(folder / f"tags_{sid}.csv", delimiter=",").reshape(-1)
    tags = tags_raw.copy()
    corrections = []
    for i, v in enumerate(tags):
        if not start-120 <= v <= end+120:
            candidates = [v+d for d in (-43200, 43200) if start-120 <= v+d <= end+120]
            if len(candidates) == 1:
                tags[i] = candidates[0]
                corrections.append(dict(index=i, original=v, corrected=tags[i], reason="unique_12h_AM_PM_correction_into_sensor_session"))
    ordered = bool(np.all(np.diff(tags) > 0))
    events = []
    used = []
    for task, a, b in TASKS:
        ia, ra = nearest_tag(tags, logs[a])
        ib, rb = nearest_tag(tags, logs[b])
        valid = ordered and ia is not None and ib is not None and ia < ib and logs[a] < logs[b]
        reason = "matched" if valid else f"start:{ra};end:{rb};ordered:{ordered}"
        ev = dict(subject=sid, event_id=f"{sid}_{task}", task=task,
            log_start_ts=logs[a], log_end_ts=logs[b], start_tag_index=ia, end_tag_index=ib,
            event_start_ts=float(tags[ia]) if ia is not None else np.nan,
            event_end_ts=float(tags[ib]) if ib is not None else np.nan,
            start_log_delta_s=float(tags[ia]-logs[a]) if ia is not None else np.nan,
            end_log_delta_s=float(tags[ib]-logs[b]) if ib is not None else np.nan,
            boundary_valid=bool(valid), boundary_reason=reason)
        events.append(ev)
        used.extend([i for i in (ia, ib) if i is not None])
    # A marker cannot stand for two task boundaries.
    for ev in events:
        if any(used.count(ev[k]) > 1 for k in ("start_tag_index", "end_tag_index") if ev[k] is not None):
            ev["boundary_valid"] = False
            ev["boundary_reason"] = "marker_reused_by_multiple_task_boundaries"
    intervals = []
    def add(a, b, source):
        a, b = max(a, common_start), min(b, end)
        if b > a:
            intervals.append(dict(subject=sid, stable_start_ts=a, stable_end_ts=b, source=source))
    first = events[0]
    if first["boundary_valid"] and logs["G"] < logs["H"] <= first["event_start_ts"]+120:
        add(logs["G"]+60, min(logs["H"]-60, first["event_start_ts"]), "log_only_initial_baseline_guard60s")
    for left, right, a, b in [(events[0], events[1], "K", "L"), (events[1], events[2], "O", "P")]:
        if left["boundary_valid"] and right["boundary_valid"]:
            x, y = left["event_end_ts"], right["event_start_ts"]
            # Relaxation must be explicitly documented and bounded by the two
            # actually observed adjoining task markers. No fixed task lengths.
            if x < y and logs[a] < logs[b] and abs(logs[a]-x) <= 120 and abs(logs[b]-y) <= 120:
                add(x, y, f"tag_bounded_documented_relax_{a}{b}")
    last = events[2]
    if last["boundary_valid"]:
        i, reason = nearest_tag(tags, logs["T"])
        if i is not None and logs["S"] < logs["T"] and abs(logs["S"]-last["event_end_ts"]) <= 120:
            add(last["event_end_ts"], tags[i], "tag_bounded_post_hypervent_relax_ST")
        if logs["W"] < logs["X"] and logs["W"] >= last["event_end_ts"]:
            add(logs["W"]+60, logs["X"]-60, "log_only_final_baseline_guard60s")
    # Reject any stable span intersecting an observed stress task; deduplicate
    # overlapping spans without moving the grid or inferring new boundaries.
    intervals = [r for r in intervals if not any(e["boundary_valid"] and
        max(r["stable_start_ts"], e["event_start_ts"]) < min(r["stable_end_ts"], e["event_end_ts"]) for e in events)]
    audit = dict(subject=sid, start_ts=start, common_start_ts=common_start, common_end_ts=end,
        duration_s=end-start, tags_raw=tags_raw.tolist(), tags_corrected=tags.tolist(),
        tags_ordered_raw=bool(np.all(np.diff(tags_raw)>0)), tags_ordered_after=ordered,
        tag_corrections=corrections, logs_utc=logs,
        sensors={k: {a:b for a,b in v.items() if a != "x"} for k,v in sensors.items()},
        stable_intervals=len(intervals), validated_events=sum(e["boundary_valid"] for e in events))
    return audit, events, intervals, sensors


def quality(x, fs, sensor):
    valid = np.isfinite(x).all(axis=1)
    z = np.r_[True, valid, True]
    changes = np.flatnonzero(z[1:] != z[:-1])
    max_run = max((changes[1::2]-changes[::2]), default=0) / fs
    good = valid.mean() >= .95 and max_run <= 5
    if sensor == "EDA" and np.any(valid):
        good &= np.nanmean(x) > .01  # sustained zero/noncontact is not a physiological precursor
    return bool(good)


def window_features(sensors, t):
    features = {}
    for name, s in sensors.items():
        a = t-180-s["t0"]
        if a < 0 or t > s["end"]:
            return None
        x = native_slice(s["x"], a, t-s["t0"], s["fs"])
        if len(x) < 180*s["fs"]-1 or not quality(x, s["fs"], name):
            return None
        streams = [(name.lower(), x[:, 0])] if name != "ACC" else [("acc_"+axis, x[:,j]) for j,axis in enumerate("xyz")] + [("acc_mag", np.linalg.norm(x, axis=1))]
        for prefix, series in streams:
            base = basic(series, s["fs"], prefix+"_")
            for stat in ("mean", "std", "range", "slope", "absdiff_per_s"):
                features[prefix+"_"+stat] = base[prefix+"_"+stat]
            features[prefix+"_delta"] = float(np.nanmean(series[-int(30*s["fs"]):])-np.nanmean(series[:int(30*s["fs"])]))
    return features


def label_at(t, intervals, events):
    stable = [r for r in intervals if r["stable_start_ts"] <= t-180 and t <= r["stable_end_ts"]]
    if not stable:
        return -1, "history_not_entirely_confirmed_stable", "", np.nan
    # An unresolved task's surrounding epoch is never accepted as negative.
    future = [e for e in events if e["boundary_valid"] and e["event_start_ts"] > t]
    next_ev = min(future, key=lambda e:e["event_start_ts"]) if future else None
    lead = next_ev["event_start_ts"]-t if next_ev else np.nan
    if next_ev and 60 < lead <= 180:
        return 1, "future_task_onset", next_ev["event_id"], lead
    if next_ev and 0 < lead <= 60:
        return -1, "within_60s_excluded", next_ev["event_id"], lead
    if any(t+180 <= r["stable_end_ts"] for r in stable) and (not next_ev or lead > 180):
        return 0, "confirmed_stable_180s_followup", "", lead
    return -1, "unknown_followup", "", lead


def process_subject(folder, cells, fingerprint):
    fingerprint += hashlib.sha256(repr([(p.name,p.stat().st_size,p.stat().st_mtime_ns)
        for p in sorted(folder.glob("*.csv")) if p.name in {"EDA.csv","HR.csv","ACC.csv",f"tags_{folder.name}.csv"}]).encode()).hexdigest()
    cache = OUT / "feature_cache" / f"{folder.name}.joblib"
    if cache.exists():
        data = joblib.load(cache)
        if data["fingerprint"] == fingerprint:
            return data
    audit, events, intervals, sensors = audit_subject(folder, cells)
    rows = []
    for rel in np.arange(180, audit["common_end_ts"]-audit["start_ts"]+1e-6, 30):
        t = audit["start_ts"]+rel
        feats = window_features(sensors, t)
        y, reason, target, lead = label_at(t, intervals, events)
        rows.append(dict(subject=folder.name, pred_ts=t, pred_rel_s=rel,
            history_start_ts=t-180, history_end_exclusive_ts=t,
            y=y if feats is not None else -1, label_reason=reason if feats is not None else "sensor_quality",
            sensor_valid=feats is not None, target_event_id=target, lead_s=lead, **(feats or {})))
    data = dict(fingerprint=fingerprint, audit=audit, events=events, intervals=intervals, rows=rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, cache)
    return data


def extract():
    commit = obtain()
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / "protocol.json", PROTOCOL)
    logs = read_logs()
    fingerprint = sha(__file__)+commit+sha(RAW / "Processed_data/Time_logs.xlsx")
    folders = sorted((RAW / "Raw_data").glob("S*"))
    failures, data = [], []
    def safe(folder):
        try:
            return process_subject(folder, logs[folder.name], fingerprint)
        except Exception:
            return dict(failed_subject=folder.name, traceback=traceback.format_exc())
    with ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as pool:
        for d in pool.map(safe, folders):
            if "failed_subject" in d:
                failures.append(d)
            else:
                data.append(d)
    dump(OUT / "subject_audit.json", [d["audit"] for d in data])
    dump(OUT / "failed_subjects.json", failures)
    events = pd.DataFrame([e for d in data for e in d["events"]])
    intervals = pd.DataFrame([r for d in data for r in d["intervals"]])
    frame = pd.DataFrame([r for d in data for r in d["rows"]])
    events["n_positive_windows"] = events.event_id.map(frame[frame.y == 1].groupby("target_event_id").size()).fillna(0).astype(int)
    events["forecast_eligible"] = events.boundary_valid & (events.n_positive_windows > 0)
    events.to_csv(OUT / "events.csv", index=False)
    intervals.to_csv(OUT / "stable_intervals.csv", index=False)
    frame.to_csv(OUT / "features.csv.gz", index=False)
    dump(OUT / "provenance.json", dict(repository="https://github.com/italha-d/Stress-Predict-Dataset", commit=commit,
        time_logs_sha256=sha(RAW / "Processed_data/Time_logs.xlsx"), script_sha256=sha(__file__),
        python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__,
        participants_downloaded=len(folders), participants_read=len(data), failures=len(failures)))
    print(json.dumps(dict(subjects=len(data), failures=len(failures), events=int(events.boundary_valid.sum()),
        eligible_events=int(events.forecast_eligible.sum()), positive=int((frame.y==1).sum()), negative=int((frame.y==0).sum()))), flush=True)
    return frame, events


def build_model(name):
    if name == "Dummy":
        estimator = DummyClassifier(strategy="prior")
    elif name == "LR":
        estimator = LogisticRegression(**PROTOCOL["lr"])
    elif name == "RF":
        estimator = RandomForestClassifier(**PROTOCOL["rf"], n_jobs=1)
    else:
        raise ValueError(name)
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), estimator)


def warning_states(prob):
    """Full-record grid, no event/state/ground-truth dependent reset."""
    active, issued, previous = [], [], False
    for p in prob:
        if not np.isfinite(p):
            active.append(False)
            issued.append(False)
            continue  # uncertainty must not manufacture a fresh rising edge
        state = bool(p >= .5)
        active.append(state)
        issued.append(state and not previous)
        previous = state
    return np.asarray(active), np.asarray(issued)


def event_results(pred, events):
    rows = []
    for e in events.itertuples():
        s = pred[pred.subject == e.subject]
        lead = e.event_start_ts-s.pred_ts
        horizon = s[(lead > 60) & (lead <= 180) & s.sensor_valid]
        # Full-record operational policy can issue during excluded states, but a
        # valid success must also have a confirmed stable 180-second history.
        eligible = horizon[horizon.y == 1]
        new = eligible[eligible.newly_issued_warning]
        active = eligible[eligible.active_warning]
        rows.append(dict(subject=e.subject, event_id=e.event_id, task=e.task,
            event_start_ts=e.event_start_ts, boundary_valid=e.boundary_valid,
            forecast_eligible=e.forecast_eligible, n_positive_windows=e.n_positive_windows,
            detected_new=bool(len(new)), detected_active=bool(len(active)),
            first_new_warning_ts=float(new.pred_ts.min()) if len(new) else np.nan,
            lead_time_s=float(e.event_start_ts-new.pred_ts.min()) if len(new) else np.nan,
            max_horizon_probability=float(eligible.prob.max()) if len(eligible) else np.nan))
    return pd.DataFrame(rows)


def aggregate(pred, event_frame):
    work = pred[pred.y.isin([0,1]) & pred.prob.notna()]
    if len(work):
        m = compute_window_metrics(work.y, work.prob, work.active_warning.astype(int))
    else:
        # Some fully audited sessions contain no admissible supervised windows.
        # Keep their operational predictions and event denominator nevertheless.
        m = {k:np.nan for k in ("prevalence", "auroc", "auprc", "accuracy", "balanced_acc", "precision", "recall", "specificity", "f1", "fpr", "fnr", "brier")}
        m.update({k:0 for k in ("n_windows", "pos_windows", "neg_windows", "tp", "fp", "tn", "fn")})
    ev = event_frame[event_frame.boundary_valid]
    eligible = ev[ev.forecast_eligible]
    # Each verified-negative evaluation time represents a prospective 30-second
    # observation interval, clipped to the next label state by the strict grid.
    # The horizon is checked for every grid point; no censored rows enter rates.
    neg = work[work.y==0]
    hours = len(neg)*30/3600
    false = int(neg.newly_issued_warning.sum())
    lead = eligible.loc[eligible.detected_new, "lead_time_s"].dropna()
    return dict(**m, validated_events=len(ev), eligible_events=len(eligible),
        detected_new=int(eligible.detected_new.sum()),
        event_recall_new=float(eligible.detected_new.mean()) if len(eligible) else np.nan,
        event_recall_active=float(eligible.detected_active.mean()) if len(eligible) else np.nan,
        event_recall_all_validated=float(ev.detected_new.mean()) if len(ev) else np.nan,
        false_new_warnings=false, verified_negative_hours=hours,
        false_alarms_per_hour=false/hours if hours else np.nan,
        mean_lead_s=float(lead.mean()) if len(lead) else np.nan,
        median_lead_s=float(lead.median()) if len(lead) else np.nan)


def train(frame, events, selected=None):
    cols = [c for c in frame if c.startswith(("eda_", "hr_", "acc_"))]
    assert len(cols) == 36 and all(frame[cols].dtypes.apply(lambda x: np.issubdtype(x, np.number)))
    dump(OUT / "feature_columns.json", cols)
    # Include subjects without supervised windows in held-out operational
    # predictions and the all-validated-event denominator (e.g. S06).
    subjects = sorted(frame.subject.unique())
    if len(subjects) < 4 or frame.y.eq(1).sum() < 10 or frame.y.eq(0).sum() < 10:
        dump(OUT / "insufficient_data.json", dict(reason="insufficient participants or positive/negative windows"))
        return
    tasks = [(m, "all", cols) for m in ("Dummy", "LR", "RF")]
    tasks += [("LR", s, [c for c in cols if c.startswith(s+"_")]) for s in ("eda", "hr", "acc")]
    tasks += [("Schedule", "time_only_diagnostic", [])]
    if selected:
        tasks = [t for t in tasks if t[0] == selected]
    summary, all_predictions, all_events, per_subject, split_rows, validation_rows = [], [], [], [], [], []
    if selected and (OUT / "predictions.csv.gz").exists():
        def old(filename):
            d = pd.read_csv(OUT / filename)
            return d[d.model != selected]
        summary = old("model_comparison.csv").to_dict("records")
        all_predictions = [old("predictions.csv.gz")]
        all_events = [old("event_early_warnings.csv")]
        per_subject = old("per_subject_metrics.csv").to_dict("records")
        split_rows = old("fold_splits.csv").to_dict("records")
        validation_rows = old("validation_metrics.csv").to_dict("records")
    for name, sensors, features in tasks:
        def fold_job(item):
            fold, sid = item
            work = frame[frame.y.isin([0,1])]
            dev = work[work.subject != sid]
            tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=42+fold).split(dev, dev.y, dev.subject))
            tr, va = dev.iloc[tr], dev.iloc[va]
            te = frame[frame.subject == sid].copy().sort_values("pred_ts")
            assert not (set(tr.subject) & set(va.subject)) and sid not in set(tr.subject) | set(va.subject)
            assert tr.y.nunique() == 2, "train fold missing outcome class"
            model_path = OUT / "models" / f"{name}_{sensors}_{sid}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if name == "Schedule":
                bins = np.floor(tr.pred_rel_s/120).astype(int)
                stats = tr.groupby(bins).y.agg(["sum", "count"])
                prior = float(tr.y.mean())
                rates = ((stats["sum"]+prior*4)/(stats["count"]+4)).to_dict()
                p = np.floor(te.pred_rel_s/120).map(rates).fillna(prior).to_numpy()
                p[~te.sensor_valid.to_numpy(bool)] = np.nan
                pv = np.floor(va.pred_rel_s/120).map(rates).fillna(prior).to_numpy()
                model = dict(bin_seconds=120, rates=rates, prior=prior)
            else:
                model = build_model(name)
                model.fit(tr[features], tr.y)
                p = np.full(len(te), np.nan)
                valid = te.sensor_valid.to_numpy(bool)
                p[valid] = model.predict_proba(te.loc[valid, features])[:,1]
                pv = model.predict_proba(va[features])[:,1]
            joblib.dump(dict(model=model, features=features, threshold=.5, training_subjects=sorted(tr.subject.unique()),
                validation_subjects=sorted(va.subject.unique()), held_out_subject=sid, protocol=PROTOCOL), model_path)
            te["prob"] = p
            te["model"], te["sensors"] = name, sensors
            te["active_warning"], te["newly_issued_warning"] = warning_states(p)
            ef = event_results(te, events[events.subject==sid])
            ef["model"], ef["sensors"] = name, sensors
            sm = dict(model=name, sensors=sensors, subject=sid, **aggregate(te, ef))
            split = dict(model=name, sensors=sensors, test_subject=sid,
                train_subjects=";".join(sorted(tr.subject.unique())), validation_subjects=";".join(sorted(va.subject.unique())),
                train_windows=len(tr), validation_windows=len(va), test_windows=int(te.y.isin([0,1]).sum()), threshold=.5)
            vm = dict(model=name, sensors=sensors, test_subject=sid, **compute_window_metrics(va.y, pv, pv>=.5))
            return te, ef, sm, split, vm
        with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as pool:
            outputs = list(pool.map(fold_job, enumerate(subjects)))
        pp = pd.concat([x[0] for x in outputs], ignore_index=True)
        ee = pd.concat([x[1] for x in outputs], ignore_index=True)
        ss = [x[2] for x in outputs]
        metric = dict(model=name, sensors=sensors, **aggregate(pp, ee),
            subject_macro_auroc=float(pd.DataFrame(ss).auroc.mean()), subject_macro_auprc=float(pd.DataFrame(ss).auprc.mean()))
        summary.append(metric)
        all_predictions.append(pp[["subject", "pred_ts", "pred_rel_s", "y", "label_reason", "sensor_valid", "target_event_id", "lead_s", "prob", "model", "sensors", "active_warning", "newly_issued_warning"]])
        all_events.append(ee)
        per_subject.extend(ss)
        split_rows.extend(x[3] for x in outputs)
        validation_rows.extend(x[4] for x in outputs)
        pd.DataFrame(summary).to_csv(OUT / "model_comparison.csv", index=False)
        print(f"{name}/{sensors}: AUROC={metric['auroc']:.4f}, AP={metric['auprc']:.4f}, new-event-recall={metric['event_recall_new']:.4f}, FA/h={metric['false_alarms_per_hour']:.3f}", flush=True)
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT / "predictions.csv.gz", index=False)
    pd.concat(all_events, ignore_index=True).to_csv(OUT / "event_early_warnings.csv", index=False)
    pd.DataFrame(per_subject).to_csv(OUT / "per_subject_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "fold_splits.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUT / "validation_metrics.csv", index=False)


def verify(frame, events):
    checks = {}
    positive = frame[frame.y == 1]
    checks["positive_events_exist"] = len(positive) > 0
    checks["positive_horizon_strict"] = bool(((positive.lead_s > 60) & (positive.lead_s <= 180)).all())
    checks["history_duration_180s"] = bool(np.allclose(frame.pred_ts-frame.history_start_ts, 180))
    checks["fixed_grid_30s"] = bool(np.allclose((frame.pred_rel_s-180)%30, 0))
    checks["negative_no_onset_within_180s"] = all(not any(e.boundary_valid and 0 < e.event_start_ts-r.pred_ts <= 180
        for e in events[events.subject==r.subject].itertuples()) for r in frame[frame.y==0].itertuples())
    toy_intervals = [dict(stable_start_ts=0., stable_end_ts=1000.)]
    toy_events = [dict(boundary_valid=True, event_start_ts=1000., event_id="toy")]
    checks["lead_boundary_60_excluded_180_included"] = label_at(940,toy_intervals,toy_events)[0] == -1 and label_at(820,toy_intervals,toy_events)[0] == 1 and label_at(819,toy_intervals,toy_events)[0] == 0
    checks["no_current_stress_in_training_history"] = all(not any(e.boundary_valid and
        max(r.history_start_ts,e.event_start_ts) < min(r.pred_ts,e.event_end_ts)
        for e in events[events.subject==r.subject].itertuples()) for r in frame[frame.y.isin([0,1])].itertuples())
    # Regression test: perturb every sample at/after t. Features must be identical.
    rng = np.random.default_rng(42)
    synthetic = {k:dict(t0=0., fs=fs, end=600., x=rng.uniform(1,2,(600*fs,3 if k=="ACC" else 1))) for k,fs in [("EDA",4),("HR",1),("ACC",32)]}
    before = window_features(synthetic, 300)
    for s in synthetic.values():
        s["x"][300*int(s["fs"]):] = 9999
    checks["future_sample_perturbation_invariance"] = before == window_features(synthetic, 300)
    a,n = warning_states([.9,.9,.9,.2,.9])
    checks["active_vs_new_distinct"] = a.tolist()==[True,True,True,False,True] and n.tolist()==[True,False,False,False,True]
    _,n = warning_states([.9,np.nan,.9])
    checks["missing_gap_does_not_create_new_warning"] = n.tolist()==[True,False,False]
    if (OUT / "fold_splits.csv").exists():
        splits = pd.read_csv(OUT / "fold_splits.csv")
        checks["train_validation_test_subject_disjoint"] = all(
            r.test_subject not in set(r.train_subjects.split(";"))|set(r.validation_subjects.split(";")) and
            not set(r.train_subjects.split(";"))&set(r.validation_subjects.split(";")) for r in splits.itertuples())
        checks["threshold_fixed_before_test"] = bool((splits.threshold == .5).all())
    dump(OUT / "validation_checks.json", checks)
    assert all(checks.values()), checks
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "features", "train", "verify", "report"], default="all")
    parser.add_argument("--model", choices=["Dummy", "LR", "RF", "Schedule"], default=None)
    args = parser.parse_args()
    if args.stage in ("all", "features"):
        frame, events = extract()
    else:
        frame = pd.read_csv(OUT / "features.csv.gz")
        events = pd.read_csv(OUT / "events.csv")
    verify(frame, events)
    if args.stage in ("all", "train"):
        train(frame, events, args.model)
        verify(frame, events)
    if args.stage in ("all", "train", "report") and (OUT / "predictions.csv.gz").exists():
        from experiments.stress_predict_forecasting.report import make_report
        make_report()


if __name__ == "__main__":
    main()
