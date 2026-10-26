"""Loader and auditor for the Hospital Nurses Stress Dataset (Hosseini et al., Scientific Data 2022).

Reads raw Empatica E4 zip archives (`EDA.csv`, `BVP.csv`, `ACC.csv`, `HR.csv`, `IBI.csv`, `TEMP.csv`, `tags.csv`)
and `SurveyResults.xlsx` without third-party Excel dependencies (uses standard library `zipfile` + `xml.etree`).
Strictly preserves raw files as read-only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NURSE_RATES = {"EDA": 4.0, "BVP": 64.0, "ACC": 32.0, "TEMP": 4.0, "HR": 1.0}
SURVEY_TIMEZONE = "America/Chicago"


def resolve_nurse_root(root_candidate: str | Path | None = None) -> Path:
    """Resolve the Nurse_Stress dataset directory across local project paths."""
    if root_candidate is not None:
        p = Path(root_candidate)
        if p.exists():
            return p
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "data" / "raw" / "Nurse_Stress",
        Path("/data/raw/Nurse_Stress"),
        Path("D:/Autism/data/raw/Nurse_Stress"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not locate Nurse_Stress dataset directory.")


def parse_survey_results_xlsx(xlsx_path: str | Path, tz_name: str = SURVEY_TIMEZONE) -> pd.DataFrame:
    """Parse SurveyResults.xlsx using standard library XML parsing and align timestamps to UTC epoch.

    Hospital location: University of Louisiana at Lafayette partner hospital (America/Chicago, UTC-5 CDT / UTC-6 CST).
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"SurveyResults.xlsx not found at {xlsx_path}")

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(xlsx_path, "r") as z:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            s_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in s_root.findall(f"{ns}si"):
                t_els = si.findall(f".//{ns}t")
                shared_strings.append("".join(t.text or "" for t in t_els))

        s1 = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows_data: list[dict[str, str | None]] = []
        for row in s1.findall(f".//{ns}row"):
            r_vals: dict[str, str | None] = {}
            for c in row.findall(f"{ns}c"):
                ref = c.attrib.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                t = c.attrib.get("t", "")
                v_el = c.find(f"{ns}v")
                val = v_el.text if v_el is not None else None
                if val is not None and t == "s":
                    val = shared_strings[int(val)]
                r_vals[col] = val
            rows_data.append(r_vals)

    if not rows_data:
        return pd.DataFrame()

    headers = {k: (v.strip() if v else k) for k, v in rows_data[0].items()}
    records = []
    for row_idx, r in enumerate(rows_data[1:], start=1):
        rec = {headers[k]: r.get(k, None) for k in headers}
        sid = rec.get("ID")
        if sid is None or str(sid).strip() == "":
            continue
        rec["excel_row_index"] = row_idx
        records.append(rec)

    df = pd.DataFrame(records)
    df["subject"] = df["ID"].astype(str).str.strip()
    df["stress_level_raw"] = df["Stress level"].astype(str).str.strip().str.lower()

    # Map label category
    def classify_stress_label(val: str) -> str:
        if val in ("2", "2.0"):
            return "confirmed_high_stress"
        if val in ("1", "1.0"):
            return "confirmed_moderate_stress"
        if val in ("0", "0.0"):
            return "survey_confirmed_no_stress"
        return "unconfirmed_algorithm_candidate"

    df["label_category"] = [classify_stress_label(v) for v in df["stress_level_raw"]]
    df["is_confirmed_stress"] = df["stress_level_raw"].isin(["1", "1.0", "2", "2.0"])
    df["is_high_stress"] = df["stress_level_raw"].isin(["2", "2.0"])
    df["is_unconfirmed_na"] = ~df["stress_level_raw"].isin(["0", "0.0", "1", "1.0", "2", "2.0"])

    tz = ZoneInfo(tz_name)
    base_excel = datetime(1899, 12, 30)

    dt_starts = []
    dt_ends = []
    t_starts = []
    t_ends = []
    sec_resolutions = []

    for _, r in df.iterrows():
        d_serial = float(r["date"])
        t_s = float(r["Start time"])
        t_e = float(r["End time"])
        # Round to nearest second to avoid floating point representation drift
        s_sec = int(round(t_s * 86400.0))
        e_sec = int(round(t_e * 86400.0))
        dt_s = base_excel + timedelta(days=int(round(d_serial)), seconds=s_sec)
        dt_e = base_excel + timedelta(days=int(round(d_serial)), seconds=e_sec)
        if dt_e < dt_s:
            dt_e += timedelta(days=1)

        dt_starts.append(dt_s)
        dt_ends.append(dt_e)
        ts_utc = dt_s.replace(tzinfo=tz).timestamp()
        te_utc = dt_e.replace(tzinfo=tz).timestamp()
        t_starts.append(ts_utc)
        t_ends.append(te_utc)
        sec_resolutions.append(60 if (dt_s.second == 0 and dt_e.second == 0) else 1)

    df["dt_start_local"] = dt_starts
    df["dt_end_local"] = dt_ends
    df["event_start_ts"] = t_starts
    df["event_end_ts"] = t_ends
    df["duration_s"] = df["event_end_ts"] - df["event_start_ts"]
    df["duration_min"] = df["duration_s"] / 60.0
    df["time_resolution_s"] = sec_resolutions

    df = df.sort_values(["subject", "event_start_ts", "event_end_ts"]).reset_index(drop=True)
    df["event_id"] = [f"EV_{i:03d}_{r['subject']}" for i, r in df.iterrows()]
    return df


def parse_e4_csv_bytes(raw_bytes: bytes, is_acc: bool = False) -> tuple[float, float, np.ndarray]:
    """Parse Empatica E4 single-channel or 3-axis ACC CSV bytes inside a zip archive."""
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return np.nan, np.nan, np.empty((0, 3) if is_acc else (0,), dtype=float)

    t0 = float(lines[0].split(",")[0].strip())
    fs = float(lines[1].split(",")[0].strip())
    if len(lines) == 2:
        return t0, fs, np.empty((0, 3) if is_acc else (0,), dtype=float)

    body = "\n".join(lines[2:])
    arr = pd.read_csv(io.StringIO(body), header=None).to_numpy(dtype=float)
    if not is_acc and arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    return t0, fs, arr


def parse_ibi_csv_bytes(raw_bytes: bytes) -> tuple[float, np.ndarray]:
    """Parse Empatica E4 IBI.csv bytes: returns (t0_epoch, array of shape (N, 2) [rel_time_s, ibi_s])."""
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return np.nan, np.empty((0, 2), dtype=float)
    t0 = float(lines[0].split(",")[0].strip())
    if len(lines) == 1:
        return t0, np.empty((0, 2), dtype=float)
    body = "\n".join(lines[1:])
    try:
        arr = pd.read_csv(io.StringIO(body), header=None).to_numpy(dtype=float)
        if arr.ndim == 1 and len(arr) == 2:
            arr = arr.reshape(1, 2)
        return t0, arr
    except Exception:
        return t0, np.empty((0, 2), dtype=float)


def load_nurse_session_zip(zip_path: str | Path, include_temp: bool = True) -> dict:
    """Load all sensor modalities from a single Empatica E4 session zip archive."""
    zip_path = Path(zip_path)
    subject = zip_path.parent.name
    session_id = zip_path.stem

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        t0_eda, fs_eda, eda = (
            parse_e4_csv_bytes(zf.read("EDA.csv"), is_acc=False)
            if "EDA.csv" in names
            else (np.nan, 4.0, np.empty((0,), dtype=float))
        )
        t0_bvp, fs_bvp, bvp = (
            parse_e4_csv_bytes(zf.read("BVP.csv"), is_acc=False)
            if "BVP.csv" in names
            else (np.nan, 64.0, np.empty((0,), dtype=float))
        )
        t0_acc, fs_acc, acc = (
            parse_e4_csv_bytes(zf.read("ACC.csv"), is_acc=True)
            if "ACC.csv" in names
            else (np.nan, 32.0, np.empty((0, 3), dtype=float))
        )
        t0_hr, fs_hr, hr = (
            parse_e4_csv_bytes(zf.read("HR.csv"), is_acc=False)
            if "HR.csv" in names
            else (np.nan, 1.0, np.empty((0,), dtype=float))
        )
        if include_temp and "TEMP.csv" in names:
            t0_temp, fs_temp, temp = parse_e4_csv_bytes(zf.read("TEMP.csv"), is_acc=False)
        else:
            t0_temp, fs_temp, temp = np.nan, 4.0, np.empty((0,), dtype=float)

        t0_ibi, ibi = (
            parse_ibi_csv_bytes(zf.read("IBI.csv"))
            if "IBI.csv" in names
            else (np.nan, np.empty((0, 2), dtype=float))
        )
        tags = []
        if "tags.csv" in names:
            for ln in zf.read("tags.csv").decode("utf-8", errors="replace").splitlines():
                if ln.strip():
                    try:
                        tags.append(float(ln.strip().split(",")[0]))
                    except ValueError:
                        pass

    # Common session start and duration from primary wrist sensors (EDA, BVP, ACC)
    valid_starts = [t for t in (t0_eda, t0_bvp, t0_acc) if np.isfinite(t)]
    start_ts = float(min(valid_starts)) if valid_starts else np.nan
    dur_eda = len(eda) / fs_eda if fs_eda > 0 else 0.0
    dur_bvp = len(bvp) / fs_bvp if fs_bvp > 0 else 0.0
    dur_acc = len(acc) / fs_acc if fs_acc > 0 else 0.0
    duration_s = float(min(dur_eda, dur_bvp, dur_acc))
    end_ts = start_ts + duration_s if np.isfinite(start_ts) else np.nan

    return {
        "subject": subject,
        "session": session_id,
        "zip_path": str(zip_path),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_s": duration_s,
        "durations_s": {"EDA": dur_eda, "BVP": dur_bvp, "ACC": dur_acc},
        "start_timestamps": {"EDA": t0_eda, "BVP": t0_bvp, "ACC": t0_acc, "HR": t0_hr, "TEMP": t0_temp, "IBI": t0_ibi},
        "rates": {"EDA": fs_eda, "BVP": fs_bvp, "ACC": fs_acc, "HR": fs_hr, "TEMP": fs_temp},
        "EDA": eda,
        "BVP": bvp,
        "ACC": acc,
        "HR": hr,
        "TEMP": temp,
        "IBI": ibi,
        "tags": np.asarray(tags, dtype=float),
    }


def audit_nurse_dataset(root_dir: str | Path | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Perform a comprehensive structural, temporal, and label audit of the Nurse_Stress dataset.

    Returns:
      - manifest_dict: dataset_manifest.json structure
      - subject_summary_df: per-subject statistics (Table 1)
      - event_audit_df: per-event audit table (Table 2)
    """
    root = resolve_nurse_root(root_dir)
    survey_path = root / "SurveyResults.xlsx"
    survey_df = parse_survey_results_xlsx(survey_path)

    subj_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    session_rows = []
    tags_total = 0

    for sd in subj_dirs:
        sid = sd.name
        for zp in sorted(sd.glob("*.zip")):
            fsize = zp.stat().st_size
            with zipfile.ZipFile(zp, "r") as zf:
                names = set(zf.namelist())
                eda_lines = zf.read("EDA.csv").splitlines() if "EDA.csv" in names else []
                bvp_lines = zf.read("BVP.csv").splitlines() if "BVP.csv" in names else []
                acc_lines = zf.read("ACC.csv").splitlines() if "ACC.csv" in names else []
                hr_lines = zf.read("HR.csv").splitlines() if "HR.csv" in names else []
                ibi_lines = zf.read("IBI.csv").splitlines() if "IBI.csv" in names else []
                temp_lines = zf.read("TEMP.csv").splitlines() if "TEMP.csv" in names else []
                tags_lines = [
                    l.decode().strip() for l in zf.read("tags.csv").splitlines() if l.strip()
                ] if "tags.csv" in names else []

                t0_eda = float(eda_lines[0].decode().split(",")[0]) if len(eda_lines) >= 2 else np.nan
                fs_eda = float(eda_lines[1].decode().split(",")[0]) if len(eda_lines) >= 2 else 4.0
                n_eda = max(0, len(eda_lines) - 2)
                dur_eda = n_eda / fs_eda if fs_eda > 0 else 0.0

                t0_bvp = float(bvp_lines[0].decode().split(",")[0]) if len(bvp_lines) >= 2 else np.nan
                fs_bvp = float(bvp_lines[1].decode().split(",")[0]) if len(bvp_lines) >= 2 else 64.0
                n_bvp = max(0, len(bvp_lines) - 2)
                dur_bvp = n_bvp / fs_bvp if fs_bvp > 0 else 0.0

                t0_acc = float(acc_lines[0].decode().split(",")[0]) if len(acc_lines) >= 2 else np.nan
                fs_acc = float(acc_lines[1].decode().split(",")[0]) if len(acc_lines) >= 2 else 32.0
                n_acc = max(0, len(acc_lines) - 2)
                dur_acc = n_acc / fs_acc if fs_acc > 0 else 0.0

                n_hr = max(0, len(hr_lines) - 2)
                n_ibi = max(0, len(ibi_lines) - 1)
                n_temp = max(0, len(temp_lines) - 2)
                dur_common = min(dur_eda, dur_bvp, dur_acc)
                tags_total += len(tags_lines)

                session_rows.append(
                    {
                        "subject": sid,
                        "session": zp.stem,
                        "zip_name": zp.name,
                        "zip_path": str(zp),
                        "size_bytes": fsize,
                        "has_eda": "EDA.csv" in names,
                        "has_bvp": "BVP.csv" in names,
                        "has_acc": "ACC.csv" in names,
                        "has_hr": "HR.csv" in names,
                        "has_ibi": "IBI.csv" in names,
                        "has_temp": "TEMP.csv" in names,
                        "start_ts": t0_eda,
                        "end_ts": t0_eda + dur_common if np.isfinite(t0_eda) else np.nan,
                        "dur_eda_s": dur_eda,
                        "dur_bvp_s": dur_bvp,
                        "dur_acc_s": dur_acc,
                        "dur_common_s": dur_common,
                        "fs_eda": fs_eda,
                        "fs_bvp": fs_bvp,
                        "fs_acc": fs_acc,
                        "n_eda": n_eda,
                        "n_bvp": n_bvp,
                        "n_acc": n_acc,
                        "n_hr": n_hr,
                        "n_ibi": n_ibi,
                        "n_temp": n_temp,
                        "n_tags": len(tags_lines),
                        "start_spread_s": float(max(abs(t0_eda - t0_bvp), abs(t0_eda - t0_acc))),
                        "dur_spread_s": float(max(dur_eda, dur_bvp, dur_acc) - dur_common),
                    }
                )

    sess_df = pd.DataFrame(session_rows).sort_values(["subject", "start_ts"]).reset_index(drop=True)

    # Check overlapping or out-of-order sessions within each subject
    sess_df["gap_from_prev_s"] = np.nan
    sess_df["is_overlapping_prev"] = False
    for sid, g in sess_df.groupby("subject", sort=False):
        prev_end = None
        for idx, r in g.iterrows():
            if prev_end is not None:
                gap = r["start_ts"] - prev_end
                sess_df.loc[idx, "gap_from_prev_s"] = gap
                if gap < -1.0:
                    sess_df.loc[idx, "is_overlapping_prev"] = True
            prev_end = max(prev_end, r["end_ts"]) if prev_end is not None else r["end_ts"]

    # Audit each survey event against sensor sessions
    event_audit_rows = []
    for _, ev in survey_df.iterrows():
        sid = ev["subject"]
        ts = ev["event_start_ts"]
        te = ev["event_end_ts"]
        sub_sess = sess_df[sess_df["subject"] == sid]

        # Find session covering onset or overlapping interval
        matching_sess = None
        overlap_s = 0.0
        pre_lead_available_s = 0.0
        onset_covered = False

        for _, srow in sub_sess.iterrows():
            z0, z1 = srow["start_ts"], srow["end_ts"]
            ov = max(0.0, min(te, z1) - max(ts, z0))
            if ov > overlap_s:
                overlap_s = ov
                matching_sess = srow["session"]
            if z0 <= ts <= z1:
                onset_covered = True
                pre_lead = ts - z0
                if pre_lead > pre_lead_available_s:
                    pre_lead_available_s = pre_lead
                    matching_sess = srow["session"]

        # Eligibility for P0 (needs >= 180s history before t_start), P1 (needs >= 240s), P2 (needs >= 360s)
        p0_eligible = bool(onset_covered and pre_lead_available_s >= 180.0 and ev["is_confirmed_stress"])
        p1_eligible = bool(onset_covered and pre_lead_available_s >= 240.0 and ev["is_confirmed_stress"])
        p2_eligible = bool(onset_covered and pre_lead_available_s >= 360.0 and ev["is_confirmed_stress"])

        if not overlap_s and not onset_covered:
            exclusion_reason = "no_sensor_recording_overlap"
        elif not onset_covered:
            exclusion_reason = "event_onset_outside_session_boundary"
        elif pre_lead_available_s < 180.0:
            exclusion_reason = "insufficient_pre_onset_history_lt_180s"
        elif ev["is_unconfirmed_na"]:
            exclusion_reason = "unconfirmed_na_algorithm_candidate_censored"
        elif ev["label_category"] == "survey_confirmed_no_stress":
            exclusion_reason = "survey_confirmed_stress_level_0_non_stress"
        else:
            exclusion_reason = "eligible_confirmed_stress_event"

        event_audit_rows.append(
            {
                "event_id": ev["event_id"],
                "excel_row_index": int(ev["excel_row_index"]),
                "subject": sid,
                "matched_session": matching_sess if matching_sess else "NONE",
                "dt_start_local": str(ev["dt_start_local"]),
                "dt_end_local": str(ev["dt_end_local"]),
                "event_start_ts": float(ts),
                "event_end_ts": float(te),
                "duration_min": float(ev["duration_min"]),
                "stress_level_raw": ev["stress_level_raw"],
                "label_category": ev["label_category"],
                "time_resolution_s": int(ev["time_resolution_s"]),
                "onset_uncertainty_note": "Minute-rounded start time (+-60s) from end-of-shift survey / algorithm boundary",
                "overlap_sensor_s": float(overlap_s),
                "onset_covered": bool(onset_covered),
                "pre_onset_history_s": float(pre_lead_available_s),
                "p0_forecast_eligible": p0_eligible,
                "p1_forecast_eligible": p1_eligible,
                "p2_forecast_eligible": p2_eligible,
                "audit_status": exclusion_reason,
                "covid_related": str(ev.get("COVID related", "na")),
                "patient_in_crisis": str(ev.get("Patient in Crisis", "na")),
                "increased_workload": str(ev.get("Increased Workload", "na")),
                "has_description": bool(str(ev.get("Description", "na")).strip().lower() not in ("na", "none", "")),
            }
        )

    event_audit_df = pd.DataFrame(event_audit_rows)

    # Mark sessions that overlap at least one survey row
    active_sessions = set(event_audit_df[event_audit_df["matched_session"] != "NONE"]["matched_session"].unique())
    sess_df["has_survey_annotation"] = sess_df["session"].isin(active_sessions)

    # Build subject summary table (Table 1)
    subj_summary_rows = []
    for sid, sg in sess_df.groupby("subject", sort=True):
        eg = event_audit_df[event_audit_df["subject"] == sid]
        total_hours = float(sg["dur_common_s"].sum() / 3600.0)
        annot_hours = float(sg[sg["has_survey_annotation"]]["dur_common_s"].sum() / 3600.0)
        unannot_hours = total_hours - annot_hours

        n_total_ev = len(eg)
        n_conf_stress = int((eg["stress_level_raw"].isin(["1", "1.0", "2", "2.0"])).sum())
        n_high_stress = int((eg["stress_level_raw"].isin(["2", "2.0"])).sum())
        n_mod_stress = int((eg["stress_level_raw"].isin(["1", "1.0"])).sum())
        n_no_stress_0 = int((eg["stress_level_raw"].isin(["0", "0.0"])).sum())
        n_na = int((eg["stress_level_raw"] == "na").sum())

        n_p0_elig = int(eg["p0_forecast_eligible"].sum())
        n_p1_elig = int(eg["p1_forecast_eligible"].sum())
        n_p2_elig = int(eg["p2_forecast_eligible"].sum())
        n_excluded_ev = n_total_ev - n_p1_elig

        min_dt = datetime.fromtimestamp(sg["start_ts"].min(), tz=ZoneInfo(SURVEY_TIMEZONE)).strftime("%Y-%m-%d")
        max_dt = datetime.fromtimestamp(sg["end_ts"].max(), tz=ZoneInfo(SURVEY_TIMEZONE)).strftime("%Y-%m-%d")

        subj_summary_rows.append(
            {
                "subject": sid,
                "n_zip_sessions": len(sg),
                "n_annotated_sessions": int(sg["has_survey_annotation"].sum()),
                "date_start": min_dt,
                "date_end": max_dt,
                "total_recorded_hours": round(total_hours, 2),
                "annotated_session_hours": round(annot_hours, 2),
                "unannotated_session_hours": round(unannot_hours, 2),
                "n_survey_rows_total": n_total_ev,
                "n_confirmed_stress_1_2": n_conf_stress,
                "n_high_stress_2": n_high_stress,
                "n_moderate_stress_1": n_mod_stress,
                "n_confirmed_non_stress_0": n_no_stress_0,
                "n_unconfirmed_na": n_na,
                "n_p0_eligible_events": n_p0_elig,
                "n_p1_eligible_events": n_p1_elig,
                "n_p2_eligible_events": n_p2_elig,
                "n_excluded_events_p1": n_excluded_ev,
                "n_e4_button_tags": int(sg["n_tags"].sum()),
            }
        )

    subject_summary_df = pd.DataFrame(subj_summary_rows)

    survey_sha256 = hashlib.sha256(survey_path.read_bytes()).hexdigest()
    manifest = {
        "dataset_name": "Hospital Nurses Stress Dataset (Hosseini et al., Scientific Data 2022)",
        "doi": "10.1038/s41597-022-01361-y",
        "dryad_doi": "10.5061/dryad.5hqbzkh6f",
        "license": "CC0 1.0 Universal (Dryad Public Domain Dedication) / Scientific Data Open Access (CC BY 4.0)",
        "root_path": str(root),
        "survey_file": str(survey_path),
        "survey_sha256": survey_sha256,
        "timezone_used": SURVEY_TIMEZONE,
        "n_subjects": int(sess_df["subject"].nunique()),
        "subjects": sorted(sess_df["subject"].unique().tolist()),
        "n_total_files": int(len(sess_df) + 1),
        "n_zip_archives": int(len(sess_df)),
        "n_annotated_zip_archives": int(sess_df["has_survey_annotation"].sum()),
        "n_unannotated_zip_archives": int((~sess_df["has_survey_annotation"]).sum()),
        "total_recorded_hours": float(round(sess_df["dur_common_s"].sum() / 3600.0, 4)),
        "annotated_session_hours": float(
            round(sess_df[sess_df["has_survey_annotation"]]["dur_common_s"].sum() / 3600.0, 4)
        ),
        "unannotated_session_hours": float(
            round(sess_df[~sess_df["has_survey_annotation"]]["dur_common_s"].sum() / 3600.0, 4)
        ),
        "sensor_modalities": {
            "EDA": {"present_in_zips": int(sess_df["has_eda"].sum()), "sampling_rate_hz": 4.0, "unit": "microSiemens (uS)"},
            "BVP": {"present_in_zips": int(sess_df["has_bvp"].sum()), "sampling_rate_hz": 64.0, "unit": "PPG raw unit"},
            "ACC": {"present_in_zips": int(sess_df["has_acc"].sum()), "sampling_rate_hz": 32.0, "unit": "1/64g integer"},
            "HR": {"present_in_zips": int(sess_df["has_hr"].sum()), "sampling_rate_hz": 1.0, "unit": "bpm (E4 derived)"},
            "IBI": {"present_in_zips": int(sess_df["has_ibi"].sum()), "sampling_rate_hz": "event-driven", "unit": "seconds"},
            "TEMP": {"present_in_zips": int(sess_df["has_temp"].sum()), "sampling_rate_hz": 4.0, "unit": "Celsius (excluded from primary model)"},
        },
        "synchronization_audit": {
            "max_start_time_spread_s": float(sess_df["start_spread_s"].max()),
            "max_duration_spread_s": float(sess_df["dur_spread_s"].max()),
            "n_overlapping_sessions": int(sess_df["is_overlapping_prev"].sum()),
        },
        "survey_event_counts": {
            "total_valid_rows": int(len(event_audit_df)),
            "confirmed_high_stress_2": int((event_audit_df["stress_level_raw"].isin(["2", "2.0"])).sum()),
            "confirmed_moderate_stress_1": int((event_audit_df["stress_level_raw"].isin(["1", "1.0"])).sum()),
            "confirmed_stress_total_1_2": int((event_audit_df["stress_level_raw"].isin(["1", "1.0", "2", "2.0"])).sum()),
            "survey_confirmed_no_stress_0": int((event_audit_df["stress_level_raw"].isin(["0", "0.0"])).sum()),
            "unconfirmed_algorithm_candidate_na": int((event_audit_df["stress_level_raw"] == "na").sum()),
            "p0_forecast_eligible_events": int(event_audit_df["p0_forecast_eligible"].sum()),
            "p1_forecast_eligible_events": int(event_audit_df["p1_forecast_eligible"].sum()),
            "p2_forecast_eligible_events": int(event_audit_df["p2_forecast_eligible"].sum()),
        },
        "label_reliability_findings": {
            "time_resolution_s": 60,
            "is_second_level_independent_video_onset": False,
            "generation_mechanism": (
                "Candidate stress intervals were generated by an upstream EDA/physiological algorithm or end-of-shift "
                "retrospective survey recall, then rated by nurses (0=no stress, 1=moderate, 2=high, na=unconfirmed)."
            ),
            "scientific_implication": (
                "Because start times are rounded to the minute (+-60s) and influenced by upstream physiological detection "
                "boundaries, P0/P1/P2 forecasting evaluates prediction of survey-validated physiological/perceived stress "
                "episode boundaries rather than independent video-coded behavioral meltdown onsets."
            ),
        },
        "sessions": sess_df.to_dict("records"),
    }

    return manifest, subject_summary_df, event_audit_df
