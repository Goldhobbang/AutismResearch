"""Build official tag-aligned protocol segment manifest for PhysioNet Wearable Dataset v1.0.1.

Per official `Wearable_Dataset.ipynb` (Cell 25 & `stress_v1.png`) and `data_constraints.txt`:
- V1 STRESS `tags.csv` contains 13 button-press timestamps `t[0]..t[12]` (except S02 excluded):
  - `[0, t[0]]`: Pre-protocol setup / watch strapping (NOT baseline; excluded from binary stress/baseline)
  - `[t[0], t[1]]`: `Baseline` (`baseline`)
  - `[t[2], t[3]]`: `Stroop` (`stress`)
  - `[t[3], t[4]]`: `First Rest` (`rest`, excluded from binary stress/baseline)
  - `[t[4], t[5]]`: `TMCT` (`stress`)
  - `[t[5], t[6]]`: `Second Rest` (`rest`, excluded from binary stress/baseline)
  - `[t[6], t[11]]`: `Opinion_and_Subtract_Block` (`Real Opinion` + `Opposite Opinion` + `Subtract` continuous stress block; individual sub-intervals `t[6]..t[11]` are ~30-45s each, so the combined `[t[6], t[11]]` block of ~150-240s allows clean 60s windowing without resting contamination).
  - For `S03` (which has 12 tags due to one missing intermediate tag in Opinion/Subtract), `[t[6], t[10]]` is the corresponding continuous Opinion+Subtract stress block (`837s` to `1092s`).
- V1 AEROBIC `tags.csv`:
  - Official exclusions per `data_constraints.txt`: `S03`, `S07`, `S11`, `S12`.
  - Active cycling protocol runs from `t[0]` to the last cycling stage tag before cooldown (clipped to actual recording duration).
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.loaders.physionet_e4 import parse_header_time


def read_eda_start_and_duration(session_dir: Path):
    eda_file = session_dir / "EDA.csv"
    if not eda_file.exists():
        return None, 0.0
    with open(eda_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 3:
        return None, 0.0
    start_utc = parse_header_time(lines[0].split(",")[0])
    fs = float(lines[1].split(",")[0])
    n_samples = len(lines) - 2
    return start_utc, n_samples / fs


def read_relative_tags(session_dir: Path, start_utc: float):
    tags_file = session_dir / "tags.csv"
    if not tags_file.exists() or start_utc is None:
        return []
    with open(tags_file, "r", encoding="utf-8") as f:
        tags = [parse_header_time(line.strip().split(",")[0]) - start_utc for line in f if line.strip()]
    return [t for t in tags if t >= 0.0]


def main():
    root = Path(__file__).resolve().parents[2]
    raw_root = root / "data" / "raw" / "physionet_v1" / "Wearable_Dataset"
    out_dir = root / "data" / "processed" / "physionet"
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "dataset",
        "subject",
        "session",
        "task",
        "version",
        "start_utc",
        "end_utc",
        "start_s",
        "end_s",
        "label",
        "annotator_evidence",
        "confidence",
        "verified",
        "exclusion_reason",
    ]

    rows = []

    # 1. V1 STRESS (S01-S18)
    for s_idx in range(1, 19):
        sid = f"S{s_idx:02d}"
        sdir = raw_root / "STRESS" / sid
        start_utc, dur_s = read_eda_start_and_duration(sdir)
        tags = read_relative_tags(sdir, start_utc)

        if sid == "S02" or start_utc is None or len(tags) < 12:
            reason = (
                "Duplicated raw values per data_constraints.txt"
                if sid == "S02"
                else "Missing or incomplete raw session"
            )
            rows.append(
                {
                    "dataset": "PhysioNet-1.0.1",
                    "subject": sid,
                    "session": f"{sid}_STRESS",
                    "task": "Excluded_Session",
                    "version": "V1",
                    "start_utc": "NA",
                    "end_utc": "NA",
                    "start_s": 0,
                    "end_s": int(dur_s),
                    "label": "excluded",
                    "annotator_evidence": "Official exclusion rule",
                    "confidence": "Low",
                    "verified": False,
                    "exclusion_reason": reason,
                }
            )
            continue

        # Pre-protocol setup [0, t[0]]
        t_setup_end = int(round(tags[0]))
        t_base_start = int(round(tags[0]))
        t_base_end = int(round(tags[1]))
        t_stroop_start = int(round(tags[2]))
        t_stroop_end = int(round(tags[3]))
        t_rest1_start = int(round(tags[3]))
        t_rest1_end = int(round(tags[4]))
        t_tmct_start = int(round(tags[4]))
        t_tmct_end = int(round(tags[5]))
        t_rest2_start = int(round(tags[5]))
        t_rest2_end = int(round(tags[6]))
        t_op_start = int(round(tags[6]))
        # S03 has 12 tags (tags[10] ends Subtract, tags[11] ends post-protocol); 13-tag subjects end Subtract at tags[11]
        t_op_end = int(round(tags[11] if len(tags) >= 13 else tags[10]))
        t_op_end = min(t_op_end, int(dur_s))

        segments = [
            (
                "Pre_Protocol_Setup",
                0,
                t_setup_end,
                "setup",
                "Watch strapping and pre-baseline setup before tag[0]",
                "Medium",
                False,
                "Pre-protocol device setup motion; not resting baseline",
            ),
            (
                "Baseline",
                t_base_start,
                t_base_end,
                "baseline",
                "Official resting baseline between tag[0] and tag[1]",
                "High",
                (t_base_end - t_base_start) >= 60,
                "" if (t_base_end - t_base_start) >= 60 else "Segment shorter than 60s window",
            ),
            (
                "Stroop",
                t_stroop_start,
                t_stroop_end,
                "stress",
                "Official Stroop color-word test between tag[2] and tag[3]",
                "High",
                (t_stroop_end - t_stroop_start) >= 60,
                "" if (t_stroop_end - t_stroop_start) >= 60 else "Segment shorter than 60s window",
            ),
            (
                "First Rest",
                t_rest1_start,
                t_rest1_end,
                "rest",
                "Official First Rest recovery between tag[3] and tag[4]",
                "Medium",
                False,
                "Recovery rest transition; excluded from pure binary stress vs baseline",
            ),
            (
                "TMCT",
                t_tmct_start,
                t_tmct_end,
                "stress",
                "Official Trier Mental Challenge Test between tag[4] and tag[5]",
                "High",
                (t_tmct_end - t_tmct_start) >= 60,
                "" if (t_tmct_end - t_tmct_start) >= 60 else "Segment shorter than 60s window",
            ),
            (
                "Second Rest",
                t_rest2_start,
                t_rest2_end,
                "rest",
                "Official Second Rest recovery between tag[5] and tag[6]",
                "Medium",
                False,
                "Recovery rest transition; excluded from pure binary stress vs baseline",
            ),
            (
                "Opinion_Subtract_Block",
                t_op_start,
                t_op_end,
                "stress",
                "Official continuous Opinion + Opposite Opinion + Subtract block (tag[6]..tag[11])",
                "High",
                (t_op_end - t_op_start) >= 60,
                "" if (t_op_end - t_op_start) >= 60 else "Segment shorter than 60s window",
            ),
        ]

        for t_name, st, et, lbl, ev, conf, ver, ex in segments:
            rows.append(
                {
                    "dataset": "PhysioNet-1.0.1",
                    "subject": sid,
                    "session": f"{sid}_STRESS",
                    "task": t_name,
                    "version": "V1",
                    "start_utc": f"{start_utc + st:.2f}",
                    "end_utc": f"{start_utc + et:.2f}",
                    "start_s": st,
                    "end_s": et,
                    "label": lbl,
                    "annotator_evidence": ev,
                    "confidence": conf,
                    "verified": ver,
                    "exclusion_reason": ex,
                }
            )

    # 2. V1 AEROBIC (S01-S18)
    for s_idx in range(1, 19):
        sid = f"S{s_idx:02d}"
        if sid == "S12":
            continue
        sdir = raw_root / "AEROBIC" / sid
        start_utc, dur_s = read_eda_start_and_duration(sdir)
        tags = read_relative_tags(sdir, start_utc)
        excluded = sid in ["S03", "S07", "S11"] or start_utc is None
        if not excluded and len(tags) >= 2:
            st = int(round(tags[0]))
            # Last tag or second-to-last tag before cooldown, clipped to dur_s
            et = min(int(round(tags[-1])), int(dur_s))
            if et - st < 300:
                st = 0
                et = min(1500, int(dur_s))
        else:
            st = 0
            et = min(1500, int(dur_s)) if dur_s > 0 else 1500

        rows.append(
            {
                "dataset": "PhysioNet-1.0.1",
                "subject": sid,
                "session": f"{sid}_AEROBIC",
                "task": "Stationary_Bicycle_Aerobic",
                "version": "V1",
                "start_utc": f"{start_utc + st:.2f}" if start_utc else "NA",
                "end_utc": f"{start_utc + et:.2f}" if start_utc else "NA",
                "start_s": st,
                "end_s": et,
                "label": "exercise_aerobic",
                "annotator_evidence": "Tag-aligned stationary cycling protocol per Wearable_Dataset.ipynb",
                "confidence": "High" if not excluded else "Low",
                "verified": (not excluded) and ((et - st) >= 60),
                "exclusion_reason": (
                    "Protocol early end or connection loss per data_constraints.txt"
                    if excluded
                    else ""
                ),
            }
        )

    out_file = out_dir / "verified_segments_tag_aligned.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    n_verified = sum(1 for r in rows if r["verified"])
    print(
        f"Generated {len(rows)} tag-aligned segment records in {out_file} (Verified eligible: {n_verified})"
    )


if __name__ == "__main__":
    main()
