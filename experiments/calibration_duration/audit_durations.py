"""Audit continuous baseline rest durations across WESAD and PhysioNet participants."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def audit():
    # 1. WESAD
    wesad_feat = pd.read_csv(ROOT / "data/processed/wesad_features.csv")
    w_base = wesad_feat[wesad_feat["phase"] == 1]
    print("=== WESAD Phase 1 (Baseline) duration per subject ===")
    wesad_summary = []
    for sid, sdf in w_base.groupby("subject"):
        dur = sdf["end_s"].max() - sdf["start_s"].min()
        n_win = len(sdf)
        wesad_summary.append({"subject": sid, "duration_s": dur, "n_windows": n_win})
        print(f"{sid}: {dur:.1f}s ({n_win} windows, range {sdf['start_s'].min()} -> {sdf['end_s'].max()})")

    # 2. PhysioNet Tag-Aligned
    segs = pd.read_csv(ROOT / "data/processed/physionet/verified_segments_tag_aligned.csv")
    p_base = segs[(segs["label"] == "baseline") & (segs["verified"] == True)]
    print("\n=== PhysioNet Tag-Aligned Baseline duration per subject ===")
    pn_summary = []
    for _, r in p_base.iterrows():
        dur = r["end_s"] - r["start_s"]
        pn_summary.append({"subject": r["subject"], "start_s": r["start_s"], "end_s": r["end_s"], "duration_s": dur})
        print(f"{r['subject']}: {dur:.1f}s (start {r['start_s']} -> end {r['end_s']})")

    # 3. PhysioNet Legacy Segments
    segs_leg = pd.read_csv(ROOT / "data/processed/physionet/verified_segments.csv")
    p_base_leg = segs_leg[(segs_leg["label"] == "baseline") & (segs_leg["verified"] == True)]
    print("\n=== PhysioNet Legacy Baseline duration per subject ===")
    for _, r in p_base_leg.iterrows():
        dur = r["end_s"] - r["start_s"]
        print(f"{r['subject']}: {dur:.1f}s (start {r['start_s']} -> end {r['end_s']})")

if __name__ == "__main__":
    audit()
