"""Multi-subject WESAD -> PhysioNet zero-shot audit.

Uses only already available verified raw PhysioNet files and frozen WESAD
fold models. Missing participants are reported, never silently imputed.
Run on research laptop:
  py experiments/phase3/physionet_multi_subject_audit.py
"""
import hashlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.phase2.zero_shot_physionet import (
    extract_windows_for_session, evaluate_models_on_windows, RAW_DIR, MODELS_DIR
)

SEGMENTS = ROOT / "data/processed/physionet/verified_segments.csv"
HASHES = ROOT / "data/registry/SHA256SUMS.txt"
OUT = ROOT / "results/phase3_strict/physionet_external"
CANDIDATE_MODELS = ("lr_acc", "lr_eda", "lr_eda_acc", "lr_all", "lr_ppg", "rf_all")


def official_hashes(path=HASHES):
    hashes = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            hashes[parts[1].strip().lstrip("*")] = parts[0].lower()
    return hashes


def verified_raw(subject, session, hashes):
    folder = RAW_DIR / session / subject
    required = ["EDA.csv", "ACC.csv"]
    if session == "STRESS":
        required += ["BVP.csv", "tags.csv"]
    for name in required:
        p = folder / name
        key = f"Wearable_Dataset/{session}/{subject}/{name}"
        if not p.is_file():
            return False, f"missing:{name}"
        expected = hashes.get(key)
        if not expected:
            return False, f"checksum_not_registered:{name}"
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != expected:
            return False, f"sha256_mismatch:{name}"
    # Optional BVP in aerobic is usable only if hash matches.
    optional = folder / "BVP.csv"
    key = f"Wearable_Dataset/{session}/{subject}/BVP.csv"
    if session == "AEROBIC" and optional.exists():
        if key not in hashes or hashlib.sha256(optional.read_bytes()).hexdigest() != hashes[key]:
            return False, "aerobic_BVP_checksum_failure"
    return True, "verified"


def select_verified_session_segments(group, subject, session):
    """Manifest uses subject-prefixed names, e.g. S01_STRESS."""
    return group.loc[group["session"].eq(f"{subject}_{session}")].copy()


def compute_subject_metrics(subject, frame):
    records = []
    stress = frame[frame.session.eq("STRESS") & frame.label.isin(["stress", "baseline"])]
    exercise = frame[frame.session.eq("AEROBIC")]
    for m in CANDIDATE_MODELS:
        col = f"prob_{m}"
        if col not in frame:
            continue
        available_stress = stress.loc[stress[col].notna()]
        if len(available_stress) and available_stress.label.nunique() == 2:
            y = available_stress.label.eq("stress").to_numpy(dtype=int)
            p = available_stress[col].to_numpy(dtype=float)
            if not np.isfinite(p).all():
                raise ValueError(f"Nonfinite probabilities {subject} {m}")
            records.append({
                "subject": subject, "model": m, "test": "stress_vs_baseline",
                "n_windows": len(y), "n_positive": int(y.sum()),
                "auroc": float(roc_auc_score(y, p)),
                "ap": float(average_precision_score(y, p)),
                "positive_fraction": float(np.mean(p >= .5)),
                "stress_recall": float(np.mean((p >= .5)[y == 1])),
                "baseline_fpr": float(np.mean((p >= .5)[y == 0])),
                "mean_probability": float(p.mean())
            })
        available_exercise = exercise.loc[exercise[col].notna()]
        if len(available_exercise):
            p = available_exercise[col].to_numpy(dtype=float)
            if not np.isfinite(p).all():
                raise ValueError(f"Nonfinite exercise probabilities {subject} {m}")
            records.append({
                "subject": subject, "model": m, "test": "aerobic_positive_rate",
                "n_windows": len(p), "n_positive": 0,
                "auroc": np.nan, "ap": np.nan,
                "positive_fraction": float(np.mean(p >= .5)),
                "stress_recall": np.nan, "baseline_fpr": np.nan,
                "mean_probability": float(p.mean())
            })
    return records


def main():
    if not SEGMENTS.exists() or not HASHES.exists():
        raise FileNotFoundError("Verified PhysioNet segment manifest / SHA registry missing")
    if not MODELS_DIR.exists():
        raise FileNotFoundError("Frozen WESAD fold models missing; copy local private weights")
    # Make all included model families complete or fail; avoids silent 1/15 ensemble.
    for m in ("lr_acc", "lr_eda", "lr_eda_acc"):
        absent = [f"{m}_fold{i}.joblib" for i in range(15)
                  if not (MODELS_DIR / f"{m}_fold{i}.joblib").is_file()]
        if absent:
            raise FileNotFoundError(f"Incomplete frozen ensemble for {m}: {absent}")
    segments = pd.read_csv(SEGMENTS)
    if "verified" not in segments:
        raise ValueError("verified label missing")
    segments = segments.loc[segments.verified.astype(str).str.lower().isin(["true", "1"])]
    hashes = official_hashes()
    all_preds, all_metrics, manifest = [], [], []
    for subject, group in segments.groupby("subject"):
        frames = []
        for session in ("STRESS", "AEROBIC"):
            # Manifest encodes subject-prefixed sessions, e.g. S01_STRESS.
            # Preserve subject grouping; do not match plain STRESS/AEROBIC.
            g = select_verified_session_segments(group, subject, session)
            if g.empty:
                continue
            good, reason = verified_raw(subject, session, hashes)
            manifest.append({"subject": subject, "session": session,
                             "status": "eligible" if good else "excluded",
                             "reason": reason})
            if not good:
                continue
            try:
                f = extract_windows_for_session(subject, session, g)
                if f is None or f.empty:
                    manifest.append({"subject": subject, "session": session,
                                     "status": "excluded", "reason": "no_valid_windows"})
                    continue
                frames.append(f)
            except Exception as exc:
                manifest.append({"subject": subject, "session": session,
                                 "status": "excluded", "reason": "extraction_error:" + str(exc)})
        if not frames:
            continue
        # Evaluate sessions independently so aerobic missing BVP does not
        # suppress BVP-inclusive evaluation in fully instrumented STRESS data.
        evaluated = pd.concat([evaluate_models_on_windows(f) for f in frames], ignore_index=True)
        all_preds.append(evaluated)
        all_metrics.extend(compute_subject_metrics(subject, evaluated))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(OUT / "coverage_manifest.csv", index=False)
    if not all_preds:
        print("No checksum-verified raw participants available. Coverage manifest written; no scores.")
        return
    pd.concat(all_preds, ignore_index=True).to_csv(OUT / "per_window_predictions.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(OUT / "per_subject_metrics.csv", index=False)
    print("Subjects evaluated:", sorted(set(row["subject"] for row in all_metrics)))
    print("Do not pool overlapping windows as independent participants.")
    print("An AUROC/AP for one participant is NOT evidence of population transfer.")

if __name__ == "__main__":
    main()
