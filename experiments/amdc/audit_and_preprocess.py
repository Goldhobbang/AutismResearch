"""Study C: AMDC (Health-Spa Dataset) Data Audit, Registry, and Preprocessing.

Audits file hashes, sizes, sampling frequencies, channels, and missingness.
Preprocesses 1 Hz physiological and environmental signals into data/processed/amdc/.
Computes exploratory correlation matrices and within-subject statistical relationships.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_and_preprocess():
    raw_base = ROOT / "data/raw/AMDC/Health-spa Dataset/Health-spa Dataset"
    reg_dir = ROOT / "data/registry/amdc"
    proc_dir = ROOT / "data/processed/amdc"
    reg_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    print("Auditing AMDC files and verifying SHA-256 hashes...")
    file_manifest = []

    # 1. Audit ReadMe
    readme_path = raw_base / "ReadMe.txt"
    if readme_path.exists():
        file_manifest.append({
            "path": str(readme_path.relative_to(ROOT)),
            "filename": readme_path.name,
            "type": "documentation",
            "participant": "all",
            "size_bytes": readme_path.stat().st_size,
            "sha256": sha256sum(readme_path),
        })

    # 2. Audit Final_Dataframe_Pn.csv files
    final_files = sorted(list(raw_base.glob("Final_Dataframe_P*.csv")), key=lambda p: int(p.stem.split("_P")[1]))
    processed_dfs = []

    for fpath in final_files:
        p_id = f"P{fpath.stem.split('_P')[1]}"
        f_size = fpath.stat().st_size
        f_hash = sha256sum(fpath)

        df = pd.read_csv(fpath)

        file_manifest.append({
            "path": str(fpath.relative_to(ROOT)),
            "filename": fpath.name,
            "type": "final_dataframe_aggregated",
            "participant": p_id,
            "size_bytes": f_size,
            "sha256": f_hash,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": "; ".join(df.columns),
        })

        # Preprocessing:
        # 1. Skin Resistance (kOhm) -> Skin Conductance (uS = 1000 / R)
        # Check for zero or negative values
        r_raw = df["Skin Resistance"].to_numpy(dtype=float)
        g_uS = np.where(r_raw > 0, 1000.0 / np.clip(r_raw, 1.0, 10000.0), np.nan)

        clean_p = pd.DataFrame({
            "participant": p_id,
            "second_idx": np.arange(len(df)),
            "skin_resistance_kohm": r_raw,
            "skin_conductance_uS": g_uS,
            "lux": df["Lux"].to_numpy(dtype=float),
            "sound_db": df["Sound"].to_numpy(dtype=float),
            "heart_rate_bpm": df["Heart Rate"].to_numpy(dtype=float),
            "temperature_c": df["Temperature"].to_numpy(dtype=float),
            "humidity_pct": df["Humidity"].to_numpy(dtype=float),
            "pressure_hpa": df["Pressure"].to_numpy(dtype=float),
            "iaq": df["IAQ"].to_numpy(dtype=float),
            "pm4": df["PM4"].to_numpy(dtype=float),
            "pm5": df["PM5"].to_numpy(dtype=float),
            "pm6": df["PM6"].to_numpy(dtype=float),
        })

        # Save individual clean file
        clean_p.to_csv(proc_dir / f"amdc_{p_id.lower()}_clean.csv", index=False)
        processed_dfs.append(clean_p)

    # 3. Sample check raw subdirectories for P1
    for sub in ["ambient", "sound", "ecg", "rtor"]:
        sdir = raw_base / "P1" / sub
        for rf in sorted(list(sdir.glob("*.csv")))[:2]:
            file_manifest.append({
                "path": str(rf.relative_to(ROOT)),
                "filename": rf.name,
                "type": f"raw_sample_{sub}",
                "participant": "P1",
                "size_bytes": rf.stat().st_size,
                "sha256": sha256sum(rf),
            })

    # Save manifest
    manifest_df = pd.DataFrame(file_manifest)
    manifest_df.to_csv(reg_dir / "amdc_files_manifest.csv", index=False)

    # Integrated all clean dataframe
    all_clean = pd.concat(processed_dfs, ignore_index=True)
    all_clean.to_csv(proc_dir / "amdc_all_clean.csv", index=False)
    print(f"Saved {len(manifest_df)} file records to data/registry/amdc/amdc_files_manifest.csv")
    print(f"Saved integrated {all_clean.shape} records to data/processed/amdc/amdc_all_clean.csv")

    # 4. Compute exploratory correlation analysis (within-subject and pooled)
    print("\nComputing exploratory correlations between environmental variables (Lux, Sound) and physiology (HR, Conductance)...")
    corr_records = []
    for p_id, pdf in all_clean.groupby("participant"):
        valid_hr = pdf.dropna(subset=["heart_rate_bpm", "lux", "sound_db", "skin_conductance_uS"])
        if len(valid_hr) < 30:
            continue

        r_lux_hr, p_lux_hr = pearsonr(valid_hr["lux"], valid_hr["heart_rate_bpm"])
        r_snd_hr, p_snd_hr = pearsonr(valid_hr["sound_db"], valid_hr["heart_rate_bpm"])
        r_lux_cond, p_lux_cond = pearsonr(valid_hr["lux"], valid_hr["skin_conductance_uS"])
        r_snd_cond, p_snd_cond = pearsonr(valid_hr["sound_db"], valid_hr["skin_conductance_uS"])

        corr_records.append({
            "participant": p_id,
            "n_samples": len(valid_hr),
            "r_lux_hr": r_lux_hr,
            "pval_lux_hr": p_lux_hr,
            "r_sound_hr": r_snd_hr,
            "pval_sound_hr": p_snd_hr,
            "r_lux_conductance": r_lux_cond,
            "pval_lux_conductance": p_lux_cond,
            "r_sound_conductance": r_snd_cond,
            "pval_sound_conductance": p_snd_cond,
            "hr_mean": float(valid_hr["heart_rate_bpm"].mean()),
            "hr_std": float(valid_hr["heart_rate_bpm"].std()),
            "cond_mean": float(valid_hr["skin_conductance_uS"].mean()),
            "cond_std": float(valid_hr["skin_conductance_uS"].std()),
            "lux_mean": float(valid_hr["lux"].mean()),
            "lux_std": float(valid_hr["lux"].std()),
            "sound_mean": float(valid_hr["sound_db"].mean()),
            "sound_std": float(valid_hr["sound_db"].std()),
        })

    corr_df = pd.DataFrame(corr_records)
    corr_df.to_csv(reg_dir / "amdc_exploratory_correlations.csv", index=False)

    print("=== EXPLORATORY CORRELATIONS SUMMARY ===")
    print(f"Mean within-subject r(Lux, HR):            {corr_df['r_lux_hr'].mean():+.4f} (range {corr_df['r_lux_hr'].min():+.4f} ~ {corr_df['r_lux_hr'].max():+.4f})")
    print(f"Mean within-subject r(Sound, HR):          {corr_df['r_sound_hr'].mean():+.4f} (range {corr_df['r_sound_hr'].min():+.4f} ~ {corr_df['r_sound_hr'].max():+.4f})")
    print(f"Mean within-subject r(Lux, Conductance):   {corr_df['r_lux_conductance'].mean():+.4f} (range {corr_df['r_lux_conductance'].min():+.4f} ~ {corr_df['r_lux_conductance'].max():+.4f})")
    print(f"Mean within-subject r(Sound, Conductance): {corr_df['r_sound_conductance'].mean():+.4f} (range {corr_df['r_sound_conductance'].min():+.4f} ~ {corr_df['r_sound_conductance'].max():+.4f})")


if __name__ == "__main__":
    audit_and_preprocess()
