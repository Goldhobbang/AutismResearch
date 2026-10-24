"""Study C: AMDC Environment-Physiology Predictive Modeling (Regression).

Formulation:
- Input: past 60s [t-59, t] at 1 Hz.
- Target: future 30s mean [t+1, t+30] of Heart Rate and Skin Conductance.
- Zero future leakage: strictly separated input and target horizons.
- Evaluator: Participant-level LOSO (14 folds).
- Models: Persistence baseline, Physio-only, +Lux, +Sound, +Lux+Sound, +Covariates, Full, Env-only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]


def bootstrap_ci_diff(diffs: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return (float(arr[0]) if len(arr) else np.nan, float(arr[0]) if len(arr) else np.nan)
    rng = np.random.default_rng(seed)
    boots = np.mean(rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def extract_window_features(series: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(series, dtype=float)
    ok = np.isfinite(x)
    if np.sum(ok) < 4:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_slope": np.nan,
            f"{prefix}_last10_mean": np.nan,
        }
    z = x[ok]
    t = np.arange(len(x))[ok]
    tc = t - np.mean(t)
    denom = max(float(np.dot(tc, tc)), 1e-12)
    slope = float(np.dot(tc, z - np.mean(z)) / denom)
    last10 = z[-min(10, len(z)) :]
    return {
        f"{prefix}_mean": float(np.mean(z)),
        f"{prefix}_std": float(np.std(z)),
        f"{prefix}_min": float(np.min(z)),
        f"{prefix}_max": float(np.max(z)),
        f"{prefix}_slope": slope,
        f"{prefix}_last10_mean": float(np.mean(last10)),
    }


def build_amdc_window_dataset(
    clean_df: pd.DataFrame,
    window_s: int = 60,
    forecast_s: int = 30,
    step_s: int = 15,
) -> pd.DataFrame:
    """Extract past 60s features and future 30s target averages per participant."""
    rows = []
    for p_id, pdf in clean_df.groupby("participant"):
        n = len(pdf)
        # Slices: input [i - window_s, i), target [i, i + forecast_s)
        for i in range(window_s, n - forecast_s + 1, step_s):
            in_slice = pdf.iloc[i - window_s : i]
            tar_slice = pdf.iloc[i : i + forecast_s]

            # Check validity
            hr_in = in_slice["heart_rate_bpm"].to_numpy()
            cond_in = in_slice["skin_conductance_uS"].to_numpy()
            hr_tar = tar_slice["heart_rate_bpm"].to_numpy()
            cond_tar = tar_slice["skin_conductance_uS"].to_numpy()

            if np.isnan(hr_in).mean() > 0.20 or np.isnan(hr_tar).mean() > 0.20:
                continue
            if np.isnan(cond_in).mean() > 0.20 or np.isnan(cond_tar).mean() > 0.20:
                continue

            row = {
                "participant": p_id,
                "input_start_s": int(in_slice["second_idx"].min()),
                "input_end_s": int(in_slice["second_idx"].max()),
                "target_start_s": int(tar_slice["second_idx"].min()),
                "target_end_s": int(tar_slice["second_idx"].max()),
                # Prediction Targets: future 30s mean
                "target_hr_mean": float(np.nanmean(hr_tar)),
                "target_cond_mean": float(np.nanmean(cond_tar)),
                # Persistence baseline predictors: last 10s of input
                "persist_hr_pred": float(np.nanmean(hr_in[-10:])),
                "persist_cond_pred": float(np.nanmean(cond_in[-10:])),
            }

            # Input Features: past 60s
            row.update(extract_window_features(hr_in, "hr"))
            row.update(extract_window_features(cond_in, "cond"))
            row.update(extract_window_features(in_slice["lux"].to_numpy(), "lux"))
            row.update(extract_window_features(in_slice["sound_db"].to_numpy(), "sound"))
            row.update(extract_window_features(in_slice["temperature_c"].to_numpy(), "temp"))
            row.update(extract_window_features(in_slice["humidity_pct"].to_numpy(), "humid"))
            row.update(extract_window_features(in_slice["pressure_hpa"].to_numpy(), "press"))
            row.update(extract_window_features(in_slice["iaq"].to_numpy(), "iaq"))

            rows.append(row)

    out = pd.DataFrame(rows)
    return out


def run_predictive_evaluation(
    wdf: pd.DataFrame,
    target_var: str = "hr",  # 'hr' or 'cond'
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run LOSO evaluation across candidate model feature sets."""
    target_col = f"target_{target_var}_mean"
    persist_col = f"persist_{target_var}_pred"

    # Define feature sets
    feat_physio = [c for c in wdf.columns if (c.startswith("hr_") or c.startswith("cond_")) and not c.endswith("last10_mean")]
    feat_lux = [c for c in wdf.columns if c.startswith("lux_") and not c.endswith("last10_mean")]
    feat_sound = [c for c in wdf.columns if c.startswith("sound_") and not c.endswith("last10_mean")]
    feat_covar = [c for c in wdf.columns if any(c.startswith(p) for p in ["temp_", "humid_", "press_", "iaq_"]) and not c.endswith("last10_mean")]

    model_configs = {
        "Model_A_Persistence": [],
        "Model_B_Physio_Only": feat_physio,
        "Model_C_Physio_Plus_Lux": feat_physio + feat_lux,
        "Model_D_Physio_Plus_Sound": feat_physio + feat_sound,
        "Model_E_Physio_Lux_Sound": feat_physio + feat_lux + feat_sound,
        "Model_F_Physio_Env_Covariates": feat_physio + feat_covar,
        "Model_G_Full_Multimodal": feat_physio + feat_covar + feat_lux + feat_sound,
        "Model_H_Env_Only": feat_covar + feat_lux + feat_sound,
    }

    participants = sorted(wdf["participant"].unique(), key=lambda x: int(x[1:]))
    per_subj_rows = []
    summary_rows = []

    for reg_type in ["Ridge", "HGB"]:
        for m_name, fcols in model_configs.items():
            subj_maes = []
            subj_rmses = []
            subj_r2s = []
            persist_maes = []
            diff_maes = []

            for p_test in participants:
                tr = wdf[wdf["participant"] != p_test].copy()
                te = wdf[wdf["participant"] == p_test].copy()

                y_tr = tr[target_col].to_numpy()
                y_te = te[target_col].to_numpy()
                p_persist = te[persist_col].to_numpy()
                mae_persist = mean_absolute_error(y_te, p_persist)

                if m_name == "Model_A_Persistence":
                    p_pred = p_persist
                else:
                    if reg_type == "Ridge":
                        pipe = Pipeline([
                            ("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler()),
                            ("reg", Ridge(alpha=10.0, random_state=42)),
                        ])
                    else:
                        pipe = Pipeline([
                            ("imp", SimpleImputer(strategy="median")),
                            ("reg", HistGradientBoostingRegressor(max_iter=60, max_depth=4, min_samples_leaf=15, learning_rate=0.05, random_state=42)),
                        ])

                    pipe.fit(tr[fcols], y_tr)
                    p_pred = pipe.predict(te[fcols])

                mae = mean_absolute_error(y_te, p_pred)
                rmse = np.sqrt(mean_squared_error(y_te, p_pred))
                r2 = r2_score(y_te, p_pred)

                diff_mae = mae_persist - mae  # positive means improvement over persistence
                rel_imp_pct = (diff_mae / max(mae_persist, 1e-6)) * 100.0

                res = {
                    "target_var": target_var,
                    "regressor": reg_type,
                    "model": m_name,
                    "participant": p_test,
                    "n_eval_windows": len(te),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "mae_persist": mae_persist,
                    "diff_mae_vs_persist": diff_mae,
                    "rel_improvement_pct": rel_imp_pct,
                }
                per_subj_rows.append(res)
                subj_maes.append(mae)
                subj_rmses.append(rmse)
                subj_r2s.append(r2)
                persist_maes.append(mae_persist)
                diff_maes.append(diff_mae)

            ci_low, ci_high = bootstrap_ci_diff(np.array(diff_maes))

            summary_rows.append({
                "target_var": target_var,
                "regressor": reg_type,
                "model": m_name,
                "n_features": len(fcols),
                "n_participants": len(participants),
                "mean_mae": float(np.mean(subj_maes)),
                "median_mae": float(np.median(subj_maes)),
                "mean_rmse": float(np.mean(subj_rmses)),
                "mean_r2": float(np.mean(subj_r2s)),
                "mean_mae_persist": float(np.mean(persist_maes)),
                "mean_diff_mae_vs_persist": float(np.mean(diff_maes)),
                "diff_mae_ci_low": ci_low,
                "diff_mae_ci_high": ci_high,
                "mean_rel_improvement_pct": float(np.mean(diff_maes) / max(np.mean(persist_maes), 1e-6)) * 100.0,
            })

    return pd.DataFrame(per_subj_rows), pd.DataFrame(summary_rows)


def main():
    run_id = f"run_{int(time.time())}"
    out_dir = ROOT / "results" / "amdc" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting Study C: AMDC Predictive Modeling (run_id: {run_id})...")

    # Load clean data
    clean_path = ROOT / "data/processed/amdc/amdc_all_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean AMDC data not found at {clean_path}")

    clean_df = pd.read_csv(clean_path)
    print(f"Loaded clean AMDC data: {clean_df.shape}")

    # Build window dataset
    cache_win_path = ROOT / "data/processed/amdc/amdc_window_dataset.csv"
    if cache_win_path.exists():
        print("Loading cached AMDC window dataset...")
        wdf = pd.read_csv(cache_win_path)
    else:
        print("Extracting 60s input -> 30s target windows (step=15s)...")
        wdf = build_amdc_window_dataset(clean_df)
        wdf.to_csv(cache_win_path, index=False)

    print(f"Window dataset shape: {wdf.shape} across {wdf['participant'].nunique()} participants")

    # 1. Heart Rate prediction
    print("\n--- Evaluating Heart Rate (HR) prediction across models (LOSO) ---")
    hr_ps, hr_sum = run_predictive_evaluation(wdf, target_var="hr", out_dir=out_dir)

    # 2. Skin Conductance prediction
    print("\n--- Evaluating Skin Conductance prediction across models (LOSO) ---")
    cond_ps, cond_sum = run_predictive_evaluation(wdf, target_var="cond", out_dir=out_dir)

    all_ps = pd.concat([hr_ps, cond_ps], ignore_index=True)
    all_sum = pd.concat([hr_sum, cond_sum], ignore_index=True)

    all_ps.to_csv(out_dir / "amdc_predictive_per_subject.csv", index=False)
    all_sum.to_csv(out_dir / "amdc_predictive_summary.csv", index=False)

    # Latest aliases
    latest_dir = ROOT / "results" / "amdc" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    all_ps.to_csv(latest_dir / "amdc_predictive_per_subject.csv", index=False)
    all_sum.to_csv(latest_dir / "amdc_predictive_summary.csv", index=False)

    print("\n=== STUDY C: PREDICTIVE MODELING SUMMARY ===")
    print(all_sum[["target_var", "regressor", "model", "n_features", "mean_mae", "mean_mae_persist", "mean_diff_mae_vs_persist", "mean_rel_improvement_pct", "mean_r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
