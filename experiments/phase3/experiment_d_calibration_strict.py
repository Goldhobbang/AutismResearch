"""Strict WESAD per-subject calibration audit (no test-label calibration).

Does not overwrite Phase 3 results. Outputs exploratory Phase 3 strict results.
Run from repository root:
  py experiments/phase3/experiment_d_calibration_strict.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data/processed/wesad_features.csv"
BASELINE = ROOT / "results/baselines/wesad_20261024/predictions.csv"
OUT = ROOT / "results/phase3_strict/calibration"
CAL_SECONDS = 300.0
HOP_SECONDS = 15.0
WINDOW_SECONDS = 60.0
GUARD_COLD_PROB = 0.90  # Exploratory, PRE-SPECIFIED; not selected on test folds.
FEATURE_PREFIXES = ("eda_", "ppg_", "acc_")


def calibration_split(subject_frame, seconds=CAL_SECONDS):
    """Return eligible calibration rows and post-calibration evaluation rows.

    The calibration period is exactly 300 seconds of the first uninterrupted
    phase-1 baseline. 60s overlapping feature windows MUST end by its boundary.
    A 60s evaluation window MUST begin at/after the boundary.
    """
    df = subject_frame.sort_values(["start_s", "end_s"]).copy()
    baseline = df.loc[df["phase"].eq(1)].copy()
    if baseline.empty:
        raise ValueError("No fully labelled baseline windows")
    start = float(baseline["start_s"].min())
    stop = start + seconds
    calibration = baseline.loc[
        baseline["start_s"].ge(start - 1e-6)
        & baseline["end_s"].le(stop + 1e-6)
    ].copy()
    if calibration.empty or float(calibration["end_s"].max()) < stop - 1e-6:
        raise ValueError("Incomplete contiguous 300-second baseline")
    if abs(float(calibration["start_s"].min()) - start) > 1e-6:
        raise ValueError("Calibration does not start at baseline boundary")
    ends = np.sort(calibration["end_s"].to_numpy(dtype=float))
    # 60-second windows every 15s -> 17 windows spanning exactly 300s.
    if len(ends) != 17 or not np.allclose(np.diff(ends), HOP_SECONDS):
        raise ValueError("Calibration missing expected 17 uninterrupted windows")
    if not np.allclose(
        calibration["end_s"].to_numpy() - calibration["start_s"].to_numpy(),
        WINDOW_SECONDS,
    ):
        raise ValueError("Nonstandard window duration in calibration")
    if "qc_common_valid" in calibration:
        good = calibration["qc_common_valid"].astype(str).str.lower().isin(["true", "1", "1.0"])
        if not bool(good.all()):
            raise ValueError("Invalid sensor quality during calibration")
    eligible = df["phase"].isin([1, 2, 3])
    if "qc_common_valid" in df:
        eligible &= df["qc_common_valid"].astype(str).str.lower().isin(["true", "1", "1.0"])
    test = df.loc[eligible & df["start_s"].ge(stop - 1e-6)].copy()
    if test.empty:
        raise ValueError("No post-calibration evaluation windows")
    assert (calibration["end_s"].max() <= test["start_s"].min() + 1e-6)
    return calibration, test, dict(start_s=start, stop_s=stop, duration_s=seconds,
                                    calibration_windows=len(calibration),
                                    evaluation_windows=len(test))


def alert_counts(frame, predictions):
    """Separate window errors from episode rising edges; reset on time gaps."""
    x = frame.sort_values("start_s")
    y = x["phase"].eq(2).to_numpy(dtype=int)
    pred = np.asarray(predictions, dtype=int)
    times = x["start_s"].to_numpy(dtype=float)
    if len(y) != len(pred):
        raise ValueError("Predictions and evaluation rows must align")
    adjacent = np.r_[False, np.isclose(np.diff(times), HOP_SECONDS)]
    starts = (pred == 1) & (~np.r_[False, pred[:-1] == 1] | ~adjacent)
    false_episodes = int(np.sum(starts & (y == 0)))
    neg_windows = int(np.sum(y == 0))
    # Window-equivalent observed duration, not real-world continuous recording.
    neg_hours_equivalent = neg_windows * HOP_SECONDS / 3600.0
    event_starts = np.where((y == 1) & (~np.r_[False, y[:-1] == 1] | ~adjacent))[0]
    detected = 0
    for j, first in enumerate(event_starts):
        next_start = event_starts[j + 1] if j + 1 < len(event_starts) else len(y)
        run = first + 1
        while run < next_start and y[run] == 1 and adjacent[run]:
            run += 1
        detected += int(np.any(pred[first:run] == 1))
    return {
        "false_alert_episodes": false_episodes,
        "negative_window_equivalent_hours": neg_hours_equivalent,
        "false_alerts_per_window_equivalent_hour":
            false_episodes / neg_hours_equivalent if neg_hours_equivalent else np.nan,
        "stress_episodes": len(event_starts),
        "stress_episodes_detected": detected,
    }


def metrics(frame, probability, decision):
    y = frame["phase"].eq(2).to_numpy(dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = np.asarray(decision, dtype=int)
    negative = (y == 0)
    positive = (y == 1)
    rec = {
        "n": len(y), "n_stress": int(positive.sum()), "n_negative": int(negative.sum()),
        "stress_f1": float(f1_score(y, pred, zero_division=0)),
        "stress_recall": float(np.mean(pred[positive])) if positive.any() else np.nan,
        "fpr": float(np.mean(pred[negative])) if negative.any() else np.nan,
        "ap": float(average_precision_score(y, p)) if positive.any() else np.nan,
        "auroc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
    }
    rec.update(alert_counts(frame, pred))
    return rec


def run(features=FEATURES, baseline=BASELINE, output=OUT):
    if not Path(features).exists() or not Path(baseline).exists():
        raise FileNotFoundError("Local WESAD features/baseline missing: run on the owner's WESAD machine")
    f = pd.read_csv(features, dtype={"subject": str})
    b = pd.read_csv(baseline, dtype={"subject": str})
    f = f.sort_values(["subject", "start_s"])
    features_cols = [c for c in f if c.startswith(FEATURE_PREFIXES)]
    if not features_cols:
        raise ValueError("No wrist EDA/BVP/ACC columns in feature table")
    b = b.loc[b["model"].eq("lr") & b["sensors"].eq("all")].copy()
    if b.empty or b.duplicated(["subject", "end_s"]).any():
        raise ValueError("Missing or duplicated baseline LR ALL predictions")
    subjects = sorted(f["subject"].unique())
    if len(subjects) != 15:
        raise ValueError(f"Expected 15 WESAD participants; found {len(subjects)}")
    partitions = {}
    for subject in subjects:
        partitions[subject] = calibration_split(f.loc[f["subject"].eq(subject)])
    rows, predictions = [], []
    for held_out in subjects:
        # Train: each other subject's first 300s phase-1 calibration is isolated.
        train_parts = []
        for subject in subjects:
            cal, test, _ = partitions[subject]
            if subject == held_out:
                continue
            ref = cal[features_cols].mean()
            temp = test[["subject", "start_s", "end_s", "phase"] + features_cols].copy()
            temp.loc[:, features_cols] = temp[features_cols].subtract(ref, axis=1)
            train_parts.append(temp)
        train = pd.concat(train_parts, ignore_index=True)
        test_cal, test_post, meta = partitions[held_out]
        test_mean = test_cal[features_cols].mean()
        x_test = test_post[features_cols].subtract(test_mean, axis=1)
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1, class_weight="balanced", max_iter=3000,
                               random_state=42),
        )
        pipe.fit(train[features_cols], train["phase"].eq(2).astype(int))
        p_delta = pipe.predict_proba(x_test)[:, 1]
        # Cold-start predictions are independent held-out subject predictions.
        b_sub = b.loc[b["subject"].eq(held_out), ["end_s", "prob", "pred", "threshold"]]
        joined = test_post[["subject", "start_s", "end_s", "phase"]].merge(
            b_sub, on="end_s", how="left", validate="one_to_one"
        )
        if joined["prob"].isna().any():
            raise ValueError(f"Missing frozen cold-start prediction for {held_out}")
        p_cold = joined["prob"].to_numpy(dtype=float)
        y_delta = (p_delta >= .5).astype(int)
        y_cold = (p_cold >= .5).astype(int)
        # Exploratory recall guard: cold model's confident positives may restore
        # stress alerts, at the explicit risk of increased false alarms.
        y_guard = (y_delta.astype(bool) | (p_cold >= GUARD_COLD_PROB)).astype(int)
        policies = [
            ("cold_fixed05", p_cold, y_cold),
            ("cold_original_tuned", p_cold, joined["pred"].to_numpy(dtype=int)),
            ("delta_strict_fixed05", p_delta, y_delta),
            ("delta_guard_cold090_exploratory", p_delta, y_guard),
        ]
        for name, prob, pred in policies:
            item = metrics(joined, prob, pred)
            item.update(subject=held_out, policy=name, calibration_start_s=meta["start_s"],
                        calibration_end_s=meta["stop_s"],
                        calibration_windows=meta["calibration_windows"])
            rows.append(item)
            for record, pp, zz in zip(joined.to_dict("records"), prob, pred):
                predictions.append({
                    "subject": held_out, "start_s": record["start_s"],
                    "end_s": record["end_s"], "phase": record["phase"],
                    "policy": name, "prob": float(pp), "pred": int(zz),
                })
    result = pd.DataFrame(rows)
    details = pd.DataFrame(predictions)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "per_subject.csv", index=False)
    details.to_csv(out / "post_calibration_predictions.csv", index=False)
    aggregate = result.groupby("policy", as_index=False).agg(
        macro_stress_f1=("stress_f1", "mean"), macro_recall=("stress_recall", "mean"),
        macro_fpr=("fpr", "mean"), macro_ap=("ap", "mean"),
        macro_auroc=("auroc", "mean"),
        total_false_alerts=("false_alert_episodes", "sum"),
        total_negative_window_equivalent_hours=("negative_window_equivalent_hours", "sum"),
        total_stress_episodes=("stress_episodes", "sum"),
        total_detected_stress_episodes=("stress_episodes_detected", "sum"),
    )
    aggregate["false_alerts_per_window_equivalent_hour"] = (
        aggregate["total_false_alerts"] /
        aggregate["total_negative_window_equivalent_hours"]
    )
    aggregate["event_detection_rate"] = (
        aggregate["total_detected_stress_episodes"] / aggregate["total_stress_episodes"]
    )
    aggregate.to_csv(out / "summary.csv", index=False)
    config = {
        "source_features": str(features), "source_cold_oof": str(baseline),
        "calibration_seconds": CAL_SECONDS, "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS, "guard_cold_prob": GUARD_COLD_PROB,
        "thresholds": "fixed or frozen prior OOF; no held-out label-based tuning",
        "note": "All numbers recomputed; STRICT outputs replace no prior results. "
                "Guard policy exploratory, not verified safety guarantee.",
    }
    (out / "protocol.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False))
    return aggregate


if __name__ == "__main__":
    run()
