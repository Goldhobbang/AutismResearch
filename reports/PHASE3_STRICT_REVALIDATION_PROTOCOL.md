# Phase 3 strict revalidation: protocol and execution status

- Date: 2026-10-24 (KST)
- Prior Phase 3 result commit: 47d1407
- New scripts: experiments/phase3/experiment_d_calibration_strict.py, experiments/phase3/integrate_strict_metrics.py, experiments/phase3/physionet_multi_subject_audit.py
- Status: **CODE COMMITTED; AUTHENTIC LOCAL-DATA RERUN PENDING**
- Do **not** represent provisional results as rigorously revalidated outcomes.

## Why the preceding headline numbers require a new run

1. The previous "5min" calibration used the first 20 overlapping 60s windows (15s hop), spanning approximately 345 seconds. Exact first-300-second calibration instead spans 17 windows.
2. Previously reported evaluation included windows also used to compute calibration statistics. Strict evaluation excludes every window starting before calibration end.
3. The prior integration script hardcoded the calibrated model's macro F1=0.8931, AUROC=0.9421 and AP=0.915; these fields were not independently derived from its calibration prediction file.
4. S2 stress F1 fell from 0.8276 to 0.0513 in the old run. S7 improvement did not establish a universally safe calibration policy.
5. The old episode FAR/hour normalizes alert counts by eligible negative window count multiplied by 15 seconds; this is window-equivalent monitoring time, not necessarily wall-clock operating time.
6. WESAD's adult current-state stress is not early forecasting of autistic meltdown events.

## New strict protocol

- First uninterrupted **300.0 seconds** of phase 1 within each subject is calibration-only.
- For 60-second sliding feature windows at 15-second hop, exactly 17 qualifying calibration windows are expected.
- All test windows must START at/after calibration end, preventing shared raw samples.
- For a held-out participant, only its unlabeled baseline feature means are used to adapt input. The stress labels and future data are never used.
- Training-side subjects use the same first-300s baseline procedure.
- Four policies are evaluated on the **identical post-calibration windows**: old frozen cold-start fixed 0.5, frozen preselected threshold, strict delta LR 0.5, and an exploratory OR-based high-confidence cold-start guard.
- The OR guard is *not* a validated safety guarantee; report its negative effects as well as any positive ones.
- Macro two-class F1 is calculated from per-subject y_true/y_pred rather than hardcoded.
- All macro AUROC/AP values are defined as means of held-out subject metrics. Report pooled metrics separately if desired.

## Local execution (Windows, from repo root)

    git pull origin main
    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_calibration_strict.py" -v
    .\.venv\Scripts\python.exe experiments/phase3/experiment_d_calibration_strict.py
    .\.venv\Scripts\python.exe experiments/phase3/integrate_strict_metrics.py

If the virtual environment is absent, create it and install the repository requirements first.

Inputs:

- data/processed/wesad_features.csv (from authentic local WESAD; rebuild if necessary)
- results/baselines/wesad_20261024/predictions.csv (frozen original OOF)

Expected strict outputs:

- results/phase3_strict/calibration/per_subject.csv
- results/phase3_strict/calibration/post_calibration_predictions.csv
- results/phase3_strict/calibration/summary.csv
- results/phase3_strict/calibration/strict_source_derived_metrics.csv
- reports/PHASE3_STRICT_CALIBRATION_RESULTS.md (generated ONLY after real input run)

The original Phase 3 files are preserved and must be labeled **PRELIMINARY** until this strict run establishes corrected figures. Do not copy the old headline numbers into the strict report.

## PhysioNet external multi-participant runner

    .\.venv\Scripts\python.exe experiments/phase3/physionet_multi_subject_audit.py

Prerequisites:

- data/processed/physionet/verified_segments.csv
- data/registry/SHA256SUMS.txt
- raw PhysioNet participant folders under data/raw/physionet_v1/Wearable_Dataset/
- all frozen WESAD models under results/baselines/wesad_20261024/models/

The runner verifies official SHA-256 for required raw files, records missing subject/session combinations and evaluates available verified subjects independently. It does **not** download missing raw participant files or train new PhysioNet models. To expand beyond S01, acquire additional publicly accessible files through the approved existing downloader / official PhysioNet site and retain hashes and protocol-specific exclusions. Fragmented sessions require a separately audited loader.

Outputs:

- results/phase3_strict/physionet_external/coverage_manifest.csv
- results/phase3_strict/physionet_external/per_window_predictions.csv (only if any eligible data)
- results/phase3_strict/physionet_external/per_subject_metrics.csv (only if any eligible data)

**Do not treat overlapping windows as independent participants**. Compare across subjects, confidence intervals and performance by task; AP must be interpreted relative to stress-label prevalence.

## Remaining acceptance gates

- Complete the actual local WESAD rerun (all 15 subjects).
- Check S2 stress recall/F1 and S7 FPR side by side against the same post-calibration windows.
- Demonstrate no train/test contamination and no calibration/evaluation overlap.
- Fix any synthetic unit-test failure before trusting results.
- Acquire multiple PhysioNet raw participants and complete independent verification.
- Only then revise model selection, final report and deployment claims.
