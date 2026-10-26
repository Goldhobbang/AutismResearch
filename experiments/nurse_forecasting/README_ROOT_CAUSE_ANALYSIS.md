# Nurses failure root cause audit

The Korean findings are in `reports/NURSE_FAILURE_ROOT_CAUSE_ANALYSIS_KO.md`.
Comparison tables, 33 LOSO prediction sets, ten figures, and the verification record
are in `results/nurse_forecasting/root_cause_analysis/`.

## Input caches in Git

The original `features.csv` and `predictions.csv` remain unchanged locally.
Git contains their lossless gzip copies in `root_cause_analysis/inputs/` because
the uncompressed predictions exceed GitHub's single-file size limit.
Restore them from the repository root before running the audit:

```powershell
.venv\Scripts\python.exe experiments/nurse_forecasting/restore_root_cause_inputs.py
```

The restore command verifies both the archives and their decompressed SHA256.
It refuses to overwrite any existing file. `--verify-only` checks the archives
without restoring files.

The raw sensor archives, original survey workbook, and joblib models are excluded
under the repository's existing ignore policy. A full raw-data audit requires the
original Nurses dataset at `data/raw/Nurse_Stress/`. The preservation verification
also requires the original model files; `verification.json` records the completed
checks in the original workspace. Saved comparisons and prediction sets can be
inspected without these raw inputs.

Use the repository requirements plus LightGBM (the audit used 4.7.0).
`environment.json` records the measured numerical environment. The report contains
the analysis commands, limitations, and the difference between reproducing saved
metrics and refitting a numerically identical model.
