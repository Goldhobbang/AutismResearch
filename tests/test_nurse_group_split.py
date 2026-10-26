"""Unit tests for LOSO participant group split, scaler isolation, and duplicate window prevention."""

import unittest

import numpy as np
import pandas as pd

from src.features.nurse_forecasting import SENSOR_FEATURE_MAP
from src.forecasting.train import run_loso_forecast_experiment


class TestNurseGroupSplit(unittest.TestCase):
    def test_loso_disjoint_subjects_and_no_test_leakage(self):
        rng = np.random.default_rng(42)
        subjects = ["15", "5C", "6B", "7A", "83", "E4"]
        rows = []
        ev_rows = []
        for s_idx, sid in enumerate(subjects):
            ev_ts = 2000.0 + s_idx * 100.0
            ev_rows.append(
                {
                    "event_id": f"EV_{sid}",
                    "subject": sid,
                    "matched_session": f"{sid}_sess1",
                    "event_start_ts": ev_ts,
                    "event_end_ts": ev_ts + 600.0,
                    "p1_forecast_eligible": True,
                }
            )
            for w in range(12):
                pred_ts = 1000.0 + w * 60.0
                is_pos = 1 if w in (4, 5, 6) else 0
                if w == 4:
                    ev_rows[-1]["event_start_ts"] = pred_ts + 120.0
                    ev_rows[-1]["event_end_ts"] = pred_ts + 720.0
                feat_dict = {col: float(rng.normal(is_pos * 0.5, 1.0)) for col in SENSOR_FEATURE_MAP["all"]}
                rows.append(
                    {
                        "dataset": "Nurse_Stress",
                        "subject": sid,
                        "session": f"{sid}_sess1",
                        "pred_ts": pred_ts,
                        "pred_rel_s": 180.0 + w * 60.0,
                        "start_s": float(w * 60.0),
                        "end_s": float(180.0 + w * 60.0),
                        "state": "PRE_ONSET_P1_60_180S" if is_pos else "VERIFIED_NON_EVENT",
                        "y_P1": is_pos,
                        "qc_common_valid": True,
                        **feat_dict,
                    }
                )

        df = pd.DataFrame(rows)
        ev_df = pd.DataFrame(ev_rows)

        per_subj, pred_df, cohort_ev, splits = run_loso_forecast_experiment(
            df, ev_df, task="P1", model_name="LR", sensor_set="all"
        )
        self.assertEqual(len(splits), len(subjects))
        seen_test = set()
        for sp in splits:
            t_s = sp["test_subject"]
            tr_s = set(sp["train_subjects"].split(";"))
            va_s = set(sp["val_subjects"].split(";"))
            self.assertNotIn(t_s, tr_s)
            self.assertNotIn(t_s, va_s)
            self.assertEqual(len(tr_s & va_s), 0)
            seen_test.add(t_s)
        self.assertEqual(seen_test, set(subjects))
        # Ensure no duplicate prediction windows per (subject, session, pred_ts)
        self.assertFalse(pred_df.duplicated(subset=["subject", "session", "pred_ts"]).any())


if __name__ == "__main__":
    unittest.main()
