"""Regression tests for the fixed 30-second full-timeline replay."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "nurse_forecasting"))
from fixed_grid_30s_replay import (  # noqa: E402
    FIXED_STEP_S,
    VERIFIED_STATE,
    UNVERIFIED_STATE,
    _hours,
    build_fixed_grid_times,
    evaluate_alert_stream,
    fold_splits,
    p1_positive_mask,
)


class TestFixedGrid30s(unittest.TestCase):
    def test_grid_has_constant_30s_spacing(self):
        times = build_fixed_grid_times(1000.0)
        self.assertEqual(times[0], 180.0)
        self.assertTrue(np.allclose(np.diff(times), FIXED_STEP_S))
        self.assertTrue(all(t <= 995.0 for t in times))

    def test_grid_is_independent_of_labels(self):
        self.assertEqual(build_fixed_grid_times(600.0), build_fixed_grid_times(600.0))
        self.assertEqual(build_fixed_grid_times(170.0), [])

    def test_p1_horizon_is_open_at_60_and_closed_at_180(self):
        states = pd.Series(["PRE_ONSET_P1_60_180S"] * 3)
        leads = pd.Series([180.0, 120.0, 60.0])
        mask = p1_positive_mask(states, leads)
        self.assertEqual(mask.tolist(), [True, True, False])

    def test_loso_folds_have_no_participant_overlap(self):
        splits = fold_splits([f"S{i:02d}" for i in range(15)])
        self.assertEqual(len(splits), 15)
        for sp in splits:
            tr = set(sp["train_subjects"].split(";"))
            va = set(sp["val_subjects"].split(";"))
            self.assertNotIn(sp["test_subject"], tr | va)
            self.assertEqual(tr & va, set())

    def _stream(self):
        times = np.arange(700.0, 861.0, 30.0)
        return pd.DataFrame({
            "subject": "s", "session": "a", "pred_ts": times,
            "state": "VERIFIED_NON_EVENT",
            "y_true": 0,
            "qc_common_valid": True,
            "prob_gated": 1.0,
            "threshold_gated": 0.5,
            "exposure_s": 30.0,
        })

    def _event(self, onset=1000.0):
        return pd.DataFrame([{
            "event_id": "e", "subject": "s", "matched_session": "a",
            "event_start_ts": onset,
        }])

    def test_active_warning_and_first_issuance_are_distinct(self):
        stream = self._stream()
        event = self._event()
        active = evaluate_alert_stream(stream, event, "val_opt", "active")
        issued = evaluate_alert_stream(stream, event, "val_opt", "issued")
        self.assertEqual(active["detected_events"], 1)
        self.assertEqual(issued["detected_events"], 0)

    def test_horizon_excludes_lead_of_exactly_60_seconds(self):
        stream = self._stream()
        stream = stream[stream.pred_ts.eq(940.0)].copy()
        m = evaluate_alert_stream(stream, self._event(), "val_opt", "active")
        self.assertEqual(m["detected_events"], 0)

    def test_verified_and_unverified_exposure_are_separate(self):
        stream = pd.DataFrame({
            "state": [VERIFIED_STATE, VERIFIED_STATE, UNVERIFIED_STATE],
            "y_true": [0, 0, 0],
            "qc_common_valid": [True, True, True],
            "exposure_s": [30.0, 30.0, 60.0],
        })
        self.assertAlmostEqual(_hours(stream, VERIFIED_STATE), 60.0 / 3600.0)
        self.assertAlmostEqual(_hours(stream, UNVERIFIED_STATE), 60.0 / 3600.0)
        self.assertAlmostEqual(_hours(stream), 120.0 / 3600.0)


if __name__ == "__main__":
    unittest.main()
