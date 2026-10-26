"""Unit tests for alert episode merging, event-level recall, lead time, and false alarm rate calculation."""

import unittest

import numpy as np
import pandas as pd

from src.forecasting.alerts import (
    apply_alert_policy_to_session,
    evaluate_event_level_forecasting,
    merge_alert_episodes,
)


class TestNurseAlerts(unittest.TestCase):
    def test_overlapping_windows_merge_into_single_alert_episode(self):
        # 5 consecutive positive windows spaced by 30s -> must merge into 1 episode, NOT 5
        pred_df = pd.DataFrame(
            {
                "subject": ["7A"] * 6,
                "session": ["7A_1"] * 6,
                "pred_ts": [1000.0, 1030.0, 1060.0, 1090.0, 1120.0, 2000.0],
                "prob": [0.8, 0.85, 0.9, 0.75, 0.7, 0.8],
                "pred": [1, 1, 1, 1, 1, 1],
                "state": ["VERIFIED_NON_EVENT"] * 6,
            }
        )
        eps = merge_alert_episodes(pred_df, pred_col="pred", merge_gap_s=180.0)
        self.assertEqual(len(eps), 2)
        self.assertEqual(eps[0]["n_windows"], 5)
        self.assertAlmostEqual(eps[0]["alert_start_ts"], 1000.0)
        self.assertAlmostEqual(eps[0]["alert_end_ts"], 1120.0)
        self.assertEqual(eps[1]["n_windows"], 1)

    def test_event_recall_lead_time_and_false_alarms_per_hour(self):
        # Event 1 at t=2000 (warned at t=1850 -> lead time 150s, and t=1910 -> merged into 1 hit episode)
        # Event 2 at t=5000 (missed)
        # False alarm episode at t=3500..3560 (in verified non-event region)
        ev_df = pd.DataFrame(
            {
                "event_id": ["EV_1", "EV_2"],
                "subject": ["7A", "7A"],
                "event_start_ts": [2000.0, 5000.0],
                "event_end_ts": [2600.0, 5600.0],
            }
        )

        # Create 60 negative windows (1 hour = 3600s) + 2 pre-onset windows
        neg_ts = np.arange(10000.0, 13600.0, 60.0)
        rows = []
        for ts in neg_ts:
            rows.append(
                {
                    "subject": "7A",
                    "session": "7A_1",
                    "pred_ts": float(ts),
                    "prob": 0.8 if 11000.0 <= ts <= 11120.0 else 0.1,
                    "pred": 1 if 11000.0 <= ts <= 11120.0 else 0,
                    "state": "VERIFIED_NON_EVENT",
                }
            )
        # Add pre-onset windows for EV_1 (t=1850 and t=1910)
        rows.append({"subject": "7A", "session": "7A_2", "pred_ts": 1850.0, "prob": 0.9, "pred": 1, "state": "PRE_ONSET_P1_60_180S"})
        rows.append({"subject": "7A", "session": "7A_2", "pred_ts": 1910.0, "prob": 0.85, "pred": 1, "state": "PRE_ONSET_P1_60_180S"})
        # Add unconfirmed NA window with pred=1 -> must NOT be counted as false alarm
        rows.append({"subject": "7A", "session": "7A_2", "pred_ts": 8000.0, "prob": 0.95, "pred": 1, "state": "UNKNOWN_NA_CENSORED"})

        pred_df = pd.DataFrame(rows)
        res = evaluate_event_level_forecasting(
            pred_df, ev_df, pred_col="pred", max_valid_lead_s=180.0, min_valid_lead_s=60.0, merge_gap_s=180.0
        )
        self.assertEqual(res["n_total_events"], 2)
        self.assertEqual(res["n_detected_events"], 1)
        self.assertEqual(res["n_missed_events"], 1)
        self.assertAlmostEqual(res["event_recall"], 0.5)
        self.assertAlmostEqual(res["lead_time_mean_s"], 150.0)
        self.assertEqual(res["n_events_warned_ge_60s"], 1)
        self.assertEqual(res["false_alarm_episodes"], 1)
        self.assertAlmostEqual(res["verified_non_event_hours"], 1.0, places=2)
        self.assertAlmostEqual(res["false_alarms_per_hour"], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
