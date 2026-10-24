"""Synthetic protocol tests only: these are NOT WESAD performance tests."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "experiments/phase3/experiment_d_calibration_strict.py"
spec = importlib.util.spec_from_file_location("calibration_strict", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make_subject():
    # 24 minutes, 60s windows sliding every 15s. First 12 minutes phase 1,
    # then 8 minutes stress, then baseline/amusement.
    records = []
    for start in range(0, 24 * 60 - 60 + 1, 15):
        end = start + 60
        phase = 1 if end <= 720 else 2 if start >= 720 and end <= 1200 else 3 if start >= 1200 else 0
        records.append({
            "subject": "S2", "start_s": start, "end_s": end,
            "phase": phase, "qc_common_valid": True,
        })
    return pd.DataFrame(records)


class StrictCalibrationTests(unittest.TestCase):
    def test_exact_300_seconds_and_no_overlap(self):
        source = make_subject()
        cal, test, meta = mod.calibration_split(source)
        self.assertEqual(meta["duration_s"], 300)
        self.assertEqual(len(cal), 17)
        self.assertEqual(cal.start_s.min(), 0)
        self.assertEqual(cal.end_s.max(), 300)
        self.assertGreaterEqual(test.start_s.min(), 300)
        self.assertEqual(len(set(cal.index) & set(test.index)), 0)

    def test_missing_calibration_window_fails_closed(self):
        frame = make_subject()
        frame = frame.loc[~frame.start_s.eq(45)]
        with self.assertRaisesRegex(ValueError, "17 uninterrupted"):
            mod.calibration_split(frame)

    def test_baseline_quality_failure_fails_closed(self):
        frame = make_subject()
        frame.loc[frame.start_s.eq(30), "qc_common_valid"] = False
        with self.assertRaisesRegex(ValueError, "sensor quality"):
            mod.calibration_split(frame)

    def test_fpr_and_episode_counts_are_distinct(self):
        frame = pd.DataFrame({
            "start_s": np.arange(0, 7) * 15,
            "phase": [1, 1, 1, 2, 2, 1, 1],
        })
        result = mod.alert_counts(frame, [0, 1, 1, 1, 1, 0, 1])
        self.assertEqual(result["false_alert_episodes"], 2)
        self.assertEqual(result["stress_episodes"], 1)
        self.assertEqual(result["stress_episodes_detected"], 1)

    def test_alert_state_resets_on_time_gap(self):
        frame = pd.DataFrame({
            "start_s": [0, 15, 150, 165],
            "phase": [1, 1, 1, 1],
        })
        result = mod.alert_counts(frame, [1, 1, 1, 1])
        self.assertEqual(result["false_alert_episodes"], 2)


if __name__ == "__main__":
    unittest.main()
