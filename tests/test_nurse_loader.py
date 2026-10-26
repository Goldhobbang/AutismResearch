"""Unit tests for Nurses Stress dataset loader, timestamp alignment, and row counts."""

import unittest
from pathlib import Path

import numpy as np

from src.loaders.nurse_stress import (
    audit_nurse_dataset,
    load_nurse_session_zip,
    parse_survey_results_xlsx,
    resolve_nurse_root,
)


class TestNurseLoader(unittest.TestCase):
    def test_root_and_survey_parsing(self):
        root = resolve_nurse_root()
        self.assertTrue(root.exists())
        survey_path = root / "SurveyResults.xlsx"
        self.assertTrue(survey_path.exists())

        df = parse_survey_results_xlsx(survey_path)
        self.assertEqual(len(df), 358)
        self.assertEqual(df["subject"].nunique(), 15)
        self.assertTrue((df["event_end_ts"] >= df["event_start_ts"]).all())
        # All survey start/end times are minute-rounded (60s resolution)
        self.assertTrue((df["time_resolution_s"] == 60).all())

    def test_session_zip_sensor_alignment(self):
        root = resolve_nurse_root()
        sample_zip = sorted((root / "15").glob("*.zip"))[0]
        sdata = load_nurse_session_zip(sample_zip, include_temp=True)

        self.assertEqual(sdata["subject"], "15")
        self.assertEqual(sdata["rates"]["EDA"], 4.0)
        self.assertEqual(sdata["rates"]["BVP"], 64.0)
        self.assertEqual(sdata["rates"]["ACC"], 32.0)
        self.assertEqual(sdata["ACC"].shape[1], 3)

        # Start timestamps of EDA, BVP, ACC must match within 0.001s
        t_eda = sdata["start_timestamps"]["EDA"]
        t_bvp = sdata["start_timestamps"]["BVP"]
        t_acc = sdata["start_timestamps"]["ACC"]
        self.assertAlmostEqual(t_eda, t_bvp, places=3)
        self.assertAlmostEqual(t_eda, t_acc, places=3)

        # Duration spread across primary wrist modalities must be < 5 seconds
        durs = list(sdata["durations_s"].values())
        self.assertLessEqual(max(durs) - min(durs), 5.0)


if __name__ == "__main__":
    unittest.main()
