"""Regression tests for PhysioNet verified manifest session name semantics."""
from pathlib import Path
import unittest
import pandas as pd
from experiments.phase3.physionet_multi_subject_audit import select_verified_session_segments

ROOT = Path(__file__).resolve().parents[1]

class ManifestSubjectSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = pd.read_csv(ROOT / "data/processed/physionet/verified_segments.csv")
        cls.manifest = cls.manifest[
            cls.manifest["verified"].astype(str).str.lower().isin(["true", "1"])
        ]

    def test_s01_stress_and_aerobic_are_discoverable(self):
        participant = self.manifest.loc[self.manifest.subject.eq("S01")]
        stress = select_verified_session_segments(participant, "S01", "STRESS")
        aerobic = select_verified_session_segments(participant, "S01", "AEROBIC")
        self.assertGreater(len(stress), 0)
        self.assertGreater(len(aerobic), 0)
        self.assertTrue(stress.session.eq("S01_STRESS").all())
        self.assertTrue(aerobic.session.eq("S01_AEROBIC").all())

    def test_other_subject_sessions_are_never_mixed(self):
        group = self.manifest.loc[self.manifest.subject.isin(["S01", "S03"])]
        selected = select_verified_session_segments(group, "S03", "STRESS")
        self.assertGreater(len(selected), 0)
        self.assertTrue(selected.subject.eq("S03").all())

if __name__ == "__main__":
    unittest.main()
