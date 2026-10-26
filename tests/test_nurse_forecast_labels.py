"""Unit tests for prospective stress forecasting label generation and censoring."""

import unittest

import numpy as np

from src.forecasting.labels import classify_prediction_timestamp


class TestNurseForecastLabels(unittest.TestCase):
    def setUp(self):
        # Confirmed stress event from t=1000 to t=1600
        self.confirmed = [("EV_001", 1000.0, 1600.0)]
        # Unconfirmed 'na' candidate from t=4000 to t=4600
        self.na_events = [("EV_002_NA", 4000.0, 4600.0)]
        # Confirmed no-stress (0) interval from t=6000 to t=6600
        self.zero_events = [("EV_003_ZERO", 6000.0, 6600.0)]

    def test_p2_p1_p0_lead_time_boundaries(self):
        # 240s before onset -> P2 positive (180..300s)
        r_p2 = classify_prediction_timestamp(760.0, self.confirmed, self.na_events, self.zero_events)
        self.assertEqual(r_p2["state"], "PRE_ONSET_P2_180_300S")
        self.assertEqual(r_p2["y_P2"], 1)
        self.assertEqual(r_p2["y_P1"], -1)
        self.assertEqual(r_p2["y_P0"], -1)
        self.assertAlmostEqual(r_p2["lead_time_to_onset_s"], 240.0)

        # 120s before onset -> P1 positive (60..180s)
        r_p1 = classify_prediction_timestamp(880.0, self.confirmed, self.na_events, self.zero_events)
        self.assertEqual(r_p1["state"], "PRE_ONSET_P1_60_180S")
        self.assertEqual(r_p1["y_P1"], 1)
        self.assertEqual(r_p1["y_P0"], -1)
        self.assertEqual(r_p1["y_P2"], -1)
        self.assertAlmostEqual(r_p1["lead_time_to_onset_s"], 120.0)

        # 30s before onset -> P0 positive (0..60s)
        r_p0 = classify_prediction_timestamp(970.0, self.confirmed, self.na_events, self.zero_events)
        self.assertEqual(r_p0["state"], "PRE_ONSET_P0_0_60S")
        self.assertEqual(r_p0["y_P0"], 1)
        self.assertEqual(r_p0["y_P1"], -1)
        self.assertEqual(r_p0["y_P2"], -1)

    def test_ongoing_and_refractory_censored(self):
        # Inside event (t=1200) -> ONGOING_STRESS_EVENT (-1)
        r_on = classify_prediction_timestamp(1200.0, self.confirmed, self.na_events, self.zero_events)
        self.assertEqual(r_on["state"], "ONGOING_STRESS_EVENT")
        self.assertEqual(r_on["y_P0"], -1)
        self.assertEqual(r_on["y_P1"], -1)
        self.assertEqual(r_on["y_P2"], -1)

        # 300s after event end (t=1900) -> POST_EVENT_REFRACTORY (-1)
        r_ref = classify_prediction_timestamp(1900.0, self.confirmed, self.na_events, self.zero_events)
        self.assertEqual(r_ref["state"], "POST_EVENT_REFRACTORY")
        self.assertEqual(r_ref["y_P1"], -1)

    def test_unconfirmed_na_never_labeled_negative(self):
        # Inside or near unconfirmed 'na' event (t=3900 or t=4200) -> UNKNOWN_NA_CENSORED (-1)
        for t_check in (3850.0, 4200.0, 4800.0):
            r_na = classify_prediction_timestamp(t_check, self.confirmed, self.na_events, self.zero_events)
            self.assertEqual(r_na["state"], "UNKNOWN_NA_CENSORED")
            self.assertEqual(r_na["y_P0"], -1)
            self.assertEqual(r_na["y_P1"], -1)
            self.assertEqual(r_na["y_P2"], -1)

    def test_verified_negative_and_survey_zero(self):
        # Clean non-event background (t=3000) -> VERIFIED_NON_EVENT (0)
        r_neg = classify_prediction_timestamp(3000.0, self.confirmed, self.na_events, self.zero_events)
        self.assertEqual(r_neg["state"], "VERIFIED_NON_EVENT")
        self.assertEqual(r_neg["y_P0"], 0)
        self.assertEqual(r_neg["y_P1"], 0)
        self.assertEqual(r_neg["y_P2"], 0)

        # Survey confirmed Stress level 0 (t=6200) -> CONFIRMED_NO_STRESS_0 (0)
        r_z = classify_prediction_timestamp(6200.0, self.confirmed, self.na_events, self.zero_events)
        self.assertEqual(r_z["state"], "CONFIRMED_NO_STRESS_0")
        self.assertEqual(r_z["y_P1"], 0)


if __name__ == "__main__":
    unittest.main()
