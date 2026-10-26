"""Regression tests for audit fixes: horizon matching and episode semantics."""
import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/nurse_forecasting'))
from root_cause_analysis import strict_events
from src.features.nurse_forecasting import compute_ibi_hrv_in_window

class TestNurseRootCauseAudit(unittest.TestCase):
    def event(self):
        return pd.DataFrame([{'event_id':'e','subject':'s','matched_session':'a','event_start_ts':1000.,'event_end_ts':1300.}])

    def rows(self,times,session='a'):
        return pd.DataFrame({'subject':'s','session':session,'pred_ts':times,'pred':1,'y_true':0,'exposure_s':30.})

    def test_wrong_horizon_is_not_true_alert(self):
        # Legacy matcher accepts 300s for every task. P1 requires 60 < lead <= 180.
        m,_=strict_events(self.rows([750.]),self.event())
        self.assertEqual(m['hit_episodes'],0)
        self.assertEqual(m['false_episodes'],1)

    def test_episode_span_without_an_actual_in_horizon_warning_is_not_hit(self):
        # Warnings at lead 200 and 40 merge; neither is in P1.
        m,_=strict_events(self.rows([800.,960.]),self.event())
        self.assertEqual(m['hit_episodes'],0)
        self.assertEqual(m['detected_events'],0)

    def test_session_boundary_is_respected(self):
        m,_=strict_events(self.rows([850.],session='b'),self.event())
        self.assertEqual(m['detected_events'],0)

    def test_minimum_lead_is_exclusive_like_labels(self):
        m,_=strict_events(self.rows([940.]),self.event())
        self.assertEqual(m['detected_events'],0)
        m,_=strict_events(self.rows([820.]),self.event())
        self.assertEqual(m['detected_events'],1)

    def test_coverage_and_first_issuance_differ(self):
        p=self.rows([700.,800.,850.,900.])
        coverage,_=strict_events(p,self.event())
        issuance,_=strict_events(p,self.event(),episode_mode='trigger')
        self.assertEqual(coverage['detected_events'],1)
        self.assertEqual(issuance['detected_events'],0)

    def test_future_ibi_does_not_enter_hrv(self):
        beats=np.column_stack([np.arange(20.,500.,.8),np.full(600,.8)])
        changed=beats.copy();changed[changed[:,0]>=300.,1]=.35
        original=compute_ibi_hrv_in_window(beats,1000.,1000.,120.,300.)
        future_changed=compute_ibi_hrv_in_window(changed,1000.,1000.,120.,300.)
        self.assertEqual(original,future_changed)

if __name__=='__main__':unittest.main()
