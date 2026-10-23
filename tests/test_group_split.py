import unittest
import numpy as np
import pandas as pd
from src.train import group_splits,build_model,select_threshold

class GroupSplitTest(unittest.TestCase):
    def setUp(self):
        self.df=pd.DataFrame({'subject':np.repeat(['a','b','c','d','e','f'],20),'y':np.tile([0,1],60),'eda_mean':np.arange(120.)})
    def test_all_outer_subjects_once_and_disjoint(self):
        seen=[]
        for _,tr,va,te,s in group_splits(self.df):
            a,b,c=[set(self.df.iloc[x].subject) for x in (tr,va,te)]
            self.assertFalse(a&b or a&c or b&c);self.assertEqual(len(c),1);seen+=s['test']
        self.assertEqual(set(seen),set(self.df.subject));self.assertEqual(len(seen),6)
    def test_preprocess_not_fit_on_test(self):
        _,tr,va,te,_=next(group_splits(self.df));m=build_model('lr')
        m.fit(self.df.iloc[tr][['eda_mean']],self.df.iloc[tr].y)
        self.assertAlmostEqual(m.named_steps['standardscaler'].mean_[0],self.df.iloc[tr].eda_mean.mean())
        prior=m.named_steps['standardscaler'].mean_.copy()
        m.predict_proba(pd.DataFrame({'eda_mean':[1e10]}))
        np.testing.assert_array_equal(prior,m.named_steps['standardscaler'].mean_)
    def test_threshold_only_uses_passed_validation(self):
        threshold=select_threshold(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]))
        self.assertTrue(.2<threshold<=.8)
if __name__=='__main__':unittest.main()
