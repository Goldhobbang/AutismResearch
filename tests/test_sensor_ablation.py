import unittest
import pandas as pd
from src.train import SENSORS,feature_columns,eligible_mask
class AblationTest(unittest.TestCase):
    def test_shared_windows_and_no_metadata_features(self):
        df=pd.DataFrame({'dataset':['synthetic_test']*4,'subject':['s']*4,'session':['a']*4,'y':[0,1,0,-1],
            'phase':[1,2,3,0],'qc_common_valid':[True,True,False,True],
            'eda_mean':[1.,2.,3.,4.],'ppg_mean':[4.,3.,2.,1.],'acc_std':[.1,.2,.3,.4],
            'qc_eda_valid':[True]*4,'end_s':[60,75,90,105]})
        common=df[eligible_mask(df)]
        for sensor in SENSORS:
            x=common[feature_columns(common,sensor)]
            self.assertEqual(list(x.index),[0,1]);self.assertFalse(set(x)&{'subject','phase','y','end_s','qc_eda_valid'})
        self.assertEqual(len(feature_columns(df,'all')),3)
    def test_missing_quality_gate_fails_closed(self):
        with self.assertRaises(ValueError):eligible_mask(pd.DataFrame({'y':[0,1]}))
if __name__=='__main__':unittest.main()
