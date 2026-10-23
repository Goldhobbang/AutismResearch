"""Recorded adult stress replay. Quality-gated; not ASD event forecasting."""
import argparse
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from .train import boolean_column
from .evaluate import write_json

def replay(args):
    out=Path(args.out); pred=pd.read_csv(out/'predictions.csv',dtype={'subject':str,'session':str})
    g=pred[(pred.subject==args.subject)&(pred.model==args.model)&(pred.sensors==args.sensors)].sort_values(['session','end_s']).copy()
    if g.empty:raise ValueError('No held-out rows for requested subject/model/sensors.')
    mode='cached OOF score replay'
    if args.raw_root:
        import pickle
        from .features.core import window_features
        mode='causal raw-window feature recomputation plus model inference'
        artifact=joblib.load(out/'models'/f'{args.model}_{args.sensors}_fold{int(g.fold.iloc[0])}.joblib')
        assert args.subject in artifact['subjects']['test']
        assert args.subject not in artifact['subjects']['train']+artifact['subjects']['validation']
        path=Path(args.raw_root)/args.subject/f'{args.subject}.pkl'
        with path.open('rb') as f:wrist=pickle.load(f,encoding='latin1')['signal']['wrist']
        probabilities=[];latencies=[]
        for row in g.itertuples():
            tic=time.perf_counter();feature=window_features(wrist,row.start_s,row.end_s)
            x=pd.DataFrame([feature])[artifact['columns']]
            probabilities.append(float(artifact['model'].predict_proba(x)[0,1]))
            latencies.append((time.perf_counter()-tic)*1000)
        np.testing.assert_allclose(probabilities,g.prob,atol=1e-7)
        g['prob']=probabilities;g['feature_and_inference_ms']=latencies
    valid=boolean_column(g.qc_common_valid).to_numpy()
    streak=0;last_alarm=-np.inf;last_time=None;last_session=None;alerts=[]
    for ok,row in zip(valid,g.itertuples()):
        if row.session!=last_session:
            streak=0;last_alarm=-np.inf;last_time=None
        if last_time is not None and row.end_s-last_time>row.step_s*1.5:streak=0
        streak=streak+1 if ok and row.prob>=row.threshold else 0
        alarm=bool(ok and streak>=2 and row.end_s-last_alarm>=60)
        if alarm:last_alarm=row.end_s;streak=0
        alerts.append(int(alarm));last_time=row.end_s;last_session=row.session
    g['alert']=alerts;g['measurement_state']=np.where(valid,'score_available','measurement_unavailable')
    stem=f'replay_{args.subject}_{args.model}_{args.sensors}'
    g.to_csv(out/(stem+'.csv'),index=False)
    eligible=boolean_column(g.evaluation_eligible)
    neg=g[eligible & g.y.eq(0)];duration=neg.step_s.sum()/3600
    report=dict(task='adult stress replay; not meltdown forecasting',mode=mode,subject=args.subject,
        alerts=int(g.alert.sum()),quality_suppressed_windows=int((~valid).sum()),
        alerts_in_scored_nonstress=int(neg.alert.sum()),scored_nonstress_hours=float(duration),
        stress_alerts_per_scored_nonstress_hour=float(neg.alert.sum()/duration) if duration else None,
        unscored_alerts=int(g.loc[~eligible,'alert'].sum()),cooldown_s=60,consecutive_threshold_exceedances=2,
        asd_event_precision=None,asd_event_recall=None,asd_false_alarms_day=None,asd_lead_time=None,
        notes='Feature lookback only; file replay faster than wall clock. No BLE/device/caregiver test. Labels do not gate alerts; sensor quality does.')
    if args.raw_root:
        report.update(feature_and_inference_p50_ms=float(g.feature_and_inference_ms.median()),
                      feature_and_inference_p95_ms=float(g.feature_and_inference_ms.quantile(.95)))
    write_json(out/(stem+'.json'),report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,4));ax.plot(g.end_s/60,g.prob.where(valid),label='Held-out stress score (uncalibrated; QC gated)')
    ax.plot(g.end_s/60,g.threshold,'--',label='Inner-validation threshold')
    ax.fill_between(g.end_s/60,0,1,where=eligible&g.y.eq(1),alpha=.15,label='Eligible stress window')
    ax.fill_between(g.end_s/60,0,1,where=~valid,alpha=.15,color='gray',label='Measurement unavailable')
    ax.scatter(g.loc[g.alert.eq(1),'end_s']/60,g.loc[g.alert.eq(1),'prob'],marker='x',color='red',label='Simulated stress alert')
    ax.set(xlabel='Minutes from recording start',ylabel='Score',ylim=(0,1),title=f'{g.dataset.iloc[0]} / {args.subject} / LOSO — STRESS, NOT MELTDOWN FORECASTING')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'figures'/(stem+'.png'),dpi=160);plt.close(fig)
    print(report)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='results/wesad')
    p.add_argument('--subject',default='S2');p.add_argument('--model',default='lr');p.add_argument('--sensors',default='all')
    p.add_argument('--raw-root',help='Official trusted WESAD root for raw recomputation')
    replay(p.parse_args())
if __name__=='__main__':main()
