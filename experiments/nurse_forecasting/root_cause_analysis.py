"""Bounded forensic audit. Never writes outside root_cause_analysis outputs.

Run from repository root with .venv/Scripts/python.exe. See --help.
Existing predictions are immutable evidence, never used to tune new experiments.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time, zipfile, io, platform
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit, logit
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, brier_score_loss, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.forecasting.train import build_forecast_model, select_validation_threshold
from src.forecasting.labels import classify_prediction_timestamp
from src.forecasting.calibration import apply_session_causal_calibration
from src.forecasting.alerts import evaluate_event_level_forecasting, apply_alert_policy_to_session, merge_alert_episodes
from src.forecasting.evaluate import compute_window_metrics
from src.features.advanced import CALIBRATED_PHYSIO_FEATURES, compute_log_base_features, apply_dynamic_motion_gate, compute_causal_exercise_score
from src.features.nurse_forecasting import SENSOR_FEATURE_MAP, extract_nurse_window_features
from src.loaders.nurse_stress import parse_survey_results_xlsx, parse_e4_csv_bytes
BASE=ROOT/'results/nurse_forecasting'
OUT=BASE/'root_cause_analysis'
RAW=ROOT/'data/raw/Nurse_Stress'
HORIZONS={'P0':(0.,60.),'P1-A':(60.,120.),'P1-B':(120.,180.),'P1':(60.,180.),'P2':(180.,300.)}
NEG=['VERIFIED_NON_EVENT','CONFIRMED_NO_STRESS_0']
KEY=['subject','session','pred_ts']
DT={'subject':str,'session':str,'matched_session':str}

def save(df,name):
    OUT.mkdir(parents=True,exist_ok=True)
    df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')

def jsave(obj,name):
    (OUT/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o)),encoding='utf-8')

def read(name):
    return pd.read_csv(BASE/(name+'.csv'),dtype={k:v for k,v in DT.items()},low_memory=False)

def data():
    return read('features'),read('event_audit'),json.loads((BASE/'dataset_manifest.json').read_text(encoding='utf-8'))

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def snapshot():
    paths=list(BASE.glob('*.csv'))+list((BASE/'models').glob('*'))+[BASE/'dataset_manifest.json']
    paths+=list((ROOT/'src/forecasting').glob('*.py'))+[ROOT/'src/features/nurse_forecasting.py',ROOT/'src/loaders/nurse_stress.py']
    paths+=list((ROOT/'reports').glob('NURSE*.md'))+[RAW/'SurveyResults.xlsx']
    save(pd.DataFrame([{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in paths]),'original_input_hashes')

def events_for(e,task):
    lo,hi=HORIZONS[task]
    return e[(e.label_category.isin(['confirmed_high_stress','confirmed_moderate_stress'])) & e.onset_covered & (e.pre_onset_history_s>=180+lo)].copy()

def task_labels(f,task):
    lo,hi=HORIZONS[task]
    y=np.full(len(f),-1,dtype=int)
    y[f.state.isin(NEG)]=0
    pre=f.state.str.startswith('PRE_ONSET_P') & (f.lead_time_to_onset_s>lo) & (f.lead_time_to_onset_s<=hi)
    y[pre]=1
    return y

def label_at_grid(f,e,shift=0):
    """Shift all survey intervals together: hypothetical clock offset, never optimized."""
    frames=[]
    for sid,g in f.groupby('subject',sort=False):
        es=e[e.subject==sid]
        lists=[[(r.event_id,r.event_start_ts+shift,r.event_end_ts+shift) for r in es.itertuples() if r.label_category in cats] for cats in [
            ['confirmed_high_stress','confirmed_moderate_stress'],['unconfirmed_algorithm_candidate'],['survey_confirmed_no_stress']]]
        d=pd.DataFrame([classify_prediction_timestamp(t,*lists) for t in g.pred_ts],index=g.index)
        frames.append(d)
    return pd.concat(frames).sort_index()

def exposure(f,mask=None):
    """Intervals [t,next original-grid timestamp), clipped to session end.
    Caller provides full cached grid (no deletion before computing intervals).
    """
    f=f.sort_values(KEY).copy()
    if mask is not None: mask=np.asarray(mask)[f.index]
    nxt=f.groupby(['subject','session']).pred_ts.shift(-1)
    sessions=json.loads((BASE/'dataset_manifest.json').read_text(encoding='utf-8'))['sessions']
    ends={s['session']:s['end_ts'] for s in sessions}
    tail=f.session.map(ends)-f.pred_ts
    step=(nxt-f.pred_ts).fillna(tail).clip(lower=0,upper=60.)
    return pd.Series(step.to_numpy(),index=f.index).sort_index().to_numpy()

def strict_events(p,e,task='P1',pred_col='pred',merge_gap=180.,episode_mode='any_hit'):
    """Task-specific warning matching, actual positive timestamps, one-to-one episodes.
    any_hit: continued warning counts when at least one actual positive time is in horizon.
    trigger: only first timestamp of merged warning counts; intended alert issuance metric.
    """
    lo,hi=HORIZONS[task]
    e=e.copy()
    all_times={k:np.sort(g.pred_ts.to_numpy()) for k,g in p.groupby(['subject','session'],sort=False)}
    pos_times={k:np.sort(g.pred_ts.to_numpy()) for k,g in p[p[pred_col]==1].groupby(['subject','session'],sort=False)}
    hits=[]
    for r in e.itertuples():
        times=pos_times.get((r.subject,r.matched_session),np.array([]));a,b=np.searchsorted(times,[r.event_start_ts-hi,r.event_start_ts-lo])
        total=all_times.get((r.subject,r.matched_session),np.array([]));aa,bb=np.searchsorted(total,[r.event_start_ts-hi,r.event_start_ts-lo])
        hits.append({'event_id':r.event_id,'subject':r.subject,'session':r.matched_session,'detected':int(b>a),
                     'lead_s':r.event_start_ts-times[a] if b>a else np.nan,'n_available_windows':int(bb-aa)})
    eps=[]
    for (sid,sess),g in p.groupby(['subject','session'],sort=False):
        q=g[g[pred_col]==1].sort_values('pred_ts')
        groups=np.cumsum(np.r_[True,np.diff(q.pred_ts)>merge_gap]) if len(q) else []
        for _,a in q.groupby(groups):
            times=a.pred_ts.to_numpy()
            if episode_mode=='trigger': times=times[:1]
            candidates=[]
            for r in e[(e.subject==sid)&(e.matched_session==sess)].itertuples():
                if ((times>=r.event_start_ts-hi)&(times<r.event_start_ts-lo)).any(): candidates.append(r.event_id)
            eps.append({'subject':sid,'session':sess,'alert_start_ts':a.pred_ts.min(),'alert_end_ts':a.pred_ts.max(),
                        'event_ids':candidates,'has_negative':bool(a.y_true.eq(0).any())})
    assigned=set(); n_hit=0; fp=0; neutral=0
    for ep in eps:
        available=[eid for eid in ep['event_ids'] if eid not in assigned]
        if available: assigned.add(available[0]);n_hit+=1
        elif ep['has_negative']: fp+=1
        else: neutral+=1
    hours=p.loc[p.y_true.eq(0),'exposure_s'].sum()/3600 if 'exposure_s' in p else np.nan
    ev=pd.DataFrame(hits)
    n_det=len(assigned) if episode_mode=='trigger' else ev.detected.sum() if len(ev) else 0
    return {'n_events':len(e),'detected_events':int(n_det),'event_recall':n_det/len(e) if len(e) else np.nan,
            'false_episodes':fp,'negative_hours':hours,'fa_hr':fp/hours if hours else np.nan,
            'hit_episodes':n_hit,'duplicate_or_censored_episodes':neutral,'episode_precision':n_hit/len(eps) if eps else np.nan},ev

def summary(p,e,task='P1',name='',pred='pred',prob='prob'):
    subs=[]
    for sid,g in p.groupby('subject',sort=True):
        m=compute_window_metrics(g.y_true,g[prob],g[pred]);m['subject']=sid
        em,_=strict_events(g,e[e.subject==sid],task,pred)
        subs.append({'experiment':name,**m,**em})
    s=pd.DataFrame(subs)
    wm=compute_window_metrics(p.y_true,p[prob],p[pred])
    em,_=strict_events(p,e,task,pred)
    row={'experiment':name,'task':task,'n':len(p),'positive':int(p.y_true.sum()),'prevalence':p.y_true.mean(),
         'macro_auroc':s.auroc.mean(),'macro_auprc':s.auprc.mean(),'median_auroc':s.auroc.median(),
         **{'pooled_'+k:wm[k] for k in ['auroc','auprc','precision','recall','fpr','brier','tp','fp','tn','fn']},**em}
    return row,s

def inventory():
    snapshot();f,e,m=data();p=read('predictions');f['exposure_s']=exposure(f)
    jsave({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,
           'sklearn':__import__('sklearn').__version__,'lightgbm':__import__('lightgbm').__version__,
           'seed':42,'date_kst':'2026-10-26','n_features':len(SENSOR_FEATURE_MAP['all'])},'environment')
    rows=[]; subjects=[]
    for model,g in p.groupby('model',sort=False):
        g=g.merge(f[KEY+['exposure_s']],on=KEY,validate='one_to_one')
        r,s=summary(g,events_for(e,'P1'),name=model);rows.append(r);subjects.append(s)
        old=evaluate_event_level_forecasting(g,events_for(e,'P1'),max_valid_lead_s=180,min_valid_lead_s=60)
        jsave(old,'reproduced_legacy_'+model)
    save(pd.DataFrame(rows),'cached_model_reproduction');save(pd.concat(subjects),'cached_subject_reproduction')
    sf=parse_survey_results_xlsx(RAW/'SurveyResults.xlsx')
    save(sf[['excel_row_index','subject','Start time','End time','date','duration','dt_start_local','dt_end_local','event_start_ts','event_end_ts','stress_level_raw']],'survey_values')
    ns='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with zipfile.ZipFile(RAW/'SurveyResults.xlsx') as z:
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        jsave({'sheets':[s.attrib for s in wb.findall('.//'+ns+'sheet')],
               'formula_cells':[c.attrib['r'] for c in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//'+ns+'c') if c.find(ns+'f') is not None],
               'all_start_seconds_zero':bool(sf.dt_start_local.dt.second.eq(0).all()),
               'all_end_seconds_zero':bool(sf.dt_end_local.dt.second.eq(0).all()),
               'max_excel_rounding_error_s':float(np.max(np.abs(sf['Start time'].astype(float)*86400-np.round(sf['Start time'].astype(float)*86400)))),
               'exact_duplicate_intervals':int(sf.duplicated(['subject','event_start_ts','event_end_ts']).sum()),
               'duplicate_onsets':int(sf.duplicated(['subject','event_start_ts']).sum())},'survey_structure')
    regenerated=label_at_grid(f,e)
    save(pd.DataFrame([{'field':c,'mismatches':int((regenerated[c].fillna('')!=f[c].fillna('')).sum())} for c in ['state','y_P0','y_P1','y_P2','target_event_id']]),'label_cache_consistency')
    save(f.groupby('state').agg(windows=('state','size'),qc_valid=('qc_common_valid','sum')).reset_index(),'state_counts')
    sess=pd.DataFrame(m['sessions']);save(sess,'original_session_manifest_copy')
    audit=[]
    for r in e.itertuples():
        es=e[(e.subject==r.subject)&e.label_category.isin(['confirmed_high_stress','confirmed_moderate_stress'])&(e.event_id!=r.event_id)]
        future_overlap=((es.event_start_ts<=r.event_start_ts)&(es.event_end_ts>=r.event_start_ts)).sum()
        q=f[(f.subject==r.subject)&(f.target_event_id==r.event_id)&(f.y_P1==1)]
        overlap180=0;overlap300=0
        for a in q.itertuples():
            overlap180+=int(((es.event_start_ts<a.pred_ts)&(es.event_end_ts>a.pred_ts-180)).any())
            overlap300+=int(((es.event_start_ts<a.pred_ts)&(es.event_end_ts>a.pred_ts-300)).any())
        audit.append({'event_id':r.event_id,'subject':r.subject,'is_confirmed':r.label_category in ['confirmed_high_stress','confirmed_moderate_stress'],
                      'eligible_p1':r.p1_forecast_eligible,'other_event_at_onset':future_overlap,'p1_windows':len(q),
                      'p1_qc_windows':int(q.qc_common_valid.sum()),'p1_feature_overlap_previous180':overlap180,
                      'p1_feature_overlap_previous300':overlap300})
    save(pd.DataFrame(audit),'event_neighborhood_audit')
    cal=apply_session_causal_calibration(f,method='low_motion')
    blind=f.copy();blind['state']='UNAVAILABLE_AT_INFERENCE'
    bc=apply_session_causal_calibration(blind,method='low_motion')
    changed=np.any(np.abs(cal[CALIBRATED_PHYSIO_FEATURES].to_numpy()-bc[CALIBRATED_PHYSIO_FEATURES].to_numpy())>1e-10,axis=1)
    save(f.loc[changed,KEY+['state','y_P1']],'label_dependent_calibration_rows')
    jsave({'changed_all_windows':int(changed.sum()),'changed_p1_eligible':int((changed&f.qc_common_valid&f.y_P1.isin([0,1])).sum()),
           'sessions':int(f.loc[changed].session.nunique()),'cache_vs_current_max_abs':float(np.max(np.abs(cal[CALIBRATED_PHYSIO_FEATURES].to_numpy()-f[CALIBRATED_PHYSIO_FEATURES].to_numpy()))),
           'negative_qc_excluded':int(((f.y_P1==0)&~f.qc_common_valid).sum()),
           'positive_qc_excluded':int(((f.y_P1==1)&~f.qc_common_valid).sum()),
           'ibi_invalid_fraction':float((f.ibi_valid_flag==0).mean()),
           'feature_missing_fraction':float(f[SENSOR_FEATURE_MAP['all']].isna().to_numpy().mean())},'calibration_leak_audit')
    split=read('splits'); sp=[]
    for r in split.itertuples():
        tr=set(r.train_subjects.split(';'));va=set(r.val_subjects.split(';'))
        sp.append({'fold':r.fold,'test':r.test_subject,'test_in_training':r.test_subject in tr,'test_in_validation':r.test_subject in va,'train_val_overlap':len(tr&va)})
    save(pd.DataFrame(sp),'split_audit')
    lr=p[p.model=='LR'].merge(f[KEY+['exposure_s']],on=KEY)
    parts=[]
    for sid,g in lr.groupby('subject'):
        z=f[f.subject==sid];neg=g[g.y_true==0];pos=g[g.y_true==1]
        parts.append({'subject':sid,'eda_median':z.eda_mean.median(),'hr_median':z.hr_mean_bpm.median(),'motion_median':z.acc_motion_change_per_s.median(),
                      'qc_excluded_fraction':1-z.qc_common_valid.mean(),'ibi_invalid_fraction':1-z.ibi_valid_flag.mean(),
                      'prevalence':g.y_true.mean(),'mean_prob':g.prob.mean(),'negative_mean_prob':neg.prob.mean(),'positive_mean_prob':pos.prob.mean(),
                      'threshold':g.threshold.iloc[0]})
    save(pd.DataFrame(parts).merge(pd.concat(subjects).query("experiment=='LR'"),on='subject'),'subject_profiles')
    dec=[]
    for sa,a in lr.groupby('subject'):
        for sb,b in lr.groupby('subject'):
            pa=a[a.y_true==1].prob.to_numpy();nb=b[b.y_true==0].prob.to_numpy()
            v=(pa[:,None]>nb).mean()+.5*(pa[:,None]==nb).mean()
            dec.append({'positive_subject':sa,'negative_subject':sb,'n_pairs':len(pa)*len(nb),'pair_auroc':v,'within_subject':sa==sb})
    save(pd.DataFrame(dec),'pooled_auc_pair_decomposition')
    # Leave-one-participant-out influence is a reaggregation, not selective reporting.
    inf=[]
    ss=pd.concat(subjects).query("experiment=='LR'")
    for sid in ss.subject:
        v=ss[ss.subject!=sid]
        inf.append({'omitted_subject':sid,'macro_auroc_remaining':v.auroc.mean(),'macro_auprc_remaining':v.auprc.mean()})
    save(pd.DataFrame(inf),'participant_influence')
    print('inventory complete',pd.DataFrame(rows)[['experiment','macro_auroc','pooled_auroc','event_recall','fa_hr']].to_string(index=False),flush=True)

def raw_audit():
    f,e,m=data();sf=parse_survey_results_xlsx(RAW/'SurveyResults.xlsx');sessions=pd.DataFrame(m['sessions'])
    headers=[];profiles=[];refs=[];checks=[];archive_hashes=[]
    for n,s in enumerate(sessions.itertuples()):
        zp=RAW/s.subject/s.zip_name
        archive_hashes.append({'path':str(zp.relative_to(ROOT)),'bytes':zp.stat().st_size,'sha256':digest(zp)})
        with zipfile.ZipFile(zp) as z:
            head={'subject':s.subject,'session':s.session}
            for sensor in ['EDA','BVP','ACC','HR','TEMP','IBI']:
                with z.open(sensor+'.csv') as stream:
                    line=stream.readline().decode().strip()
                    head[sensor+'_start_ts']=float(line.split(',')[0]) if line else np.nan
                    if sensor!='IBI': head[sensor+'_rate']=float(stream.readline().decode().split(',')[0])
            head['HR_offset_s']=head['HR_start_ts']-head['EDA_start_ts']
            head['IBI_offset_s']=head['IBI_start_ts']-head['EDA_start_ts']
            headers.append(head)
            if not s.has_survey_annotation: continue
            # EDA/HR full series for descriptive trajectories; other high-rate signals only initial 300s.
            t0,fs,eda=parse_e4_csv_bytes(z.read('EDA.csv'))
            h0,hfs,hr=parse_e4_csv_bytes(z.read('HR.csv'))
            arrays={'EDA':eda,'HR':hr}
            for sensor in ['BVP','ACC']:
                count=int(head[sensor+'_rate']*300)
                with z.open(sensor+'.csv') as stream:
                    lines=[stream.readline() for _ in range(count+2)]
                _,_,arrays[sensor]=parse_e4_csv_bytes(b''.join(lines),is_acc=sensor=='ACC')
            sd={'start_ts':t0,'start_timestamps':{k:head[k+'_start_ts'] for k in ['EDA','BVP','ACC','HR','IBI']},
                'rates':{k:head[k+'_rate'] for k in ['EDA','BVP','ACC','HR']},**arrays,'IBI':np.empty((0,2))}
            # IBI isn't used in the nine calibrated baseline columns. Do not synthesize HRV.
            for duration in [60.,180.,300.]:
                if duration>s.dur_common_s: continue
                if duration==60.:
                    from src.features.core import basic,spectrum
                    from scipy.signal import find_peaks
                    ed=eda[:int(fs*duration)];bp=arrays['BVP'][:int(head['BVP_rate']*duration)]
                    hh=hr[:max(0,int((duration-head['HR_offset_s'])*hfs))]
                    ff=basic(ed,fs,'eda_');ff.update(basic(bp,head['BVP_rate'],'ppg_'))
                    ff['eda_positive_diff_sum']=np.maximum(np.diff(ed),0).sum()
                    ff['eda_scr_candidate_count']=len(find_peaks(ed,prominence=.05,distance=int(fs))[0])
                    ff['ppg_peak_hz'],ff['ppg_band_fraction']=spectrum(bp,head['BVP_rate'],.7,3.5)
                    ff['hr_mean_bpm']=float(np.mean(hh)) if len(hh) else ff['ppg_peak_hz']*60
                else:
                    ff=extract_nurse_window_features(sd,duration,window_s=duration)
                obs=compute_log_base_features(pd.DataFrame([ff]));obs['hr_bpm_raw']=ff['hr_mean_bpm']
                refs.append({'subject':s.subject,'session':s.session,'duration_s':duration,
                             **dict(zip(CALIBRATED_PHYSIO_FEATURES,obs.iloc[0].to_numpy()))})
            # Verify cached features against raw at t=180 (all sessions); non-HRV fields only.
            q=f[f.session.eq(s.session)&f.subject.eq(s.subject)].iloc[0]
            ff=extract_nurse_window_features(sd,float(q.end_s),window_s=180.)
            cols=[c for c in SENSOR_FEATURE_MAP['all'] if c in ff and not c.startswith('ibi_')]
            rel=np.array([abs(float(ff[c])-float(q[c]))/(1+abs(float(q[c]))) for c in cols])
            # Mutate available future data, preserve samples before prediction time.
            changed={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in sd.items()}
            for sensor in ['EDA','BVP','ACC','HR']:
                offset=head[sensor+'_start_ts']-t0
                cut=max(0,int(np.ceil((q.end_s-offset)*head[sensor+'_rate']-1e-10)))
                changed[sensor][cut:]=9999
            mut=extract_nurse_window_features(changed,float(q.end_s),window_s=180.)
            checks.append({'subject':s.subject,'session':s.session,'n_replayed_features':len(cols),'max_relative_cache_error':float(rel.max()),
                           'future_mutation_max_abs':max(abs(ff[c]-mut[c]) for c in cols if np.isfinite(ff[c]))})
            ses_events=e[(e.subject==s.subject)&(e.matched_session==s.session)]
            for ev in ses_events.itertuples():
                for left in range(-600,301,60):
                    a=ev.event_start_ts+left;b=a+60
                    if a<t0 or b>s.end_ts: continue
                    ed=eda[int(round((a-t0)*fs)):int(round((b-t0)*fs))]
                    hh=hr[max(0,int(round((a-h0)*hfs))):max(0,int(round((b-h0)*hfs)))]
                    profiles.append({'event_id':ev.event_id,'subject':s.subject,'category':ev.label_category,'relative_start_s':left,
                                     'eda_mean':float(np.nanmean(ed)),'hr_mean':float(np.nanmean(hh)) if len(hh) else np.nan,
                                     'eda_nonfinite_fraction':float(1-np.isfinite(ed).mean())})
        if (n+1)%50==0: print('raw',n+1,'/',len(sessions),flush=True)
    save(pd.DataFrame(headers),'raw_sensor_headers');save(pd.DataFrame(refs),'raw_initial_baselines')
    save(pd.DataFrame(profiles),'raw_event_60s_trajectories');save(pd.DataFrame(checks),'raw_feature_replay');save(pd.DataFrame(archive_hashes),'raw_archive_hashes')
    tz=[]
    # Alternate time conventions audited for coverage only; no accuracy-based offset selection.
    for mode in ['America/Chicago','fixed_CST_UTCminus6','UTC','offset_minus60','offset_plus60','offset_minus3600','offset_plus3600']:
        for r in sf.itertuples():
            from datetime import timezone,timedelta
            ts=r.event_start_ts;te=r.event_end_ts
            if mode=='fixed_CST_UTCminus6':
                ts=r.dt_start_local.replace(tzinfo=timezone(timedelta(hours=-6))).timestamp();te=r.dt_end_local.replace(tzinfo=timezone(timedelta(hours=-6))).timestamp()
            elif mode=='UTC': ts=r.dt_start_local.replace(tzinfo=timezone.utc).timestamp();te=r.dt_end_local.replace(tzinfo=timezone.utc).timestamp()
            elif mode.startswith('offset_'):
                offset=float(mode.split('_')[1].replace('minus','-').replace('plus',''));ts+=offset;te+=offset
            se=sessions[sessions.subject.eq(r.subject)]
            ov=np.maximum(0,np.minimum(se.end_ts,te)-np.maximum(se.start_ts,ts)).sum()
            covered=((se.start_ts<=ts)&(se.end_ts>=ts)).any()
            tz.append({'mode':mode,'subject':r.subject,'event_id':r.event_id,'interval_overlap_s':ov,'onset_covered':covered})
    save(pd.DataFrame(tz),'timezone_coverage_per_event')
    save(pd.DataFrame(tz).groupby('mode').agg(overlap_events=('interval_overlap_s',lambda v:(v>0).sum()),onset_covered=('onset_covered','sum'),overlap_hours=('interval_overlap_s',lambda v:v.sum()/3600)).reset_index(),'timezone_coverage_summary')
    print('raw audit complete',flush=True)

def fit_loso(f,e,name,task='P1',model='LR',cols=None,weight='balanced',threshold_mode='legacy',mask_extra=None,time_weighted=False):
    """Same fixed 15 outer splits, inner seed=42+fold. No test-based choices.
    Full-grid scores saved for causal policy replay and label sensitivity without re-training.
    """
    path=OUT/'predictions'/f'{name}.csv.gz'
    if path.exists():
        print('reuse',name,flush=True);return pd.read_csv(path,dtype=DT,low_memory=False)
    tic=time.perf_counter();cols=cols or SENSOR_FEATURE_MAP['all']
    y=task_labels(f,task);mask=f.qc_common_valid.to_numpy()&(y>=0)
    if mask_extra is not None:mask &= mask_extra
    work=f.copy();work['y_true']=y
    allp=[];meta=[];coefs=[]
    for fold,sid in enumerate(sorted(f.subject.unique())):
        dev=work[mask&work.subject.ne(sid)];te=work[work.subject.eq(sid)].copy()
        gss=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42+fold)
        tr,va=next(gss.split(dev,dev.y_true,dev.subject));train=dev.iloc[tr];val=dev.iloc[va]
        assert sid not in set(train.subject)|set(val.subject)
        def make():
            pipe=build_forecast_model(model,n_jobs=2)
            if model=='LR':pipe.set_params(clf__class_weight=None if time_weighted else weight)
            return pipe
        def fit(pipe,g):
            if time_weighted:
                sw=g.exposure_s.to_numpy()/60
                cw=sw.copy()
                if weight=='balanced':
                    for cls in [0,1]:cw[g.y_true.to_numpy()==cls]*=sw.sum()/(2*sw[g.y_true.to_numpy()==cls].sum())
                pipe.fit(g[cols],g.y_true,clf__sample_weight=cw,scaler__sample_weight=sw)
            else:pipe.fit(g[cols],g.y_true)
        inner=make();fit(inner,train);pv=inner.predict_proba(val[cols])[:,1]
        if threshold_mode=='legacy' and not time_weighted:thr=select_validation_threshold(val.y_true.to_numpy(),pv)
        else:
            grid=np.linspace(.20,.85,66) if threshold_mode=='legacy' else np.unique(np.r_[np.linspace(.001,.1,100),np.linspace(.11,.95,85)])
            scores=[f1_score(val.y_true,pv>=t,zero_division=0,sample_weight=val.exposure_s if time_weighted else None) for t in grid]
            thr=grid[np.flatnonzero(np.asarray(scores)==max(scores))[-1]]
        pipe=make();fit(pipe,dev);pt=pipe.predict_proba(te[cols])[:,1]
        te['prob']=pt;te['threshold']=thr;te['pred']=(pt>=thr).astype(int);te['fold']=fold
        # Fixed class-prior odds correction diagnostic (monotone, no test-label access).
        pi=dev.y_true.mean();te['prob_prior_corrected']=expit(logit(np.clip(pt,1e-9,1-1e-9))+logit(pi)) if weight=='balanced' else pt
        # Sigmoid calibration on inner validation only; refit shift limitation recorded in report.
        if len(np.unique(val.y_true))==2:
            cal=LogisticRegression(C=1e3,max_iter=1000).fit(logit(np.clip(pv,1e-7,1-1e-7))[:,None],val.y_true)
            te['prob_sigmoid']=cal.predict_proba(logit(np.clip(pt,1e-7,1-1e-7))[:,None])[:,1]
        allp.append(te[KEY+['state','target_event_id','lead_time_to_onset_s','start_s','end_s','qc_common_valid','y_true','prob','threshold','pred','fold','prob_prior_corrected','prob_sigmoid','exposure_s','acc_motion_change_per_s','acc_unchanged_fraction','acc_dynamic_energy']])
        meta.append({'experiment':name,'fold':fold,'test_subject':sid,'train_subjects':';'.join(sorted(train.subject.unique())),
                     'val_subjects':';'.join(sorted(val.subject.unique())),'n_train':len(train),'n_val':len(val),'n_dev':len(dev),
                     'n_test':int(mask[work.subject.eq(sid)].sum()),'threshold':thr,'dev_prevalence':pi,'validation_f1':f1_score(val.y_true,pv>=thr,zero_division=0)})
        if model=='LR':
            coefs.extend([{'experiment':name,'fold':fold,'test_subject':sid,'feature':c,'coefficient':v} for c,v in zip(cols,pipe['clf'].coef_[0])])
    result=pd.concat(allp,ignore_index=True)
    path.parent.mkdir(parents=True,exist_ok=True);result.to_csv(path,index=False,compression='gzip')
    save(pd.DataFrame(meta),'splits_'+name)
    if coefs:save(pd.DataFrame(coefs),'coefficients_'+name)
    jsave({'experiment':name,'seconds':time.perf_counter()-tic,'task':task,'model':model,'weight':weight,'threshold_mode':threshold_mode,'time_weighted':time_weighted,'columns':cols},'config_'+name)
    print('trained',name,round(time.perf_counter()-tic,1),'s',flush=True)
    return result

def baseline_variants(f):
    blind=f.copy();blind['state']='UNAVAILABLE_AT_INFERENCE'
    corrected=apply_session_causal_calibration(blind,method='low_motion');corrected['state']=f.state
    yield 'labelblind_same_method',corrected
    # Same columns across methods, label-independent nine physiological observables.
    obs=compute_log_base_features(f);obs['hr_bpm_raw']=f.hr_mean_bpm
    raw=f.copy();raw[CALIBRATED_PHYSIO_FEATURES]=obs.to_numpy()
    yield 'raw',raw
    refs=pd.read_csv(OUT/'raw_initial_baselines.csv',dtype=DT)
    for dur in [60,180,300]:
        z=raw.copy()
        for (sid,sess),g in z.groupby(['subject','session'],sort=False):
            rr=refs[(refs.subject==sid)&(refs.session==sess)&(refs.duration_s==dur)]
            if rr.empty:continue
            ref=rr[CALIBRATED_PHYSIO_FEATURES].iloc[0].to_numpy()
            completed=g.end_s>=dur
            z.loc[g.index[completed],CALIBRATED_PHYSIO_FEATURES]=g.loc[completed,CALIBRATED_PHYSIO_FEATURES].to_numpy()-ref
            # Before 5-minute baseline completes, first 180s only (observed at all t>=180).
            if (~completed).any():
                early=refs[(refs.subject==sid)&(refs.session==sess)&(refs.duration_s==180)][CALIBRATED_PHYSIO_FEATURES].iloc[0].to_numpy()
                z.loc[g.index[~completed],CALIBRATED_PHYSIO_FEATURES]=g.loc[~completed,CALIBRATED_PHYSIO_FEATURES].to_numpy()-early
        yield f'actual_{dur}s',z
    # Quiet initial windows only, with explicit time bound and causal prefix.
    quiet=raw.copy()
    for _,g in raw.groupby(['subject','session'],sort=False):
        vals=g[CALIBRATED_PHYSIO_FEATURES].to_numpy();ends=g.end_s.to_numpy();jerk=g.acc_motion_change_per_s.to_numpy()
        deltas=vals.copy()
        for i in range(len(g)):
            candidates=(ends<=min(300,ends[i]))&(jerk<=38)
            ref=vals[candidates].mean(axis=0) if candidates.any() else vals[0]
            deltas[i]-=ref
        quiet.loc[g.index,CALIBRATED_PHYSIO_FEATURES]=deltas
    yield 'quiet_initial300s',quiet

def train():
    f,e,m=data();f['exposure_s']=exposure(f)
    # Exact existing configuration first; leak correction changes only state access in calibration.
    fit_loso(f,e,'legacy_reproduction')
    fit_loso(f,e,'legacy_matched_threads')
    variants=dict(baseline_variants(f));clean=variants['labelblind_same_method']
    fit_loso(clean,e,'labelblind_same_method')
    # All further comparisons share the label-independent version; legacy thresholds fixed.
    for task in HORIZONS:
        if task=='P1':continue
        fit_loso(clean,e,'horizon_'+task,task=task)
    # Five-minute comparison also constrained to a common set of 171 eligible events.
    common=set(events_for(e,'P2').event_id)
    common_mask=~f.state.str.startswith('PRE_ONSET_P')|f.target_event_id.isin(common)
    for task in ['P0','P1-A','P1-B','P1','P2']:
        fit_loso(clean,e,'common171_'+task,task=task,mask_extra=common_mask.to_numpy())
    for model in ['RF','LightGBM','Dummy']:
        fit_loso(clean,e,'model_'+model,model=model)
    # Isolate weighting under same legacy grid and same expanded grid.
    fit_loso(clean,e,'unweighted_legacy_grid',weight=None)
    fit_loso(clean,e,'balanced_full_grid',threshold_mode='full')
    fit_loso(clean,e,'unweighted_full_grid',weight=None,threshold_mode='full')
    for sensor in ['eda','ppg','acc','eda_ppg']:
        fit_loso(clean,e,'sensor_'+sensor,cols=SENSOR_FEATURE_MAP[sensor])
    fit_loso(clean,e,'simple_eda_slope',cols=['eda_recent60_minus_prev120'])
    fit_loso(clean,e,'simple_hr_change',cols=['hr_delta_recent60_bpm'])
    reduced=list(CALIBRATED_PHYSIO_FEATURES)+['eda_recent60_minus_prev120','eda_log_slope','eda_rise_fraction','hr_slope_bpm_per_s','hr_delta_recent60_bpm','acc_motion_change_per_s','acc_unchanged_fraction']
    for name,v in variants.items():
        fit_loso(v,e,'calibration_'+name,cols=reduced)
    # Confirmed-zero-only control tests annotation status, not unbiased deployment accuracy.
    zero_mask=f.state.ne('VERIFIED_NON_EVENT').to_numpy()
    fit_loso(clean,e,'confirmed_zero_only',mask_extra=zero_mask)
    # Previous-event recovery contamination removed from all classes symmetrically.
    safe=np.ones(len(f),bool)
    for sid,g in f.groupby('subject'):
        ev=e[e.subject.eq(sid)&e.label_category.isin(['confirmed_high_stress','confirmed_moderate_stress'])]
        for r in ev.itertuples():
            safe[g.index[(g.pred_ts>r.event_start_ts)&(g.pred_ts-300<r.event_end_ts+900)]]=False
    fit_loso(clean,e,'clean_history300_refractory900',mask_extra=safe)
    fit_loso(clean,e,'time_weighted_LR',time_weighted=True)
    print('training phase complete',flush=True)

def load_pred(name,f=None):
    p=pd.read_csv(OUT/'predictions'/f'{name}.csv.gz',dtype=DT,low_memory=False)
    if f is not None:
        # Exposure convention is recomputed from immutable full grid, including exact tail.
        p=p.drop(columns=['exposure_s']).merge(f[KEY+['exposure_s']],on=KEY,validate='one_to_one')
    return p

def participant_ci(s,metric,seed=743):
    v=s[metric].dropna().to_numpy();rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(v,size=(5000,len(v)),replace=True),axis=1)
    return {'metric':metric,'estimate':float(v.mean()),'lower95':float(np.quantile(boot,.025)),'upper95':float(np.quantile(boot,.975)),
            'unit':'participant','n_participants':len(v),'n_resamples':5000}

def analyze():
    f,e,m=data();f['exposure_s']=exposure(f)
    allrows=[];allsub=[];ci=[];evs=events_for(e,'P1')
    configs=sorted(OUT.glob('config_*.json'))
    for config in configs:
        c=json.loads(config.read_text());name=c['experiment'];task=c['task']
        p=load_pred(name,f);p=p[p.qc_common_valid & p.y_true.isin([0,1])].copy();es=events_for(e,task)
        if name.startswith('common171_'):
            ids=set(events_for(e,'P2').event_id);p=p[(p.y_true==0)|p.target_event_id.isin(ids)];es=es[es.event_id.isin(ids)]
        elif name=='confirmed_zero_only':p=p[p.state!='VERIFIED_NON_EVENT']
        elif name=='clean_history300_refractory900':
            keep=np.ones(len(p),bool)
            for r in e[e.label_category.isin(['confirmed_high_stress','confirmed_moderate_stress'])].itertuples():
                keep &= ~((p.subject==r.subject)&(p.pred_ts>r.event_start_ts)&(p.pred_ts-300<r.event_end_ts+900))
            p=p[keep]
        r,s=summary(p,es,task,name);allrows.append(r);allsub.append(s)
        for metric in ['auroc','auprc','fpr','recall','event_recall']:
            ci.append({'experiment':name,**participant_ci(s,metric)})
    rows=pd.DataFrame(allrows);subs=pd.concat(allsub)
    save(rows,'controlled_experiments');save(subs,'controlled_per_subject');save(pd.DataFrame(ci),'participant_bootstrap_ci')
    p=load_pred('labelblind_same_method',f);evalmask=p.qc_common_valid & p.y_true.isin([0,1]);q=p[evalmask].copy()
    # Exact re-training reproduction against preserved OOF probabilities.
    old=read('predictions');old=old[old.model=='LR'];rep=load_pred('legacy_reproduction',f)
    z=old[KEY+['prob','threshold','pred']].merge(rep[KEY+['prob','threshold','pred']],on=KEY,suffixes=('_old','_new'),validate='one_to_one')
    jsave({'n':len(z),'max_abs_prob_error':float((z.prob_old-z.prob_new).abs().max()),'prediction_disagreements':int((z.pred_old!=z.pred_new).sum()),'threshold_max_error':float((z.threshold_old-z.threshold_new).abs().max())},'retraining_reproduction')
    # Preserve evaluator semantics to reproduce policy and gate claims exactly.
    cached_policy=[]
    for policy in ['val_opt','hysteresis','duration_2win','cooldown_300s','fixed_05']:
        groups=[]
        for _,g in old.groupby(['subject','session'],sort=False):
            g=g.sort_values('pred_ts').copy();g['pr']=apply_alert_policy_to_session(g.prob.to_numpy(),g.end_s.to_numpy(),g.threshold.iloc[0],policy=policy,df_session=g);groups.append(g)
        cp=pd.concat(groups)
        cached_policy.append({'policy':policy,**evaluate_event_level_forecasting(cp,evs,pred_col='pr',max_valid_lead_s=180,min_valid_lead_s=60)})
    save(pd.DataFrame(cached_policy),'legacy_alert_policy_reproduction')
    cached_gate=[]
    for gate in ['none','legacy_hard','soft_dynamic']:
        groups=[]
        for sid,g in old.groupby('subject',sort=False):
            g=g.copy();pg,_,pr=apply_dynamic_motion_gate(g.prob.to_numpy(),g,mode=gate,base_threshold=g.threshold.iloc[0]);g['pg']=pg;g['pr']=pr;groups.append(g)
        cg=pd.concat(groups);cg['quartile']=pd.qcut(cg.acc_motion_change_per_s.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
        em=evaluate_event_level_forecasting(cg,evs,pred_col='pr',max_valid_lead_s=180,min_valid_lead_s=60)
        cached_gate.append({'gate':gate,'q4_negative_fpr':cg.loc[(cg.y_true==0)&(cg.quartile=='Q4'),'pr'].mean(),**em})
    save(pd.DataFrame(cached_gate),'legacy_motion_gate_reproduction')
    # Frozen-model timing sensitivity; hypothetical offsets not corrected ground truth.
    sensitivity=[];changes=[];states=[]
    for shift in [-60,-30,0,30,60]:
        lab=label_at_grid(f,e,shift);zz=p.merge(f[KEY].assign(_row=f.index),on=KEY,validate='one_to_one').sort_values('_row')
        zz['state']=lab.state.to_numpy();zz['y_true']=lab.y_P1.to_numpy();zz['target_event_id']=lab.target_event_id.to_numpy();zz['lead_time_to_onset_s']=lab.lead_time_to_onset_s.to_numpy()
        ee=evs.copy();ee['event_start_ts']+=shift;ee['event_end_ts']+=shift
        se=zz[zz.qc_common_valid&zz.y_true.isin([0,1])]
        r,s=summary(se,ee,name=f'frozen_offset_{shift:+d}s');sensitivity.append(r)
        changes.append({'offset_s':shift,'changed_labels_all':int((lab.y_P1.to_numpy()!=f.y_P1.to_numpy()).sum()),
                        'lost_original_positives':int(((f.y_P1==1)&(lab.y_P1!=1)).sum()),'new_positives':int(((f.y_P1!=1)&(lab.y_P1==1)).sum()),
                        'positive_windows':int((lab.y_P1==1).sum()),'changed_to_direct_negative':int(((f.y_P1==1)&(lab.y_P1==0)).sum())})
        states.append(lab.y_P1.to_numpy())
    save(pd.DataFrame(sensitivity),'timing_offset_sensitivity');save(pd.DataFrame(changes),'timing_label_changes')
    conserv=[]
    for radius,indices in [(30,[1,2,3]),(60,[0,1,2,3,4])]:
        arr=np.array([states[i] for i in indices]);stable=(arr==arr[0]).all(axis=0)&(arr[0]>=0)
        zz=p.merge(f[KEY].assign(_row=f.index),on=KEY).sort_values('_row');zz=zz[stable&zz.qc_common_valid.to_numpy()]
        if zz.y_true.nunique()==2:
            r,s=summary(zz,evs,name=f'stable_plusminus{radius}');conserv.append(r)
        else:conserv.append({'experiment':f'stable_plusminus{radius}','n':len(zz),'positive':int(zz.y_true.eq(1).sum()),'macro_auroc':np.nan,'macro_auprc':np.nan,'event_recall':np.nan,'note':'No invariant P1-positive window under full timing uncertainty; discrimination undefined.'})
    save(pd.DataFrame(conserv),'conservative_timing_evaluation')
    # Data/annotation controls with fixed scores only, all definitions explicit.
    controls=[]
    for name,negative in [('all_background',NEG),('confirmed_zero_only',['CONFIRMED_NO_STRESS_0']),('unverified_background_only',['VERIFIED_NON_EVENT'])]:
        sub=q[(q.y_true==1)|q.state.isin(negative)];r,_=summary(sub,evs,name=name);controls.append(r)
    save(pd.DataFrame(controls),'negative_label_source_control')
    matched_subjects=[sid for sid,g in q.groupby('subject') if g.y_true.eq(1).any() and g.state.eq('CONFIRMED_NO_STRESS_0').any() and g.state.eq('VERIFIED_NON_EVENT').any()]
    negmatched=[];negsub=[]
    for name,negative in [('confirmed_zero',['CONFIRMED_NO_STRESS_0']),('unverified_background',['VERIFIED_NON_EVENT'])]:
        zz=q[q.subject.isin(matched_subjects)&((q.y_true==1)|q.state.isin(negative))]
        ee=evs[evs.subject.isin(matched_subjects)];r,s=summary(zz,ee,name=name);negmatched.append(r);negsub.append(s)
    save(pd.DataFrame(negmatched),'matched_subject_negative_source_control');save(pd.concat(negsub),'matched_subject_negative_source_metrics')
    # Event denominators: original / exact dedup / actually observable / isolated.
    den=[];ed=[]
    for name,ee in [('original176',evs),('exact_interval_deduplicated',evs.drop_duplicates(['subject','event_start_ts','event_end_ts']))]:
        r,detail=strict_events(q,ee);detail['denominator']=name;ed.append(detail);den.append({'denominator':name,**r})
        avail=ee[ee.event_id.isin(detail.loc[detail.n_available_windows>0,'event_id'])]
        r,_=strict_events(q,avail);den.append({'denominator':name+'_observable',**r})
    save(pd.DataFrame(den),'event_denominator_controls');save(pd.concat(ed),'event_detection_details')
    # Gates use same probabilities, thresholds and evaluation rows. Compute both legacy
    # filtered stream score and full-grid causal score to expose state-skipping effects.
    gm=[];gq=[];gs=[];ge=[]
    edges=np.quantile(q.acc_motion_change_per_s,[0,.25,.5,.75,1])
    save(pd.DataFrame({'quantile':[0,.25,.5,.75,1],'acc_motion_change_per_s':edges}),'motion_quartile_edges')
    for source,whole in [('filtered',q),('full_stream',p)]:
        for gate in ['none','legacy_hard','soft_dynamic']:
            groups=[]
            for sid,g in whole.groupby('subject',sort=False):
                g=g.sort_values(['session','pred_ts']).copy();pg,th,pr=apply_dynamic_motion_gate(g.prob.to_numpy(),g,mode=gate,base_threshold=g.threshold.iloc[0]);g['pg']=pg;g['pr']=pr;groups.append(g)
            gated=pd.concat(groups);gated=gated[gated.qc_common_valid&gated.y_true.isin([0,1])].copy()
            gated['quartile']=pd.cut(gated.acc_motion_change_per_s,edges,labels=['Q1','Q2','Q3','Q4'],include_lowest=True)
            r,s=summary(gated,evs,name=source+'_'+gate,pred='pr',prob='pg');gm.append(r)
            _,detail=strict_events(gated,evs,pred_col='pr');detail['gate']=gate;detail['source']=source;ge.append(detail)
            for quart,g in gated.groupby('quartile',observed=True):
                wm=compute_window_metrics(g.y_true,g.pg,g.pr);gq.append({'source':source,'gate':gate,'quartile':quart,**wm})
            for (sid,quart),g in gated.groupby(['subject','quartile'],observed=True):
                gs.append({'source':source,'gate':gate,'subject':sid,'quartile':quart,**compute_window_metrics(g.y_true,g.pg,g.pr)})
    save(pd.DataFrame(gm),'motion_gate_controls');save(pd.DataFrame(gq),'motion_quartile_metrics');save(pd.DataFrame(gs),'motion_subject_quartile_metrics')
    ged=pd.concat(ge);save(ged,'motion_event_detection_details')
    # Event motion is median of its evaluated pre-onset windows, fixed before gate.
    event_motion=q[q.y_true==1].groupby('target_event_id').acc_motion_change_per_s.median()
    lost=ged[ged.source=='filtered'].pivot(index=['event_id','subject'],columns='gate',values='detected').reset_index()
    lost['median_pre_onset_motion']=lost.event_id.map(event_motion);lost['lost_by_soft_gate']=(lost['none']==1)&(lost.soft_dynamic==0)
    save(lost,'events_lost_by_motion_gate')
    corr=[]
    fq=f.merge(q[KEY+['prob']],on=KEY)
    for sid,g in fq.groupby('subject'):
        for feature in ['eda_mean','eda_recent60_minus_prev120','hr_mean_bpm','hr_delta_recent60_bpm','ppg_absdiff_per_s','prob']:
            for label,sub in [('all',g),('negative',g[g.y_P1==0]),('positive',g[g.y_P1==1])]:
                corr.append({'subject':sid,'subset':label,'feature':feature,'n':len(sub),'spearman_r':spearmanr(sub.acc_motion_change_per_s,sub[feature]).statistic})
    save(pd.DataFrame(corr),'motion_physiology_correlations')
    high=fq[fq.acc_motion_change_per_s>=edges[3]]
    save(high.groupby(['subject','y_P1']).agg(n=('y_P1','size'),eda=('eda_mean','median'),hr=('hr_mean_bpm','median'),motion=('acc_motion_change_per_s','median'),prob=('prob','median')).reset_index(),'high_motion_event_vs_background')
    # Alert policy comparison: same threshold, same model; policy on filtered vs full timeline.
    policies=[];perpol=[]
    for source,whole in [('filtered',q),('full_stream',p)]:
        for pol in ['val_opt','fixed_05','hysteresis','duration_2win','cooldown_300s','soft_gate_hysteresis']:
            groups=[]
            for _,g in whole.groupby(['subject','session'],sort=False):
                g=g.sort_values('pred_ts').copy();pr=g.prob.to_numpy().copy();pr[~g.qc_common_valid]=0
                g['policy_pred']=apply_alert_policy_to_session(pr,g.end_s.to_numpy(),float(g.threshold.iloc[0]),policy=pol,df_session=g)
                positive=g[g.policy_pred==1]
                starts=positive.index[np.r_[True,np.diff(positive.pred_ts)>180]] if len(positive) else []
                g['issued']=0;g.loc[starts,'issued']=1;groups.append(g)
            full=pd.concat(groups);sub=full[full.qc_common_valid&full.y_true.isin([0,1])].copy()
            burden={'stream_issued_episodes':int(full.issued.sum()),'excluded_trigger_episodes':int(full.loc[~(full.qc_common_valid&full.y_true.isin([0,1])),'issued'].sum()),
                    'stream_observed_hours':full.exposure_s.sum()/3600,'active_fraction':full.policy_pred.mean()}
            r,s=summary(sub,evs,name=source+'_'+pol,pred='policy_pred');policies.append({'metric_definition':'active_warning',**burden,**r});perpol.append(s)
            r,s=summary(sub,evs,name=source+'_'+pol+'_issued',pred='issued');policies.append({'metric_definition':'episode_first_issuance',**burden,**r})
    save(pd.DataFrame(policies),'alert_policy_replay');save(pd.concat(perpol),'alert_policy_per_subject')
    # Descriptive test threshold sweep only, NEVER used for model/policy selection.
    curve=[]
    for thr in np.linspace(0,1,41):
        zz=q.copy();zz['sweep_pred']=(zz.prob>=thr).astype(int)
        r,_=summary(zz,evs,name='descriptive_threshold_sweep',pred='sweep_pred');curve.append({'threshold':thr,**r})
    save(pd.DataFrame(curve),'threshold_descriptive_curve')
    # Reliability (equal-width bins) and calibrators fit on development/validation only.
    reliability=[];calibration=[]
    for name in ['labelblind_same_method','balanced_full_grid','unweighted_full_grid']:
        pp=load_pred(name,f);pp=pp[pp.qc_common_valid&pp.y_true.isin([0,1])]
        for col in ['prob','prob_prior_corrected','prob_sigmoid']:
            v=pp[col].to_numpy();yy=pp.y_true.to_numpy();bins=np.digitize(v,np.linspace(0,1,11)[1:-1])
            for b in range(10):
                ii=bins==b
                if ii.any():reliability.append({'experiment':name,'probability':col,'bin':b,'n':int(ii.sum()),'mean_prob':float(v[ii].mean()),'observed_fraction':float(yy[ii].mean())})
            ss=[]
            for sid,g in pp.groupby('subject'):
                ss.append(brier_score_loss(g.y_true,g[col]))
            calibration.append({'experiment':name,'probability':col,'pooled_brier':brier_score_loss(yy,v),'macro_brier':np.mean(ss),
                                'mean_probability':v.mean(),'prevalence':yy.mean(),'pooled_auroc':roc_auc_score(yy,v),'pooled_auprc':average_precision_score(yy,v)})
    save(pd.DataFrame(reliability),'probability_reliability');save(pd.DataFrame(calibration),'probability_calibration_comparison')
    analysis_tail(f,e,p,q,rows,subs)

def analysis_tail(f,e,p,q,rows,subs):
    evs=events_for(e,'P1')
    fq=f.merge(q[KEY+['prob','threshold']],on=KEY)
    with zipfile.ZipFile(RAW/'15/15_1594213322.zip') as z:
        lines=z.read('EDA.csv').decode().splitlines()[:3]
    parsed=pd.read_csv(io.StringIO('\n'.join(lines)),nrows=1)
    jsave({'source':'sources/upstream_predictor_source.py:60-63','raw_first_row_epoch':float(lines[0]),
           'raw_second_row_sampling_rate':float(lines[1]),'upstream_reference_time_expression_value':float(parsed.iloc[0,0]),
           'inference':'Public notebook reads sample rate as reference_time with these distributed raw files. This does NOT establish how SurveyResults.xlsx was produced.',
           'minute_rounding_code_found':False},'upstream_reference_time_reproduction')
    # Corrected per-participant physiology + standardized distribution shift vs other 14 subjects.
    prof=[];dist=[];phys=['eda_mean','eda_recent60_minus_prev120','hr_mean_bpm','acc_motion_change_per_s','ppg_absdiff_per_s','ibi_valid_flag']
    for sid,g in fq.groupby('subject'):
        dev=fq[fq.subject!=sid]
        for feat in phys:
            scale=dev[feat].std();dist.append({'subject':sid,'feature':feat,'standardized_mean_shift':(g[feat].mean()-dev[feat].mean())/scale,
                                             'wasserstein_over_dev_sd':wasserstein_distance(g[feat],dev[feat])/scale})
        rawf=f[f.subject==sid]
        prof.append({'subject':sid,**{feat+'_median':g[feat].median() for feat in phys},'qc_excluded_fraction':1-rawf.qc_common_valid.mean(),
                     'ibi_invalid_fraction':1-rawf.ibi_valid_flag.mean(),'mean_probability':g.prob.mean(),'negative_probability':g[g.y_P1==0].prob.mean(),
                     'positive_probability':g[g.y_P1==1].prob.mean(),'prevalence':g.y_P1.mean(),'threshold':g.threshold.iloc[0]})
    save(pd.DataFrame(dist),'subject_distribution_shift')
    profile=pd.DataFrame(prof).merge(subs[subs.experiment=='labelblind_same_method'],on='subject');save(profile,'corrected_subject_profiles')
    cor=[]
    for feature in [c for c in profile if c.endswith('_median') or c in ['qc_excluded_fraction','ibi_invalid_fraction','negative_probability']]:
        for metric in ['auroc','auprc','fpr','event_recall']:
            r=spearmanr(profile[feature],profile[metric]);cor.append({'feature':feature,'metric':metric,'spearman_r':r.statistic,'p_unadjusted_descriptive':r.pvalue,'n':15})
    save(pd.DataFrame(cor),'participant_physiology_performance_correlations')
    # Paired participant bootstrap for pre-specified controlled contrasts.
    paired=[]
    contrasts=[('labelblind_same_method','legacy_reproduction'),('model_RF','labelblind_same_method'),('model_LightGBM','labelblind_same_method'),
               ('unweighted_full_grid','balanced_full_grid'),('horizon_P0','labelblind_same_method'),('horizon_P1-A','horizon_P1-B'),
               ('calibration_actual_60s','calibration_raw'),('calibration_actual_180s','calibration_raw'),('calibration_actual_300s','calibration_raw'),
               ('calibration_quiet_initial300s','calibration_raw'),('calibration_labelblind_same_method','labelblind_same_method'),
               ('common171_P0','common171_P1'),('common171_P1-A','common171_P1-B'),('common171_P2','common171_P1')]
    if 'legacy_matched_threads' in set(subs.experiment):contrasts.append(('labelblind_same_method','legacy_matched_threads'))
    for a,b in contrasts:
        aa=subs[subs.experiment==a].set_index('subject');bb=subs[subs.experiment==b].set_index('subject');idx=aa.index.intersection(bb.index)
        for metric in ['auroc','auprc','event_recall','fpr']:
            delta=(aa.loc[idx,metric]-bb.loc[idx,metric]).dropna();c=participant_ci(pd.DataFrame({'d':delta}),'d');paired.append({'experiment_a':a,'experiment_b':b,'metric':metric,**{k:v for k,v in c.items() if k!='metric'}})
    save(pd.DataFrame(paired),'paired_participant_bootstrap_differences')
    # Scores by exact lead bins with the SAME trained P1 model, avoiding different-model comparison.
    leadrows=[]
    for task,(lo,hi) in HORIZONS.items():
        yy=task_labels(f,task);pp=p.merge(f[KEY].assign(_row=f.index),on=KEY).sort_values('_row');pp['y_true']=yy
        pp=pp[pp.qc_common_valid&pp.y_true.isin([0,1])];r,_=summary(pp,events_for(e,task),task,name='frozen_P1_'+task);leadrows.append(r)
    save(pd.DataFrame(leadrows),'fixed_model_horizon_comparison')
    # Dependence: thin score evaluation onto a prespecified session-anchored 180s grid.
    thin=[]
    for hop in [30,60,180,300]:
        zz=q[np.isclose((q.end_s-180)%hop,0)];r,_=summary(zz,evs,name=f'thin_{hop}s');thin.append(r)
    save(pd.DataFrame(thin),'nonoverlap_evaluation_sensitivity')
    # Original negative label is a proxy: verify no future/current event mislabeled negative.
    contamination=[]
    for category,es in [('confirmed',e[e.label_category.isin(['confirmed_high_stress','confirmed_moderate_stress'])]),('na',e[e.label_category=='unconfirmed_algorithm_candidate'])]:
        for r in es.itertuples():
            mask=(f.subject==r.subject)&(f.pred_ts>=r.event_start_ts)&(f.pred_ts<=r.event_end_ts)
            contamination.append({'category':category,'event_id':r.event_id,'negative_windows_inside':int((mask&f.y_P1.eq(0)).sum()),'p1_positive_inside':int((mask&f.y_P1.eq(1)).sum())})
    save(pd.DataFrame(contamination),'label_interval_contamination')
    cc=pd.read_csv(OUT/'coefficients_labelblind_same_method.csv')
    importance=cc.groupby('feature').coefficient.agg(mean_coefficient='mean',sd_coefficient='std',median_coefficient='median',positive_fold_fraction=lambda v:(v>0).mean()).reset_index()
    importance['abs_mean_coefficient']=importance.mean_coefficient.abs();save(importance.sort_values('abs_mean_coefficient',ascending=False),'feature_coefficient_stability')
    # Participant cluster ratio CIs for cohort event recall and FA/hour.
    ss=subs[subs.experiment=='labelblind_same_method'];rng=np.random.default_rng(719)
    ix=rng.integers(0,len(ss),size=(5000,len(ss)))
    boot_ev=ss.detected_events.to_numpy()[ix].sum(axis=1)/ss.n_events.to_numpy()[ix].sum(axis=1)
    boot_fa=ss.false_episodes.to_numpy()[ix].sum(axis=1)/ss.negative_hours.to_numpy()[ix].sum(axis=1)
    save(pd.DataFrame([{'metric':'cohort_event_recall','estimate':ss.detected_events.sum()/ss.n_events.sum(),'lower95':np.quantile(boot_ev,.025),'upper95':np.quantile(boot_ev,.975)},
                       {'metric':'cohort_fa_hr','estimate':ss.false_episodes.sum()/ss.negative_hours.sum(),'lower95':np.quantile(boot_fa,.025),'upper95':np.quantile(boot_fa,.975)}]),'cohort_participant_cluster_ci')
    # Grid dependence and time-weighted held-out discrimination.
    grid=[]
    for name in ['labelblind_same_method','time_weighted_LR']:
        if not (OUT/'predictions'/f'{name}.csv.gz').exists():continue
        pp=load_pred(name,f);pp=pp[pp.qc_common_valid&pp.y_true.isin([0,1])];w=pp.exposure_s
        auc=[];ap=[]
        for sid,g in pp.groupby('subject'):
            auc.append(roc_auc_score(g.y_true,g.prob,sample_weight=g.exposure_s));ap.append(average_precision_score(g.y_true,g.prob,sample_weight=g.exposure_s))
        grid.append({'experiment':name,'window_prevalence':pp.y_true.mean(),'time_weighted_prevalence':np.average(pp.y_true,weights=w),
                     'time_weighted_pooled_auroc':roc_auc_score(pp.y_true,pp.prob,sample_weight=w),'time_weighted_pooled_auprc':average_precision_score(pp.y_true,pp.prob,sample_weight=w),
                     'time_weighted_macro_auroc':np.mean(auc),'time_weighted_macro_auprc':np.mean(ap),
                     'positive_mean_step_s':pp.loc[pp.y_true==1,'exposure_s'].mean(),'negative_mean_step_s':pp.loc[pp.y_true==0,'exposure_s'].mean()})
    save(pd.DataFrame(grid),'sampling_time_weight_controls')
    # Event-aligned raw trajectories use a fixed cohort available for every bin.
    tr=pd.read_csv(OUT/'raw_event_60s_trajectories.csv',dtype={'subject':str})
    unique_e=e.drop_duplicates(['subject','event_start_ts','event_end_ts'])
    tr=tr[tr.event_id.isin(unique_e.event_id)]
    complete=tr.groupby('event_id').relative_start_s.nunique();ids=complete[complete==16].index
    tr=tr[tr.event_id.isin(ids)].copy()
    reference=tr[tr.relative_start_s.between(-600,-360)].groupby('event_id').eda_mean.mean()
    tr['eda_delta_remote']=tr.eda_mean-tr.event_id.map(reference)
    save(tr,'paired_raw_event_trajectories')
    traj=[]
    for (cat,t),g in tr.groupby(['category','relative_start_s']):
        by=g.groupby('subject').eda_delta_remote.mean()
        ci0=participant_ci(pd.DataFrame({'delta':by}),'delta')
        traj.append({'category':cat,'relative_start_s':t,'events':g.event_id.nunique(),'subjects':g.subject.nunique(),
                     'event_mean_eda':g.eda_mean.mean(),'subject_mean_eda_delta':by.mean(),'delta_lower95':ci0['lower95'],'delta_upper95':ci0['upper95']})
    save(pd.DataFrame(traj),'paired_trajectory_summary')
    # Cohort-independent all-0.5 scores distinguish Dummy pooled artifact from signal.
    dummy=q.copy();dummy['prob']=.5;dummy['pred']=0;r,_=summary(dummy,evs,name='constant_0.5_dummy')
    save(pd.DataFrame([r]),'constant_dummy_control')
    print('analysis complete',flush=True)

def resume_analysis():
    """Resume after already persisted curves/reliability; no model fits or repeat replays."""
    f,e,m=data();f['exposure_s']=exposure(f);p=load_pred('labelblind_same_method',f);q=p[p.qc_common_valid&p.y_true.isin([0,1])]
    rows=pd.read_csv(OUT/'controlled_experiments.csv');subs=pd.read_csv(OUT/'controlled_per_subject.csv',dtype={'subject':str})
    if not rows.experiment.eq('legacy_matched_threads').any():
        pp=load_pred('legacy_matched_threads',f);pp=pp[pp.qc_common_valid&pp.y_true.isin([0,1])];r,s=summary(pp,events_for(e,'P1'),name='legacy_matched_threads')
        rows=pd.concat([rows,pd.DataFrame([r])],ignore_index=True);subs=pd.concat([subs,s],ignore_index=True)
        save(rows,'controlled_experiments');save(subs,'controlled_per_subject')
    matched=[sid for sid,g in q.groupby('subject') if g.y_true.eq(1).any() and g.state.eq('CONFIRMED_NO_STRESS_0').any() and g.state.eq('VERIFIED_NON_EVENT').any()]
    rs=[];ss=[]
    for name,negative in [('confirmed_zero',['CONFIRMED_NO_STRESS_0']),('unverified_background',['VERIFIED_NON_EVENT'])]:
        zz=q[q.subject.isin(matched)&((q.y_true==1)|q.state.isin(negative))]
        r,s=summary(zz,events_for(e,'P1')[events_for(e,'P1').subject.isin(matched)],name=name);rs.append(r);ss.append(s)
    save(pd.DataFrame(rs),'matched_subject_negative_source_control');save(pd.concat(ss),'matched_subject_negative_source_metrics')
    with zipfile.ZipFile(RAW/'15/15_1594213322.zip') as z:
        lines=z.read('EDA.csv').decode().splitlines()[:3]
    parsed=pd.read_csv(io.StringIO('\n'.join(lines)),nrows=1)
    jsave({'source':'sources/upstream_predictor_source.py:60-63','raw_first_row_epoch':float(lines[0]),
           'raw_second_row_sampling_rate':float(lines[1]),'upstream_reference_time_expression_value':float(parsed.iloc[0,0]),
           'inference':'Public notebook reads sample rate as reference_time with these distributed raw files. This does NOT establish how SurveyResults.xlsx was produced.',
           'minute_rounding_code_found':False},'upstream_reference_time_reproduction')
    analysis_tail(f,e,p,q,rows,subs)

def figures():
    def csv(n):return pd.read_csv(OUT/(n+'.csv'),dtype={'subject':str})
    (OUT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'figure.dpi':120,'axes.spines.top':False,'axes.spines.right':False})
    def finish(fig,name,caption='15-participant LOSO; survey interval boundaries; exploratory audit, not ASD outcomes'):
        fig.text(.01,.006,caption,fontsize=8,color='#444444');fig.tight_layout(rect=(0,.035,1,1));fig.savefig(OUT/'figures'/(name+'.png'),dpi=160);plt.close(fig)
    f,e,m=data();f['exposure_s']=exposure(f);p=load_pred('labelblind_same_method',f);p=p[p.qc_common_valid&p.y_true.isin([0,1])]
    r=csv('controlled_experiments');sub=csv('corrected_subject_profiles')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for name,label in [('labelblind_same_method','LR'),('model_RF','RF'),('model_LightGBM','LightGBM'),('model_Dummy','Dummy'),('simple_eda_slope','Single EDA trend')]:
        pp=load_pred(name,f);pp=pp[pp.qc_common_valid&pp.y_true.isin([0,1])];pr,re,th=precision_recall_curve(pp.y_true,pp.prob)
        axs[0].plot(re,pr,label=f'{label} AP={average_precision_score(pp.y_true,pp.prob):.3f}')
    axs[0].axhline(p.y_true.mean(),ls='--',color='gray',label='Window prevalence');axs[0].set(xlabel='Window recall',ylabel='Precision',ylim=(0,.3),title='P1 pooled PR (zoom 0-0.30)');axs[0].legend(fontsize=8)
    re=csv('probability_reliability');re=re[re.experiment=='labelblind_same_method']
    for col,label in [('prob','Class-balanced score'),('prob_prior_corrected','Prior odds correction'),('prob_sigmoid','Inner-validation sigmoid')]:
        g=re[re.probability==col];axs[1].plot(g.mean_prob,g.observed_fraction,'o-',label=label)
    axs[1].plot([0,1],[0,1],'k--',lw=1);axs[1].set(xlabel='Mean predicted score',ylabel='Observed P1 fraction',title='Reliability (equal-width bins)');axs[1].legend(fontsize=8)
    finish(fig,'01_pr_and_reliability')
    curve=csv('threshold_descriptive_curve');fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    axs[0].plot(curve.threshold,curve.pooled_precision,label='Window precision');axs[0].plot(curve.threshold,curve.pooled_recall,label='Window recall');axs[0].set(xlabel='Threshold',ylabel='Rate',title='Frozen P1 scores: descriptive sweep');axs[0].legend()
    axs[1].plot(curve.fa_hr,curve.event_recall,'o-',ms=3)
    for t in [.3,.5,.7,.9]:
        g=curve.iloc[(curve.threshold-t).abs().argmin()];axs[1].annotate(f'{g.threshold:.2f}',(g.fa_hr,g.event_recall))
    axs[1].set(xlabel='False episodes / background hour',ylabel='Event coverage recall',title='Task-specific episodes; no test threshold selected');finish(fig,'02_threshold_and_event_tradeoff')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    order=['P0','P1-A','P1-B','P1','P2'];g=r[r.experiment.isin(['common171_'+t for t in order])].set_index('task').loc[order]
    axs[0].plot(order,g.macro_auroc,'o-',label='Mean AUROC');axs[0].plot(order,g.macro_auprc,'s-',label='Mean AUPRC');axs[0].set(title='Identical 171-event cohort',ylabel='Participant-mean metric');axs[0].legend()
    axs[1].plot(order,g.event_recall,'o-',label='Event coverage recall');ax2=axs[1].twinx();ax2.plot(order,g.fa_hr,'s--',color='#b23b2f',label='FA/hour');axs[1].set(ylabel='Event recall',title='Same features/splits; independent train by horizon');ax2.set_ylabel('False episodes / hour',color='#b23b2f');finish(fig,'03_horizon_comparison')
    fig,axs=plt.subplots(1,2,figsize=(13,4.8));x=np.arange(len(sub))
    axs[0].bar(x-.2,sub.auroc,.4,label='AUROC');axs[0].bar(x+.2,sub.auprc,.4,label='AUPRC');axs[0].set_xticks(x,sub.subject);axs[0].set(title='All 15 held-out participants',ylabel='Metric');axs[0].axhline(.5,color='gray',ls='--');axs[0].legend()
    axs[1].bar(x-.2,sub.event_recall,.4,label='Event recall');axs[1].bar(x+.2,sub.fpr,.4,label='Window FPR');axs[1].set_xticks(x,sub.subject);axs[1].set(title='Same validation-selected thresholds',ylabel='Rate');axs[1].legend();finish(fig,'04_all_participants')
    g=csv('motion_quartile_metrics');g=g[g.source=='filtered'];fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    for gate,ss in g.groupby('gate',sort=False):
        for ax,col in zip(axs,['fpr','recall','auroc']):ax.plot(ss.quartile,ss[col],'o-',label=gate);ax.set(title=col.upper(),xlabel='Activity quartile',ylabel=col)
    axs[0].legend(fontsize=8);finish(fig,'05_motion_gate_errors')
    g=csv('timing_offset_sensitivity');fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    xs=[-60,-30,0,30,60];axs[0].plot(xs,g.macro_auroc,'o-',label='Mean AUROC');axs[0].plot(xs,g.macro_auprc,'s-',label='Mean AUPRC');axs[0].legend();axs[0].set(xlabel='Hypothetical survey offset (s)',ylabel='Metric',title='Frozen model, fixed prediction grid')
    axs[1].plot(xs,g.event_recall,'o-',label='Event recall');axs[1].set(xlabel='Hypothetical survey offset (s)',ylabel='Event coverage recall',title='Sensitivity only; offsets are not known corrections');finish(fig,'06_timestamp_sensitivity')
    g=csv('paired_trajectory_summary');fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    names={'confirmed_high_stress':'High stress','confirmed_moderate_stress':'Moderate stress','survey_confirmed_no_stress':'Confirmed zero','unconfirmed_algorithm_candidate':'Unconfirmed'}
    for cat,ss in g.groupby('category'):
        axs[0].plot(ss.relative_start_s,ss.event_mean_eda,label=names[cat]);axs[1].plot(ss.relative_start_s,ss.subject_mean_eda_delta,label=names[cat]);axs[1].fill_between(ss.relative_start_s,ss.delta_lower95,ss.delta_upper95,alpha=.10)
    for ax in axs:ax.axvline(0,color='black',ls='--');ax.set_xlabel('Seconds relative to survey boundary');ax.legend(fontsize=8)
    axs[0].set(ylabel='EDA (uS)',title='Fixed event cohort; raw non-overlapping 60s bins');axs[1].set(ylabel='EDA minus remote baseline (uS)',title='Equal participant weighting, cluster bootstrap 95% CI');finish(fig,'07_raw_event_trajectories')
    g=r[r.experiment.str.startswith('calibration_')].copy();g['method']=g.experiment.str.replace('calibration_','')
    order=['raw','actual_60s','actual_180s','actual_300s','labelblind_same_method','quiet_initial300s'];g=g.set_index('method').loc[order].reset_index()
    labels=['Raw','60 s','180 s','300 s','Initial windows','Quiet <=300 s'];fig,axs=plt.subplots(1,2,figsize=(13,5))
    axs[0].bar(labels,g.macro_auroc);axs[0].set(ylabel='Participant mean AUROC',title='Same 16 features; baseline changed only',ylim=(.45,.8));axs[0].tick_params(axis='x',rotation=25)
    for label,a in zip(labels,g.itertuples()):axs[1].scatter(a.fa_hr,a.event_recall,label=label,s=40)
    axs[1].set(xlabel='False episodes / hour',ylabel='Event coverage recall',title='Calibration trade-offs');axs[1].legend(fontsize=9,loc='lower right');finish(fig,'08_calibration_control')
    co=csv('feature_coefficient_stability').head(15).iloc[::-1];fig,ax=plt.subplots(figsize=(10,6));ax.barh(co.feature,co.mean_coefficient,xerr=co.sd_coefficient,color=np.where(co.mean_coefficient>0,'#2b6f8e','#b35042'));ax.set(xlabel='Mean standardized LR coefficient; error = fold SD',title='Associations only; correlated features, not physiological causes');finish(fig,'09_coefficient_stability')
    policy_all=csv('alert_policy_replay');g=policy_all[policy_all.metric_definition=='active_warning'];fig,axs=plt.subplots(1,2,figsize=(14,5.5));ax=axs[0]
    for source in ['filtered','full_stream']:
        ss=g[g.experiment.str.startswith(source+'_')];ax.scatter(ss.fa_hr,ss.event_recall,label=source)
        for a in ss.itertuples():ax.annotate(a.experiment.replace(source+'_',''),(a.fa_hr,a.event_recall),fontsize=7,xytext=(2,3 if source=='filtered' else -10),textcoords='offset points')
    ax.set(xlabel='False episodes / background hour',ylabel='Active-warning event coverage',title='Filtered vs complete timeline');ax.legend(fontsize=8)
    ax.set_ylim(.10,.75)
    names=['val_opt','fixed_05','hysteresis','duration_2win','cooldown_300s','soft_gate_hysteresis'];x=np.arange(len(names))
    active=policy_all.set_index('experiment').loc[['full_stream_'+n for n in names],'event_recall']
    issued=policy_all.set_index('experiment').loc[['full_stream_'+n+'_issued' for n in names],'event_recall']
    axs[1].bar(x-.18,active,.36,label='Warning active in P1');axs[1].bar(x+.18,issued,.36,label='First issuance in P1')
    axs[1].set_xticks(x,['Validation','Fixed 0.5','Hysteresis','2 windows','Cooldown','Soft + hyst.'],rotation=30,ha='right');axs[1].set(ylabel='Event recall',title='Full timeline: two distinct warning endpoints');axs[1].legend(fontsize=8)
    finish(fig,'10_policy_stream_replay')
    print('figures complete',flush=True)

def verify():
    checks=[];mismatches=[]
    for filename in ['original_input_hashes','raw_archive_hashes']:
        hashes=pd.read_csv(OUT/(filename+'.csv'))
        for r in hashes.itertuples():
            p=ROOT/r.path
            if not p.exists() or digest(p)!=r.sha256:mismatches.append(r.path)
        checks.append({'check':filename+'_unchanged','n':len(hashes),'passed':not any(x in set(hashes.path) for x in mismatches)})
    f,e,m=data();checks.append({'check':'feature_windows_51234','passed':len(f)==51234})
    checks.append({'check':'cached_labels_reproduced','passed':bool(csv_mismatch_zero())})
    configs=[json.loads(p.read_text()) for p in OUT.glob('config_*.json')]
    splits=[]
    for c in configs:
        sf=pd.read_csv(OUT/('splits_'+c['experiment']+'.csv'),dtype={'test_subject':str})
        for r in sf.itertuples():
            assert r.test_subject not in set(r.train_subjects.split(';'))|set(r.val_subjects.split(';'))
            assert not set(r.train_subjects.split(';'))&set(r.val_subjects.split(';'))
        splits.append(len(sf)==15)
    checks.append({'check':'new_experiment_split_isolation','n_experiments':len(configs),'passed':all(splits)})
    figures=list((OUT/'figures').glob('*.png'));checks.append({'check':'ten_nonempty_figures','passed':len(figures)==10 and all(p.stat().st_size>10000 for p in figures)})
    p=ROOT/'reports/NURSE_FAILURE_ROOT_CAUSE_ANALYSIS_KO.md';report=p.read_text(encoding='utf-8')
    import re
    broken=[]
    for target in re.findall(r'\]\(([^)]+)\)',report):
        if not target.startswith(('https://','http://')) and not (p.parent/target).resolve().exists():broken.append(target)
    checks.append({'check':'report_local_links','passed':not broken,'broken':broken})
    tests=(OUT/'test_results.txt').read_text(encoding='utf-8-sig',errors='replace')
    checks.append({'check':'twenty_regression_tests','passed':'Ran 20 tests' in tests and 'OK' in tests})
    jsave({'all_passed':all(c['passed'] for c in checks),'checks':checks,'changed_original_files':mismatches,
           'scope':'Local code/data/metric verification. Independent event truth and uniform-grid prospective replay remain unavailable.',
           'report_sha256':digest(p)},'verification')
    print(json.dumps({'all_passed':all(c['passed'] for c in checks),'checks':checks},ensure_ascii=False),flush=True)

def csv_mismatch_zero():
    return pd.read_csv(OUT/'label_cache_consistency.csv').mismatches.eq(0).all()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['inventory','raw_audit','train','analyze','resume_analysis','figures','verify'])
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        globals()[a.phase]()
