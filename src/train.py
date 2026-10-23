"""LOSO adult stress classification. Run from project root: python -m src.train --help."""
import argparse
import hashlib
import platform
import subprocess
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .evaluate import score,summarise,ablation,write_json,make_figures

SENSORS={'eda':['eda_'],'ppg':['ppg_'],'acc':['acc_'],
 'eda_ppg':['eda_','ppg_'],'eda_acc':['eda_','acc_'],
 'ppg_acc':['ppg_','acc_'],'all':['eda_','ppg_','acc_']}
EXPECTED_WESAD={f'S{i}' for i in list(range(2,12))+list(range(13,18))}

def feature_columns(frame,sensor):
    cols=[c for c in frame.columns if any(c.startswith(p) for p in SENSORS[sensor])]
    if not cols:raise ValueError('No features for '+sensor)
    return cols

def boolean_column(series):
    if pd.api.types.is_bool_dtype(series):return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):return series.fillna(0).astype(int).eq(1)
    return series.astype(str).str.lower().isin(['true','1','1.0'])

def eligible_mask(frame):
    if 'qc_common_valid' not in frame:
        raise ValueError('Missing qc_common_valid: require audited common-window QC, no implicit fallback.')
    return frame.y.isin([0,1]) & boolean_column(frame.qc_common_valid)

def group_splits(df):
    if df.subject.nunique()<4:raise ValueError('Need >=4 independent subjects.')
    outer=LeaveOneGroupOut()
    for fold,(dev_idx,test_idx) in enumerate(outer.split(df,df.y,df.subject)):
        dev=df.iloc[dev_idx]
        splitter=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42+fold)
        tr_rel,va_rel=next(splitter.split(dev,dev.y,dev.subject))
        train_idx,val_idx=dev_idx[tr_rel],dev_idx[va_rel]
        sets=[set(df.iloc[ix].subject) for ix in (train_idx,val_idx,test_idx)]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        assert len(sets[2])==1
        if any(df.iloc[ix].y.nunique()!=2 for ix in (train_idx,val_idx)):
            raise ValueError('Train/validation lacks a class; label review required, no test-adaptive resplitting.')
        yield fold,train_idx,val_idx,test_idx,dict(fold=fold,train=sorted(sets[0]),validation=sorted(sets[1]),test=sorted(sets[2]))

def build_model(name,jobs=4):
    if name=='dummy':model=DummyClassifier(strategy='prior')
    elif name=='lr':model=LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=42)
    elif name=='rf':model=RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,
        class_weight='balanced_subsample',random_state=42,n_jobs=jobs)
    else:raise ValueError(name)
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),model)

def select_threshold(y,prob):
    thresholds=np.linspace(.05,.95,91)
    scores=np.asarray([f1_score(y,prob>=t,zero_division=0) for t in thresholds])
    return float(thresholds[np.flatnonzero(np.isclose(scores,scores.max()))[-1]])

def train(args):
    source=Path(args.features);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    raw=pd.read_csv(source,dtype={'subject':str,'session':str})
    required={'dataset','subject','session','start_s','end_s','step_s','phase','y','qc_common_valid'}
    if required-set(raw):raise ValueError('Missing fields: '+str(required-set(raw)))
    if raw.dataset.nunique()!=1:raise ValueError('One dataset/protocol experiment per run; do not pool datasets silently.')
    if raw.duplicated(['subject','session','end_s']).any():raise ValueError('Duplicate participant/session/end windows.')
    raw['evaluation_eligible']=eligible_mask(raw)
    df=raw[raw.evaluation_eligible].reset_index(drop=True)
    if df.empty:raise ValueError('No eligible windows.')
    dataset=str(raw.dataset.iloc[0]); subjects=sorted(df.subject.unique())
    if args.expected_subjects is not None and len(subjects)!=args.expected_subjects:
        raise ValueError(f'Expected {args.expected_subjects} eligible subjects, found {len(subjects)}. Audit first.')
    if dataset.lower()=='wesad' and args.expected_subjects==15 and set(subjects)!=EXPECTED_WESAD:
        raise ValueError('WESAD participant identity mismatch.')
    quality=[]
    for subject,g in raw.groupby('subject'):
        quality.append(dict(subject=subject,total_windows=len(g),eligible_windows=int(g.evaluation_eligible.sum()),
          invalid_label_windows=int((~g.y.isin([0,1])).sum()),quality_invalid_windows=int((~boolean_column(g.qc_common_valid)).sum()),
          excluded_union=int((~g.evaluation_eligible).sum())))
    pd.DataFrame(quality).to_csv(out/'quality_counts.csv',index=False)
    quality_by_phase = raw.groupby(['dataset','subject','phase','y'],dropna=False).agg(
        total_windows=('y','size'), eligible_windows=('evaluation_eligible','sum'))
    quality_by_phase['excluded_windows'] = quality_by_phase.total_windows-quality_by_phase.eligible_windows
    quality_by_phase.to_csv(out/'quality_by_phase.csv')
    pairs=[('dummy','all'),('lr','all'),('rf','all')]+[('lr',s) for s in SENSORS if s!='all'] if args.plan=='core' else [(m,s) for m in args.models for s in args.sensors]
    config=vars(args)|dict(task='adult current stress classification',seed=42,dataset=dataset,
        shared_window_policy='y in {0,1} AND qc_common_valid; identical for every model/sensor',
        outer_split='LOSO',inner_split='one GroupShuffleSplit, test_size=.25, seed=42+outer_fold',
        threshold_rule='validation F1 max over .05:.01:.95; largest threshold on ties; dummy=.5',
        model_refit='No refit on inner validation after threshold selection',
        pairs=pairs,feature_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    write_json(out/'config.json',config)
    environment=dict(python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),
        numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        runtime_note='Measured on this execution host, not user Windows laptop or MCU.')
    try:environment['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:environment['git_commit']=None
    write_json(out/'environment.json',environment)
    per=[];predictions=[];splits=[];negative=[];pooled=[]
    for fold,it,iv,ie,split in group_splits(df):
        splits.append(split);tr,va,te=df.iloc[it],df.iloc[iv],df.iloc[ie]
        test_all=raw[raw.subject.isin(split['test'])].copy()
        for name,sensor in pairs:
            cols=feature_columns(df,sensor)
            model=build_model(name,args.jobs)
            tic=time.perf_counter();model.fit(tr[cols],tr.y);fit_s=time.perf_counter()-tic
            threshold=.5 if name=='dummy' else select_threshold(va.y,model.predict_proba(va[cols])[:,1])
            tic=time.perf_counter();p=model.predict_proba(test_all[cols])[:,1];infer_ms=(time.perf_counter()-tic)*1000/len(test_all)
            current=test_all[['dataset','subject','session','start_s','end_s','step_s','phase','y','evaluation_eligible','qc_common_valid']].copy()
            current['prob']=p;current['pred']=(p>=threshold).astype(int)
            current['pred_threshold05']=(p>=.5).astype(int)
            current['threshold']=threshold;current['fold']=fold;current['model']=name;current['sensors']=sensor
            predictions.append(current)
            g=current[current.evaluation_eligible]
            r=dict(dataset=dataset,subject_id=split['test'][0],model=name,sensors=sensor,fold=fold,threshold=threshold,
                fit_s=fit_s,batch_inference_ms_per_window=infer_ms,**score(g.y,g.prob,threshold))
            r.update({'threshold05_'+k:v for k,v in score(g.y,g.prob,.5).items() if k in ['precision','recall','f1','macro_f1','balanced_acc']})
            r['excluded_windows']=int((~test_all.evaluation_eligible).sum());per.append(r)
            for phase,sg in g[g.y==0].groupby('phase'):
                negative.append(dict(dataset=dataset,subject_id=split['test'][0],model=name,sensors=sensor,negative_phase=phase,
                    n_windows=len(sg),false_stress_windows=int(sg.pred.sum()),false_stress_rate=float(sg.pred.mean())))
            joblib.dump(dict(model=model,columns=cols,threshold=threshold,subjects=split,sensor=sensor,dataset=dataset,
                task='adult current stress classification'),out/'models'/f'{name}_{sensor}_fold{fold}.joblib')
            print(f'fold={fold+1}/{len(subjects)} subject={split["test"][0]} model={name} sensors={sensor} fit_s={fit_s:.3f}',flush=True)
        pd.DataFrame(per).to_csv(out/'per_subject.partial.csv',index=False)
        write_json(out/'splits.json',splits)
    per=pd.DataFrame(per);per.to_csv(out/'per_subject.csv',index=False)
    pred=pd.concat(predictions,ignore_index=True);pred.to_csv(out/'predictions.csv',index=False)
    summary=summarise(per);summary.to_csv(out/'summary.csv',index=False)
    abl,paired=ablation(per);abl.to_csv(out/'ablation.csv',index=False);paired.to_csv(out/'paired_sensor_deltas.csv',index=False)
    pd.DataFrame(negative).to_csv(out/'negative_class_errors.csv',index=False)
    for (model,sensor),g in pred[pred.evaluation_eligible].groupby(['model','sensors']):
        # Thresholds differ by held-out fold. Score discrete decisions directly where needed.
        pm=score(g.y,g.prob,.5);from sklearn.metrics import f1_score,precision_score,recall_score,balanced_accuracy_score,confusion_matrix
        pm.update(precision=float(precision_score(g.y,g.pred,zero_division=0)) if g.pred.sum() else np.nan,recall=float(recall_score(g.y,g.pred,zero_division=0)),
            f1=float(f1_score(g.y,g.pred,zero_division=0)),macro_f1=float(f1_score(g.y,g.pred,average='macro',zero_division=0)),
            balanced_acc=float(balanced_accuracy_score(g.y,g.pred)))
        tn,fp,fn,tp=confusion_matrix(g.y,g.pred,labels=[0,1]).ravel();pm.update(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),fpr=float(fp/(fp+tn)))
        pooled.append(dict(model=model,sensors=sensor,**pm))
    pd.DataFrame(pooled).to_csv(out/'pooled_oof.csv',index=False)
    result=dict(dataset_version=args.dataset_version or dataset,participants=len(subjects),subject_ids=subjects,
        task_definition='Adult current stress classification; not prospective ASD behavior prediction',
        labels={'positive':'WESAD stress=2 or explicitly audited stress task','negative':'WESAD baseline=1/amusement=3 or audited exercise task'},
        window_s=sorted((raw.end_s-raw.start_s).unique().tolist()),hop_s=sorted(raw.step_s.unique().tolist()),sensor_set=list(SENSORS),
        outer_split='LOSO',inner_split=config['inner_split'],model=[m for m,s in pairs],seed=42,prevalence=float(df.y.mean()),
        precision={r['model']+'/'+r['sensors']:r['precision_mean'] for r in summary.to_dict('records')},
        recall={r['model']+'/'+r['sensors']:r['recall_mean'] for r in summary.to_dict('records')},
        macro_f1={r['model']+'/'+r['sensors']:r['macro_f1_mean'] for r in summary.to_dict('records')},
        balanced_accuracy={r['model']+'/'+r['sensors']:r['balanced_acc_mean'] for r in summary.to_dict('records')},
        auprc={r['model']+'/'+r['sensors']:r['auprc_mean'] for r in summary.to_dict('records')},
        auroc={r['model']+'/'+r['sensors']:r['auroc_mean'] for r in summary.to_dict('records')},
        latency_ms={'batch_inference_ms_per_window_median':float(per.batch_inference_ms_per_window.median()),'host':environment},
        subject_summary=summary.to_dict('records'),pooled_oof_secondary=pooled,
        asd_event_recall=None,asd_event_precision=None,asd_false_alarms_day=None,asd_lead_time=None,
        notes=['AP uses sklearn average_precision_score, not trapezoidal PR-AUC.',
          '95% CIs are descriptive participant bootstrap (2000 draws); small n and shared training folds remain limitations.',
          'Common QC selection may favor cleaner conditions; excluded counts reported.',
          'ASD event metrics N/A: no ASD onset ground truth.',
          'Inference timings exclude feature extraction, BLE and MCU; batch timing is not per-call latency.'])
    write_json(out/'metrics.json',result)
    make_figures(out,dataset,'stress vs baseline+amusement' if dataset.lower()=='wesad' else 'audited stress vs exercise')
    print('DONE: Adult stress classification only. Results:',out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',default='data/processed/wesad_features.csv');p.add_argument('--out',default='results/wesad')
    p.add_argument('--plan',choices=['core','custom'],default='core')
    p.add_argument('--models',nargs='+',choices=['dummy','lr','rf'],default=['dummy','lr','rf'])
    p.add_argument('--sensors',nargs='+',choices=list(SENSORS),default=['all'])
    p.add_argument('--expected-subjects',type=int);p.add_argument('--dataset-version');p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args();train(args)
if __name__=='__main__':main()
