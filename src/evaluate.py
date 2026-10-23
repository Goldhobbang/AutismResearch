"""Subject-level evaluation of adult stress classification; no ASD alert metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve)

METRICS = ['balanced_acc','f1','macro_f1','precision','recall','auprc','auroc','brier','fpr']

def score(y, prob, threshold=.5):
    y, prob = np.asarray(y, dtype=int), np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    both = len(np.unique(y)) == 2
    reasons = []
    if not both:
        reasons.append('Only one class; AUROC/AP/BA undefined.')
    if not (y == 1).any():
        reasons.append('No positive labels; stress recall/F1 undefined.')
    if not (pred == 1).any():
        reasons.append('No predicted positives; stress precision undefined.')
    return dict(n_windows=len(y), positive_windows=int((y==1).sum()),
        negative_windows=int((y==0).sum()), prevalence=float(y.mean()),
        balanced_acc=float(balanced_accuracy_score(y,pred)) if both else np.nan,
        precision=float(precision_score(y,pred,zero_division=0)) if (pred==1).any() else np.nan,
        recall=float(recall_score(y,pred,zero_division=0)) if (y==1).any() else np.nan,
        f1=float(f1_score(y,pred,zero_division=0)) if (y==1).any() else np.nan,
        macro_f1=float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
        auprc=float(average_precision_score(y,prob)) if both else np.nan,
        auroc=float(roc_auc_score(y,prob)) if both else np.nan,
        brier=float(brier_score_loss(y,prob)), fpr=float(fp/(fp+tn)) if fp+tn else np.nan,
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        excluded_reason=' '.join(reasons))

def bootstrap(values):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return dict(valid_subjects=0, **{k:np.nan for k in ['mean','std','min','q25','max','ci_low','ci_high']})
    rng=np.random.default_rng(42)
    means=rng.choice(x,size=(2000,len(x)),replace=True).mean(axis=1)
    return dict(valid_subjects=int(len(x)),mean=float(x.mean()),std=float(x.std(ddof=1)) if len(x)>1 else 0.,
        min=float(x.min()),q25=float(np.quantile(x,.25)),max=float(x.max()),
        ci_low=float(np.quantile(means,.025)),ci_high=float(np.quantile(means,.975)))

def summarise(per):
    rows=[]
    for (model,sensors),g in per.groupby(['model','sensors'],sort=False):
        r=dict(model=model,sensors=sensors,n_subjects=g.subject_id.nunique(),n_windows=int(g.n_windows.sum()))
        for metric in METRICS:
            r.update({metric+'_'+k:v for k,v in bootstrap(g[metric]).items()})
        rows.append(r)
    return pd.DataFrame(rows)

def ablation(per):
    rows=[]; paired=[]
    for model,g in per.groupby('model'):
        base=g[g.sensors=='eda'].set_index('subject_id')
        if base.empty: continue
        for sensor,sub in g.groupby('sensors',sort=False):
            join=sub.set_index('subject_id').join(base[['auprc','balanced_acc','macro_f1']],rsuffix='_eda',how='inner')
            delta=join.auprc-join.auprc_eda
            row=dict(task='adult current stress classification',split='LOSO',model=model,
                sensor_combination=sensor,n_subjects=len(join),n_windows=int(join.n_windows.sum()),
                macro_f1_mean=float(join.macro_f1.mean()),auprc_mean=float(join.auprc.mean()),
                balanced_accuracy_mean=float(join.balanced_acc.mean()),delta_vs_baseline=float(delta.mean()),
                baseline='EDA; delta metric = subject AP difference',improved_subjects=int((delta>1e-12).sum()),
                worsened_subjects=int((delta< -1e-12).sum()),tied_subjects=int((delta.abs()<=1e-12).sum()),
                validity_notes='Identical common-QC eligible windows and subject splits; threshold selected on inner validation only.')
            row.update({'delta_ap_'+k:v for k,v in bootstrap(delta).items()})
            rows.append(row)
            for subject,r in join.iterrows():
                paired.append(dict(model=model,sensors=sensor,subject_id=subject,delta_ap=r.auprc-r.auprc_eda,
                    delta_balanced_acc=r.balanced_acc-r.balanced_acc_eda,delta_macro_f1=r.macro_f1-r.macro_f1_eda))
    return pd.DataFrame(rows),pd.DataFrame(paired)

def json_safe(obj):
    if isinstance(obj,dict):return {str(k):json_safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [json_safe(v) for v in obj]
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(np.floating,float)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,Path):return str(obj)
    return obj

def write_json(path,obj):
    Path(path).write_text(json.dumps(json_safe(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def make_figures(out, dataset, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out); dest=out/'figures';dest.mkdir(exist_ok=True)
    per=pd.read_csv(out/'per_subject.csv'); summary=pd.read_csv(out/'summary.csv')
    pred=pd.read_csv(out/'predictions.csv'); eligible=pred[pred.evaluation_eligible.astype(bool)]
    meta=json.loads((out/'metrics.json').read_text())
    caption=(f'{dataset} | {labels} | windows {meta["window_s"]} s / hop {meta["hop_s"]} s | LOSO n={meta["participants"]}\n'
             'Common QC windows; inner-validation threshold | adult stress, NOT ASD forecasting')
    if 'synthetic' in dataset.lower():
        caption += ' | SYNTHETIC ENGINEERING CHECK ONLY'
    def save(fig,name):
        fig.text(.5,.005,caption,ha='center',fontsize=7)
        fig.tight_layout(rect=(0,.07,1,1));fig.savefig(dest/name,dpi=160);plt.close(fig)
    g=per[per.sensors=='all']
    fig,ax=plt.subplots(figsize=(11,4))
    for model,sub in g.groupby('model',sort=False):ax.plot(sub.subject_id,sub.auprc,'o-',label=model)
    ax.set(ylabel='Per-subject AP',title='Unseen-participant performance (all sensors)',ylim=(0,1.03));ax.legend();save(fig,'subject_performance.png')
    g=summary[summary.sensors=='all']
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(g.model,g.auprc_mean)
    ax.errorbar(np.arange(len(g)),g.auprc_mean,yerr=[np.maximum(0,g.auprc_mean-g.auprc_ci_low),np.maximum(0,g.auprc_ci_high-g.auprc_mean)],fmt='none',color='black')
    ax.set(ylabel='Participant-mean AP',title='Model comparison; descriptive subject-bootstrap 95% CI',ylim=(0,1.03));save(fig,'model_comparison.png')
    g=summary[summary.model=='lr']
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(g.sensors,g.auprc_mean)
    ax.errorbar(np.arange(len(g)),g.auprc_mean,yerr=[np.maximum(0,g.auprc_mean-g.auprc_ci_low),np.maximum(0,g.auprc_ci_high-g.auprc_mean)],fmt='none',color='black')
    ax.set(ylabel='Participant-mean AP',title='LR sensor ablation; identical eligible windows',ylim=(0,1.03));save(fig,'sensor_ablation.png')
    groups=list(eligible[eligible.sensors=='all'].groupby('model',sort=False))
    fig,axes=plt.subplots(1,max(1,len(groups)),figsize=(4*max(1,len(groups)),4),squeeze=False)
    for ax,(model,g) in zip(axes[0],groups):
        cm=confusion_matrix(g.y,g.pred,labels=[0,1]);ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Nonstress','Stress'],yticklabels=['Nonstress','Stress'],xlabel='Predicted',ylabel='Protocol label',title=model+'; pooled windows')
    save(fig,'confusion_matrix.png')
    fig,ax=plt.subplots(figsize=(8,4))
    for model,g in groups:
        if g.y.nunique()==2:
            pr,re,_=precision_recall_curve(g.y,g.prob);ax.plot(re,pr,label=model)
    if groups:ax.axhline(groups[0][1].y.mean(),linestyle='--',color='gray',label='Positive prevalence')
    ax.set(xlabel='Recall',ylabel='Precision',title='Pooled OOF PR; descriptive secondary result');ax.legend();save(fig,'pr_curve.png')
    quality=pd.read_csv(out/'quality_counts.csv')
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(quality.subject,quality.total_windows,label='All windows',color='#ccd5df')
    ax.bar(quality.subject,quality.eligible_windows,label='Label and common QC eligible',color='#24798c')
    ax.set(ylabel='Window count (not independent sample size)',title='Quality and label inclusion');ax.legend();save(fig,'quality_flow.png')
