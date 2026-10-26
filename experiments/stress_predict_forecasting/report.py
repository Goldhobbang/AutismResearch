"""Reproducible Korean report and final artifact integrity checks."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.stress_predict_forecasting.run import OUT, RAW, REPORT, ROOT, PROTOCOL, dump, sha


def table(frame, digits=3):
    """Small dependency-free Markdown tables."""
    frame = frame.copy()
    def fmt(x):
        if pd.isna(x):
            return "산출 불가"
        if isinstance(x, (float, np.floating)):
            return f"{x:.{digits}f}"
        return str(x).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(map(str,frame.columns)) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)


def wilson(success, n):
    if not n:
        return (np.nan, np.nan)
    p, z = success/n, 1.95996398454
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-radius, center+radius


def make_report():
    features = pd.read_csv(OUT / "features.csv.gz")
    events = pd.read_csv(OUT / "events.csv")
    stable = pd.read_csv(OUT / "stable_intervals.csv")
    predictions = pd.read_csv(OUT / "predictions.csv.gz")
    warnings = pd.read_csv(OUT / "event_early_warnings.csv")
    summary = pd.read_csv(OUT / "model_comparison.csv")
    participants = pd.read_csv(OUT / "per_subject_metrics.csv")
    splits = pd.read_csv(OUT / "fold_splits.csv")
    audit = json.loads((OUT / "subject_audit.json").read_text(encoding="utf-8"))
    provenance = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
    checks = json.loads((OUT / "validation_checks.json").read_text(encoding="utf-8"))

    # Determine provenance of stable labels, without adding any such field to X.
    support = features[features.y.isin([0,1])][["subject","pred_ts","history_start_ts","y","target_event_id"]].copy()
    support["stable_support"] = "unassigned"
    for r in stable.itertuples():
        mask = (support.subject==r.subject) & (support.history_start_ts>=r.stable_start_ts) & (support.pred_ts<=r.stable_end_ts)
        support.loc[mask,"stable_support"] = r.source
    assert not support.stable_support.eq("unassigned").any()
    support.to_csv(OUT / "window_label_provenance.csv", index=False)
    composition = support.groupby("stable_support").y.agg(positive=lambda x:int((x==1).sum()), negative=lambda x:int((x==0).sum())).reset_index()
    composition.to_csv(OUT / "label_support_composition.csv", index=False)

    # New warning timestamps across the entire recording, including excluded
    # epochs, remain available for independent auditing of the warning policy.
    issued = predictions[predictions.newly_issued_warning].copy()
    issued["evaluation_status"] = issued.y.map({0:"false_on_verified_negative_tick",1:"in_target_forecast_horizon",-1:"excluded_or_unknown_epoch"})
    issued.to_csv(OUT / "newly_issued_warning_log.csv", index=False)
    operational = issued.groupby(["model","sensors","evaluation_status"]).size().rename("count").reset_index()
    operational.to_csv(OUT / "warning_exposure_audit.csv", index=False)

    # Validate a serialized held-out LR pipeline against training-only statistics
    # and saved predictions. This tests the fitted object, not just split labels.
    chosen = splits[(splits.model=="LR") & (splits.sensors=="all")].iloc[0]
    saved = joblib.load(OUT / "models" / f"LR_all_{chosen.test_subject}.joblib")
    train = features[features.subject.isin(chosen.train_subjects.split(";")) & features.y.isin([0,1])]
    model, columns = saved["model"], saved["features"]
    imputed = model.steps[0][1].transform(train[columns])
    checks["serialized_scaler_fit_training_only"] = bool(np.allclose(imputed.mean(axis=0),model.steps[1][1].mean_))
    test = features[(features.subject==chosen.test_subject) & features.sensor_valid].sort_values("pred_ts")
    recorded = predictions[(predictions.subject==chosen.test_subject) & (predictions.model=="LR") & (predictions.sensors=="all") & predictions.sensor_valid].sort_values("pred_ts")
    checks["serialized_model_reproduces_heldout_probabilities"] = bool(np.allclose(model.predict_proba(test[columns])[:,1],recorded.prob))
    checks["forbidden_metadata_not_model_features"] = all(c.startswith(("eda_","hr_","acc_")) for c in columns) and len(columns)==36
    checks["all_models_have_disjoint_fit_subjects"] = True
    for p in (OUT / "models").glob("*.joblib"):
        obj = joblib.load(p)
        checks["all_models_have_disjoint_fit_subjects"] &= obj["held_out_subject"] not in set(obj["training_subjects"])|set(obj["validation_subjects"]) and not set(obj["training_subjects"])&set(obj["validation_subjects"])
    checks["raw_repository_unchanged"] = not subprocess.check_output(["git","-C",str(RAW),"status","--porcelain"],text=True).strip()
    checks["all_expected_subject_model_predictions_present"] = predictions.groupby(["model","sensors"]).subject.nunique().eq(35).all().item()
    checks["no_duplicate_prediction_ticks"] = not predictions.duplicated(["model","sensors","subject","pred_ts"]).any()
    checks["all_detected_leads_strictly_60_to_180"] = bool(warnings.loc[warnings.detected_new,"lead_time_s"].between(60,180,inclusive="right").all())
    # A long active warning is not counted as newly issued at every positive tick.
    dummy = warnings[(warnings.model=="Dummy") & warnings.forecast_eligible]
    checks["dummy_active_warning_not_misreported_as_event_detection"] = bool(dummy.detected_active.all() and not dummy.detected_new.any())
    checks["all_original_events_retained_in_event_ledger"] = warnings.groupby(["model","sensors"]).size().eq(len(events)).all().item()
    dump(OUT / "validation_checks.json", checks)
    assert all(checks.values()), checks

    # Paired subject bootstrap, no refitting or selection. CIs are descriptive;
    # overlapping windows do not constitute independent evidence.
    pivot = participants.pivot(index="subject",columns=["model","sensors"],values="auroc")
    contrast = []
    rng = np.random.default_rng(42)
    for other in [("Dummy","all"),("Schedule","time_only_diagnostic")]:
        for name in ["LR","RF"]:
            delta = (pivot[(name,"all")]-pivot[other]).dropna().to_numpy()
            samples = rng.choice(delta,(2000,len(delta)),replace=True).mean(axis=1)
            contrast.append(dict(model=name, comparator=other[0], paired_subjects=len(delta),
                mean_subject_auroc_difference=float(delta.mean()), ci95_low=float(np.quantile(samples,.025)),ci95_high=float(np.quantile(samples,.975))))
    contrast = pd.DataFrame(contrast)
    contrast.to_csv(OUT / "paired_subject_auroc_contrasts.csv", index=False)
    for i,r in summary.iterrows():
        lo,hi = wilson(r.detected_new,r.eligible_events)
        summary.loc[i,"event_recall_new_ci95_low"] = lo
        summary.loc[i,"event_recall_new_ci95_high"] = hi
    summary.to_csv(OUT / "model_comparison.csv",index=False)

    per = features[features.y.isin([0,1])].groupby("subject").y.agg(positive=lambda x:int((x==1).sum()),negative=lambda x:int((x==0).sum()))
    per = per.reindex([a["subject"] for a in audit],fill_value=0)
    for name in ("LR","RF"):
        m = participants[(participants.model==name)&(participants.sensors=="all")].set_index("subject")
        per[name+" AUROC"] = m.auroc
        per[name+" 신규탐지/적격사건"] = m.detected_new.astype(int).astype(str)+"/"+m.eligible_events.astype(int).astype(str)
        per[name+" 오경보/시간"] = m.false_alarms_per_hour
    per.reset_index().to_csv(OUT / "participant_report_table.csv",index=False)

    event_copy = events.copy()
    for c in ["event_start_ts","event_end_ts","log_start_ts","log_end_ts"]:
        event_copy[c.replace("_ts","_utc")] = pd.to_datetime(event_copy[c],unit="s",utc=True).astype(str)
    event_copy.to_csv(OUT / "event_boundary_alignment.csv",index=False)
    valid_events = int(events.boundary_valid.sum())
    eligible_events = int(events.forecast_eligible.sum())
    npos,nneg = int(features.y.eq(1).sum()),int(features.y.eq(0).sum())
    primary = summary[(summary.model=="LR")&(summary.sensors=="all")].iloc[0]
    rf = summary[(summary.model=="RF")&(summary.sensors=="all")].iloc[0]
    schedule = summary[summary.model=="Schedule"].iloc[0]
    correction_list = [(a["subject"],r["index"],r["original"],r["corrected"]) for a in audit for r in a["tag_corrections"]]
    comparison = summary[["model","sensors","auroc","auprc","subject_macro_auroc","detected_new","eligible_events","event_recall_new","event_recall_active","false_alarms_per_hour","median_lead_s"]].copy()
    comparison.columns = ["모델","센서","pooled AUROC","AUPRC(AP)","참가자 평균 AUROC","신규탐지","적격사건","신규 탐지율","Active 탐지율","오경보/시간","선행시간 중앙값(초)"]
    task_counts = events.groupby("task").agg(전체=("event_id","size"),경계검증=("boundary_valid","sum"),예측적격=("forecast_eligible","sum"),양성창=("n_positive_windows","sum")).reset_index()
    reasons = features.label_reason.value_counts().rename_axis("원인").reset_index(name="창 수")
    uncovered = events[events.boundary_valid & ~events.forecast_eligible][["subject","task","n_positive_windows"]]
    excluded = events[~events.boundary_valid][["subject","task","boundary_reason"]]
    examples = []
    for name in ("LR","RF"):
        w = warnings[(warnings.model==name)&(warnings.sensors=="all")&warnings.forecast_eligible]
        for label, selection in [("성공",w[w.detected_new].head(2)),("실패: Active만 존재",w[~w.detected_new&w.detected_active].head(2)),("실패: 경보 없음",w[~w.detected_new&~w.detected_active].head(1))]:
            for r in selection.itertuples():
                examples.append(dict(모델=name,사례=label,사건=r.event_id,선행시간초=r.lead_time_s,Active=r.detected_active,최대확률=r.max_horizon_probability))
    example_table = table(pd.DataFrame(examples))
    fp_examples = issued[(issued.sensors=="all") & issued.model.isin(["LR","RF"]) & issued.y.eq(0)].head(4).copy()
    fp_examples["UTC"] = pd.to_datetime(fp_examples.pred_ts,unit="s",utc=True).astype(str)
    fp_table = table(fp_examples[["model","subject","UTC","pred_rel_s","prob"]])
    raw_order_errors = [a["subject"] for a in audit if not a["tags_ordered_raw"]]
    checks_text = "\n".join(f"- `{k}`: 통과" for k in checks)
    strict = composition[composition.stable_support.str.startswith("tag_bounded")][["positive","negative"]].sum()
    all_case_subjects = int(features.loc[features.y.isin([0,1]),"subject"].nunique())
    main_rows = summary[(summary.sensors=="all") & summary.model.isin(["LR","RF"])].copy()
    ci_text = "\n".join(f"- {r.model}: 신규 {int(r.detected_new)}/{int(r.eligible_events)} = {r.event_recall_new:.1%}, Wilson 95% 구간 {r.event_recall_new_ci95_low:.1%}–{r.event_recall_new_ci95_high:.1%}; 선행시간 평균 {r.mean_lead_s:.1f}초, 중앙값 {r.median_lead_s:.1f}초." for r in main_rows.itertuples())

    content = f"""# Stress-Predict 미래 스트레스 과제 시작 예측 실험

작성일: 2026-10-27 (Asia/Seoul). CPU scikit-learn 실데이터 실행 결과. 기존 WESAD/WEPH 및 Nurses 결과는 수정하지 않았다.

## 판정

**미래 스트레스 예측 성공을 입증하지 못했다.** 미래 과제 시작을 목표로 하는 인과적 입력 구성과 참가자 독립 평가는 구현·실행했지만, 기본 LR의 신규 조기경보는 {int(primary.detected_new)}/{eligible_events} ({primary.event_recall_new:.1%}), RF는 {int(rf.detected_new)}/{eligible_events} ({rf.event_recall_new:.1%})에 그쳤다. LR/RF의 AUROC {primary.auroc:.3f}/{rf.auroc:.3f}는 시간표 진단 기준선 {schedule.auroc:.3f}보다 낮다. 생체신호에 시간표를 넘어서는 실제 미래 예측 정보가 있다고 결론 내릴 수 없다.

이것은 현재 스트레스 상태 분류 실험이 아니다. 타깃은 **60초 초과~180초 이내 시작될 예정된 실험 과제**이다. 과제 시작 마커는 생리적 스트레스 반응이 정확히 발생한 시점이 아니며, 회복 구간도 프로토콜상 휴식일 뿐 심리적 무스트레스가 증명된 상태는 아니다.

## 1. 원본 확보와 참가자

- 공식 원본: [Stress-Predict GitHub](https://github.com/italha-d/Stress-Predict-Dataset), shallow clone(`--depth 1`). 기존 원본이 있으면 다운로드를 건너뛴다. 원본 Git 작업트리 변경 없음.
- 고정 커밋: `{provenance['commit']}`.
- 원본 경로: `data/raw/Stress-Predict-Dataset/`. 참가자 다운로드 **35명**, 센서/태그/시간 로그 읽기·감사 **35명**, 파일 처리 실패 **0명**.
- 유효 지도학습 창이 있는 참가자는 **{all_case_subjects}명**. S06은 경계 누락과 짧은 연결 휴식 때문에 지도학습 창이 0개다. 그 참가자도 외부 테스트 예측과 전체 검증 사건 분모에는 포함했다.
- EDA, 기기 제공 HR, ACC만 사용했다. BVP, IBI, TEMP, 설문 및 가공된 정답 CSV는 모델 입력·라벨 생성에 사용하지 않았다.
- 환경: Python {provenance['python']}, numpy {provenance['numpy']}, pandas {provenance['pandas']}, scikit-learn {provenance['sklearn']}. 기존 `.venv` 재사용, 신규 패키지/GPU/PyTorch 설치 없음.

## 2. 시간축과 사건 경계 검증

센서 CSV의 첫 행은 UTC UNIX 초, 둘째 행은 샘플링 주파수로 읽었다. EDA **4 Hz**, HR **1 Hz**, ACC **32 Hz**를 35명 전원 검사했다. ACC는 원본 1/64 g를 g로 변환했다. 채널별 시작시각을 따로 유지하며, HR은 모든 참가자에서 EDA/ACC보다 **10초 늦게 시작**한다. 이 차이를 실제 샘플시각으로 보정했으며 행 번호로 강제 정렬하거나 미래 값으로 채우지 않았다.

시간 로그의 날짜 및 시각은 `Europe/Dublin`으로 해석했다. 실제 수집일(2022-02-07~03-09)은 모두 UTC+0이다. 시간 로그에 섞인 12시간/24시간 표기는 센서 기록의 실제 날짜·시간 범위에 맞는 AM/PM 후보로 해석했다. 한국 시간대로 원본을 이동시키지 않았다.

태그 파일의 순서를 보존했다. 원본 순서 이상 참가자는 {', '.join(raw_order_errors)}이다. 센서 범위 밖인 태그가 ±12시간 이동으로 단 하나의 센서 범위 내 후보를 가질 때만 명시적으로 복구했다. 복구는 {len(correction_list)}개(S17 2개, S18 1개, S30 1개)이며 원시 값과 복구 값은 `subject_audit.json`에 저장했다. 실제 과제 길이나 예측 성능으로 복구한 것이 아니다.

각 과제의 로그 시작·종료와 실제 버튼 마커를 **120초 이내**에서 대조했다. 가장 가까운 후보와 차이가 30초 미만인 경쟁 후보가 있으면 불확실로 제외하고, 마커 중복 사용·역순을 거부했다. 120초는 마커와 과제 이름을 연결하기 위한 허용차이며 정답 시작시각을 ±120초로 흔드는 설정이 아니다. 시작/종료 정답은 매칭된 실제 마커다. 매칭 자체의 의미상 불확실성은 여전히 한계이며 영상 등 독립 검증은 없다.

검증된 사건의 로그 대비 마커 차이는 시작 **{events.loc[events.boundary_valid,'start_log_delta_s'].min():.1f}~{events.loc[events.boundary_valid,'start_log_delta_s'].max():.1f}초**, 종료 **{events.loc[events.boundary_valid,'end_log_delta_s'].min():.1f}~{events.loc[events.boundary_valid,'end_log_delta_s'].max():.1f}초**였다. 고정 5분/10분 과제 길이를 라벨로 사용하지 않았다. S01의 초기 baseline 로그가 Stroop 뒤에 놓인 순서 오류는 초기 안정 구간 제외로 처리했다.

{table(task_counts)}

총 105개 과제 중 **{valid_events}개 경계 검증**, 7개 불확실 경계 제외. 경계를 검증했지만 유효 예측 창이 없는 사건은 **{valid_events-eligible_events}개**, 실제 예측 평가 적격 사건은 **{eligible_events}개**다. 전체 105개가 모두 검증된 것으로 보고하지 않는다.

제외 사건:

{table(excluded)}

경계는 검증했지만 유효 전조 창이 없는 사건:

{table(uncovered)}

## 3. 실제 안정 구간과 학습 창

두 과제 사이 휴식은 **이전 과제 종료 태그→다음 과제 시작 태그**가 실제로 존재하고, 해당 시간 로그의 Relax 시작/종료와 각각 120초 이내 일치할 때만 사용했다. 초기·최종 baseline은 마커가 없는 경우 시간 로그에 명시된 구간의 양끝을 각각 60초 제외했다. 이 60초는 분 단위 로그의 경계를 피하기 위한 보수적 규칙으로 사전에 고정했으며, 태그로 완전히 검증된 경계와 동일한 확실성은 없다. 로그가 부정확하거나 휴식 중 스트레스가 남았다면 여전히 라벨 오염 가능성이 있다.

입력 이력 180초 전체가 이 안정 구간 안에 들어와야 지도학습에 포함하는 추가 보수 조건을 적용했다. 그 결과 과거 과제 수행 신호를 이용한 현재 스트레스 분류로 바뀌지 않는다. 예측 시점의 정답이 이미 시작된 과제인 양성 창은 **0개**이다.

- 각 기록의 최초 센서시각에 고정한 `180, 210, 240, ...초` 그리드, 30초 간격. 사건 위치에 따라 그리드를 조밀하게 만들거나 이동하지 않았다.
- 관측 `[t−180, t)`만 사용한다. `t` 이후 값, 미래 정답, 참가자 ID, 과제 순서, 남은 시간은 특징에 없다.
- 양성: 확인된 미래 과제 시작까지 `60 < lead ≤ 180초`.
- 음성: 안정 구간 및 향후 180초 추적이 확인되고 그 안에 과제 시작이 없음. 180초 이내 과제가 있지만 60초 이하 남으면 제외한다.
- 센서 유효 비율 95% 미만, 연속 결측 5초 초과, EDA 지속적 0 수준 등은 제외한다. 원본 센서 값의 NaN/물리범위 이탈은 35명 모두 0%였다. 품질 제외 35개 창은 HR의 10초 늦은 시작 때문에 첫 180초 창에 충분한 HR 이력이 없는 경우다.
- 원본은 시작시각·샘플링률을 가진 규칙 샘플 열이므로, 명시되지 않은 장치 내부 패킷 유실·시계 드리프트까지 이 파일만으로 독립 검증할 수는 없다.

총 그리드 **{len(features)}개**, 지도학습 창 **{npos+nneg}개**(양성 **{npos}**, 음성 **{nneg}**), 양성 비율 **{npos/(npos+nneg):.1%}**. 제외 **{len(features)-npos-nneg}개**. 180초 이력과 180초 음성 추적을 동시에 요구하면 약 5분 휴식에서 음성 창이 거의 남지 않는다는 구조적 한계가 있다.

{table(reasons)}

안정 구간 근거별 실제 라벨 구성:

{table(composition)}

**태그로 양끝이 고정된 휴식만 남기는 엄격 해석에서는 양성 {int(strict.positive)}개, 음성 {int(strict.negative)}개뿐이다.** 이 부분만으로 신뢰할 만한 참가자 독립 이진 예측 평가를 할 수 없다. 주 평가의 음성 대부분은 시간 로그 기반 초기 baseline에서 나온다. 따라서 여기서 얻은 점수는 모든 경계가 태그로 입증된 실생활 예측 성능이 아니며, 휴식 종류·과제 순서·기록 경과시간을 구분하는 단서에 의존할 수 있다.

## 4. 특징·모델·분리

기존 `src.features.core.basic/native_slice`와 `src.forecasting.evaluate.compute_window_metrics`를 재사용했다. EDA, HR 각각 6개, ACC x/y/z/크기 각각 6개로 총 **36개 기본 통계**다: 평균, 표준편차, 범위, 선형 기울기, 초당 절대 변화량, 마지막 30초 평균−첫 30초 평균. 스펙트럼·HRV·딥러닝·추가 특징 탐색은 하지 않았다.

Dummy(prior), LR(`C=0.25`, balanced), RF(200 trees, depth 6, leaf 최소 8, balanced_subsample)의 설정은 기존 Nurses 기본 모델과 동일하다. 별도 경로에서 학습했으며 LightGBM은 실행하지 않았다. 센서 단독 비교는 LR만 EDA/HR/ACC 각각 수행했다. 임의 모델 선택/대규모 탐색은 없다.

외부 **35-fold LOSO**에서 테스트 참가자를 완전히 제외한다. 나머지 유효 창 보유 참가자는 내부 GroupShuffleSplit(검증 20%, seed 42+fold)로 학습/검증을 다시 분리한다. 훈련 참가자는 fold당 {splits.train_subjects.str.split(';').map(len).min()}~{splits.train_subjects.str.split(';').map(len).max()}명, 검증은 {splits.validation_subjects.str.split(';').map(len).min()}~{splits.validation_subjects.str.split(';').map(len).max()}명이다. 중앙값 결측 대치와 StandardScaler는 훈련 참가자에만 fit한다. 검증 지표는 별도 기록했지만 **모든 임계값은 사전 고정 0.5**이고 검증/테스트 최적화가 없다. 검증 참가자로 재학습하지 않았으므로 학습·검증·테스트가 엄격히 분리된다.

## 5. 경보 정책과 성능

**Active Warning**은 해당 시점 확률이 0.5 이상인 상태다. **Newly Issued Warning**은 이전 유효 시점의 비활성 상태에서 활성으로 전환되는 상승 에지 한 번이다. 최초 관측에서 높으면 그 시점에 한 번 발령한다. 센서 결측은 이전 내부 상태를 유지하여 복구 때 가짜 신규 발령을 만들지 않는다. 쿨다운·히스테리시스·사건별 초기화는 없다.

정책은 라벨이 없는 구간과 과제 진행 중을 포함해 **전체 기록의 센서 유효 그리드**에서 인과적으로 연속 실행했다. 테스트 정답으로 경보를 켜고 끄거나 사건 직전 강제 초기화하지 않았다. 따라서 3분보다 일찍 발령된 경보가 계속 켜져 있으면 목표 구간의 Active에는 들어가지만 신규 탐지 성공에는 들어가지 않는다. 목표 구간 내 최초 신규 발령만 선행시간으로 집계한다.

{table(comparison)}

AUROC/AUPRC는 모든 LOSO held-out 창을 모은 micro 지표이며, AUPRC는 sklearn Average Precision이다. 각 참가자 AUROC 평균도 별도 제시했다. Dummy의 pooled AUROC가 정확히 0.5가 아닌 이유는 fold별 훈련 유병률이 달라 참가자 간 확률 상수가 달라지기 때문이다. 참가자 내부에서 양 클래스가 있는 Dummy AUROC는 0.5다.

{ci_text}

이 신뢰구간은 사건 단위 Wilson 기술통계이며 참가자 내 사건 상관을 보정한 임상 불확실성 구간은 아니다. 겹치는 창을 독립 표본으로 간주하지 않는다. 참가자 단위 AUROC 차이의 paired bootstrap(2,000회, seed 42)도 저장했다:

{table(contrast)}

분모를 낙관적으로 축소하지 않도록 **경계 검증 전체 {valid_events}개 사건** 기준 신규 탐지율도 저장했다. LR {int(primary.detected_new)}/{valid_events} = {primary.event_recall_all_validated:.1%}, RF {int(rf.detected_new)}/{valid_events} = {rf.event_recall_all_validated:.1%}. 경계 미검증 7개까지 모두 포함한 원래 105개에 대한 관측 탐지 하한은 LR {int(primary.detected_new)/105:.1%}, RF {int(rf.detected_new)/105:.1%}이며, 미검증 사건의 정확한 성공/실패 판정은 할 수 없다.

Dummy는 유병률이 50%를 넘어서 거의 전 기록에 활성 상태를 유지했고 적격 사건의 Active 탐지율은 **100%**, 신규 탐지율은 **0%**였다. Active를 신규 경보로 잘못 세면 성능을 크게 부풀리는 실제 사례다.

### 오경보율의 분모와 제한

음성으로 확인된 {nneg}개 30초 평가 틱 = **{nneg*30/3600:.3f}시간**을 명목 관찰시간으로 사용한다. 음성 틱에서 새로 발령된 경보만 오경보로 센다. LR은 {int(primary.false_new_warnings)}회/{primary.verified_negative_hours:.3f}시간 = **{primary.false_alarms_per_hour:.3f}회/시간**, RF는 {int(rf.false_new_warnings)}회 = **{rf.false_alarms_per_hour:.3f}회/시간**이다. 이는 이산 그리드 기준의 검증 음성 노출률이며 연속 실시간 관찰시간을 정확히 복원한 값은 아니다.

과제 진행 중, 60초 이하 경고, 불확실·검열 구간의 새 경보를 무조건 오경보로 판정하지 않았다. 모든 발령은 `newly_issued_warning_log.csv`에 보존하고 그 구간들을 제외 상태로 기록했다. 따라서 위 수치는 전체 기록 또는 일상생활에 대한 오경보율이 아니며, 음성 관찰시간이 매우 짧아 불안정하다. 음성 창이 없는 S03/S06/S22의 참가자별 오경보율은 산출 불가다.

## 6. 시간표 편향 점검

생체신호 모델 입력에는 경과시간이 없지만, 과제·회복에 따른 생체신호가 간접적으로 시간표를 드러낼 수 있다. 이를 확인하기 위해 **기록 시작 이후 경과시간만** 사용하는 120초 bin 경험적 양성 비율 기준선을 훈련 참가자로만 만들었다(훈련 prior 4개 상당 Laplace 수축). 테스트 참가자의 과제 시각·순서·ID·태그를 예측 입력으로 사용하지 않았다. 이 기준선은 편향 진단용으로만 따로 저장했으며 EDA/HR/ACC 모델에 합치지 않았다.

시간표 기준선의 AUROC **{schedule.auroc:.3f}**, AP **{schedule.auprc:.3f}**, 신규 탐지 **{int(schedule.detected_new)}/{eligible_events}**, 오경보 **{schedule.false_alarms_per_hour:.3f}/시간**이다. 단순 시간표만으로 양성/음성 창을 잘 구분한다. 높은 AUROC가 곧 신규 발령에 성공한다는 뜻도 아니다. 생체신호 결합 모델은 이 기준선을 넘어서는 판별 성능을 보이지 않았다. EDA만의 AP가 결합 LR보다 높고 ACC만의 신규 탐지율이 결합 LR보다 높은 것도 생리적 전조의 일관된 우월성을 뒷받침하지 않는다. 센서별 차이는 이 실험에 한정된 기술적 비교이며 다중비교 유의성을 주장하지 않는다.

## 7. 참가자별 결과와 사례

{table(per.reset_index())}

양성 또는 음성만 있는 참가자의 AUROC/AP는 두 클래스 비교가 불가능하므로 결측으로 둔다. S06은 지도학습 평가 창 0개지만 전체 센서 유효 시점의 예측을 저장했다.

대표 사건(고성능 사례만 고르지 않고 사건 ID 순으로 최초 사례 선택):

{example_table}

`실패: Active만 존재`는 목표 구간에서 경보가 켜져 있었지만 그 구간 안에 신규 발령이 없다는 뜻이다. `경보 없음`은 유효 목표 창에서도 경보가 없었던 사건이다. 모델별 모든 105개 사건의 경계 상태·적격 여부·신규/Active 탐지·최초 발령·선행시간은 `event_early_warnings.csv`에 저장했다.

실제 음성 틱의 신규 오경보 예:

{fp_table}

## 8. 검증 기록과 재현

{checks_text}

최초 실행 중 유효 지도학습 창이 없는 S06의 빈 지표 입력을 기존 sklearn 함수가 거부하는 문제가 발견되어, 신규 실험의 집계 코드에서만 빈 참가자를 0개 창/지표 결측으로 처리했다. S06을 사건 분모에서 제거하지 않았다. 시간표 기준선도 생체신호 모델과 동일하게 센서 불충분 틱을 차단한 뒤 최종 재평가했다. 기존 공용 모듈과 기존 실험 결과는 수정하지 않았다.

실행 명령(저장소 루트):

```powershell
.venv/Scripts/python.exe -m experiments.stress_predict_forecasting.run
```

이미 생성된 특징으로 학습: `--stage train`. 학습 없이 보고서·검증 재생성: `--stage report`. 특징은 참가자별 캐시하며 원본 파일 시각/크기, 커밋, 로그 및 추출 코드 해시가 바뀌면 무효화한다. 참가자 단위 CPU 병렬 처리(최대 6개)이며 불필요한 그래프는 생성하지 않았다.

주요 산출물:

- `results/stress_predict_forecasting/feature_cache/`, `features.csv.gz`: 참가자별 캐시와 전체 고정 그리드 특징/라벨.
- `events.csv`, `event_boundary_alignment.csv`, `stable_intervals.csv`, `subject_audit.json`, `failed_subjects.json`: 실제 경계, 시간 정합, 보정, 실패 원인.
- `models/`: 35 folds × 7 모델/센서 조건 = **245개** 학습 아티팩트. 3종 필수 모델과 LR 센서 비교, 시간표 진단 포함. 전체 참가자에 재학습한 배포 모델은 만들지 않았다.
- `predictions.csv.gz`, `per_subject_metrics.csv`, `event_early_warnings.csv`, `newly_issued_warning_log.csv`: held-out 예측, 참가자별 성능, 사건별 최초 신규 경보와 전체 발령 기록.
- `model_comparison.csv`, `validation_metrics.csv`, `fold_splits.csv`, `validation_checks.json`, `paired_subject_auroc_contrasts.csv`: 최종 성능과 누수 방지 증거.
- `protocol.json`, `provenance.json`, `artifact_manifest.json`: 사전 고정 규칙, 원본/환경/코드 및 결과 해시.

## 9. 해석 범위

이 데이터는 건강한 성인이 정해진 순서의 Stroop·인터뷰·과호흡 과제를 수행한 실험이다. 기기 HR 자체도 BVP로부터 제조사가 산출한 평균값이며 이것을 검증된 미래 스트레스 바이오마커로 간주하지 않는다. 전조처럼 보이는 신호는 과제 안내, 이전 과제 회복, 자세 변화, 실험 경과시간을 반영할 수 있다. 버튼 마커는 과제 시작 표지이고 실제 주관적·생리적 스트레스 발생 순간과 동일하지 않다.

**예정된 과제 시작 예측**, **일상에서 자연발생하는 스트레스**, **자폐인의 멜트다운**은 타깃·모집단·환경·정답 측정이 다르다. 이번 결과로 자연발생 스트레스나 자폐 멜트다운 예측 성능을 주장하거나 임상 적용을 권고할 수 없다. 이번 작업은 미래 예측 파이프라인의 실행과 한계 확인을 완료한 것이며, 미래 스트레스 예측 모델의 실용적 성공은 입증하지 못했다.

원 데이터 설명 및 인용: [공식 README](https://github.com/italha-d/Stress-Predict-Dataset#readme), [Iqbal et al., Sensors 2022, 22(21), 8135](https://doi.org/10.3390/s22218135).
"""
    REPORT.write_text(content,encoding="utf-8")
    provenance["final_run_script_sha256"] = sha(ROOT / "experiments/stress_predict_forecasting/run.py")
    provenance["report_script_sha256"] = sha(__file__)
    provenance["protocol_sha256"] = sha(OUT / "protocol.json")
    provenance["models_saved"] = len(list((OUT / "models").glob("*.joblib")))
    dump(OUT / "provenance.json",provenance)
    artifacts = [p for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="artifact_manifest.json"]
    artifacts += [REPORT, ROOT / "experiments/stress_predict_forecasting/run.py", Path(__file__)]
    dump(OUT / "artifact_manifest.json", {str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in artifacts})
    print(f"Report written: {REPORT}; {len(checks)} integrity checks passed",flush=True)


if __name__ == "__main__":
    make_report()
