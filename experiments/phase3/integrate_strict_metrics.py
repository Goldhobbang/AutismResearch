"""Recompute strict calibration report from genuine locally generated OOF rows."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/phase3_strict/calibration"
REPORT = ROOT / "reports/PHASE3_STRICT_CALIBRATION_RESULTS.md"
POLICIES = ("cold_fixed05", "cold_original_tuned", "delta_strict_fixed05",
            "delta_guard_cold090_exploratory")

def main():
    path = DATA / "per_subject.csv"
    pred_path = DATA / "post_calibration_predictions.csv"
    if not path.exists() or not pred_path.exists():
        raise FileNotFoundError(
            "Strict results missing. Run experiment_d_calibration_strict.py "
            "on local authentic WESAD before generating reported metrics."
        )
    df, pred = pd.read_csv(path), pd.read_csv(pred_path)
    if set(POLICIES) != set(df.policy.unique()):
        raise ValueError("Unexpected policy set")
    if df.groupby("policy").subject.nunique().ne(15).any():
        raise ValueError("Incomplete 15-subject LOSO")
    num = df.groupby("policy").agg(
        macro_stress_f1=("stress_f1", "mean"),
        macro_recall=("stress_recall", "mean"),
        macro_fpr=("fpr", "mean"),
        macro_ap=("ap", "mean"),
        macro_auroc=("auroc", "mean"),
        false_alarms=("false_alert_episodes", "sum"),
        observed_nonstress_equiv_hours=("negative_window_equivalent_hours", "sum"),
        events=("stress_episodes", "sum"),
        detected_events=("stress_episodes_detected", "sum"),
    )
    num["fa_per_equiv_hour"] = num.false_alarms / num.observed_nonstress_equiv_hours
    num["event_detection_fraction"] = num.detected_events / num.events
    macro_f1 = {}
    for policy, g in pred.groupby("policy"):
        by_sub = []
        for _, sg in g.groupby("subject"):
            y = sg.phase.eq(2).astype(int)
            by_sub.append(f1_score(y, sg.pred, average="macro", zero_division=0))
        macro_f1[policy] = float(np.mean(by_sub))
    num["macro_two_class_f1"] = pd.Series(macro_f1)
    num.to_csv(DATA / "strict_source_derived_metrics.csv")
    entries = [
        "# WESAD 엄격한 300초 보정 재검증 결과", "",
        "실행 데이터: 실 WESAD 로컬 특징 + 동결된 기존 OOF 예측.",
        "평가 구간: 최초 연속 Baseline 300초 종료 후 **새로 시작하는** 60초 창만.",
        "15명 LOSO, 보정에 평가 참가자의 미래 신호/스트레스 라벨 사용 금지.", "",
        "## 재계산 결과", "",
        "| 정책 | Macro F1 (2-class) | Stress F1 | FPR | AP (macro) | AUROC (macro) | Event 검출률 | FAR/동등시간 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for k in POLICIES:
        r = num.loc[k]
        entries.append(
            f"| {k} | {r.macro_two_class_f1:.4f} | {r.macro_stress_f1:.4f} |"
            f" {r.macro_fpr:.4f} | {r.macro_ap:.4f} | {r.macro_auroc:.4f} |"
            f" {r.event_detection_fraction:.4f} | {r.fa_per_equiv_hour:.2f} |"
        )
    s2 = df.loc[df.subject.eq("S2"), ["policy", "stress_f1", "stress_recall", "fpr"]]
    entries += [
        "", "## S2 실패 방지 검토", "", s2.to_csv(index=False).strip(), "",
        "Cold Start 확률 0.90 이상 OR 정책은 탐색적 비교군이다. "
        "오경보 증가 위험이 있으며 보증된 안전정책이 아니다.",
        "", "## 주의사항", "",
        "- 보정에 이용한 처음 300초는 평가에서 전부 제외했다.",
        "- 시간당 오경보 분모는 비스트레스 적격 창 × 15초의 **창 동등 관측시간**이다.",
        "- WESAD 현재 스트레스 분류는 ASD 사건 60초 선행 예측 성능이 아니다.",
        "- 외부 다중 참가자 재검증 전까지 이 결과는 WESAD 내부 검증이다.",
        ""
    ]
    REPORT.write_text("\n".join(entries), encoding="utf-8")
    print("Report:", REPORT)
    print(num.to_string())

if __name__ == "__main__":
    main()
