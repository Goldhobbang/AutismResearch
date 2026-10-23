"""Publish completed real-data outputs into the master-spec result contract.

Reject synthetic runs. Existing narrative reports retain their dated execution status.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='results/wesad')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = Path(args.source).resolve()
    data = json.loads((source/'metrics.json').read_text())
    if 'synthetic' in str(source).lower() or 'synthetic' in str(data.get('dataset_version', '')).lower():
        raise ValueError('Synthetic checks cannot populate real research results.')
    if data.get('participants', 0) < 1:
        raise ValueError('No evaluated participants.')
    per = pd.read_csv(source/'per_subject.csv')
    if 'synthetic' in ' '.join(per.dataset.astype(str).unique()).lower():
        raise ValueError('Synthetic dataset label is not real evidence.')
    if per.duplicated(['subject_id', 'model', 'sensors']).any():
        raise ValueError('Duplicate held-out participant results.')
    summary = pd.read_csv(source/'summary.csv')
    target = root/'results'
    (target/'figures').mkdir(parents=True, exist_ok=True)
    for name in ['metrics.json', 'per_subject.csv', 'ablation.csv']:
        shutil.copy2(source/name, target/name)
    for figure in (source/'figures').glob('*.png'):
        shutil.copy2(figure, target/'figures'/figure.name)
    lines = ['# 직접 실행 결과 / 성인 스트레스 분류', '',
             f'생성 UTC: {datetime.now(timezone.utc).isoformat()}', '',
             f"데이터: {data['dataset_version']}. 평가 참가자: {data['participants']}명. "
             f"분할: {data['outer_split']}. 아동 멜트다운 예측 성능은 N/A.", '',
             '| 모델 | 센서 | 참가자 평균 AP | stress F1 | macro-F1 | balanced accuracy |',
             '|---|---|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.model} | {row.sensors} | {row.auprc_mean:.4f} | {row.f1_mean:.4f} | {row.macro_f1_mean:.4f} | {row.balanced_acc_mean:.4f} |')
    lines += ['', 'AP는 average precision이다. 참가자별 분포·CI·제외율·음성조건별 오류는 원본 CSV를 함께 확인한다.',
              '이 파일 생성 이전의 FINAL_REPORT_KO.md는 당시 실행 상태를 기록한 문서다. 이 결과를 검토해 발표·보고서의 해당 항목만 갱신한다.']
    (root/'reports/RESULTS_CURRENT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Published real adult-data results; ASD event metrics remain N/A.')


if __name__ == '__main__':
    main()
