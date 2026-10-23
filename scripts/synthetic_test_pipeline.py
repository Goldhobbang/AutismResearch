"""Full pipeline engineering check. All generated results stay under synthetic_test.

No real WESAD observations or physiological performance are generated here.
"""
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    out = ROOT / 'results' / 'synthetic_test'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for sid in ['S2', 'S3', 'S4', 'S5', 'S6', 'S7']:
        phases = np.repeat([1, 2, 3], 150)
        wrist = {}
        for key, fs, axes in [('EDA', 4, 1), ('BVP', 64, 1), ('ACC', 32, 3)]:
            x = rng.normal(0, .3, (len(phases)*fs, axes))
            x += np.repeat(phases == 2, fs)[:, None] * .12
            wrist[key] = x
        directory = out / 'raw' / sid
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / (sid+'.pkl')).open('wb') as handle:
            pickle.dump({'subject': sid, 'signal': {'wrist': wrist},
                         'label': np.repeat(phases, 700)}, handle)
    features = out / 'synthetic_test_features.csv'
    run('-m', 'src.features.extract', '--root', str(out/'raw'), '--out', str(features),
        '--allow-partial', '--window', '60', '--step', '15')
    frame = pd.read_csv(features)
    frame['dataset'] = 'synthetic_test'
    frame.to_csv(features, index=False)
    manifest_path = features.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    manifest.update(dataset='synthetic_test', version='generated engineering fixture; NOT WESAD',
                    task='software checks only; no physiological performance')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    result = out / 'synthetic_test_evaluation'
    run('-m', 'src.train', '--features', str(features), '--out', str(result),
        '--plan', 'core', '--expected-subjects', '6', '--jobs', '2',
        '--dataset-version', 'synthetic_test_engineering_only')
    run('-m', 'src.replay', '--out', str(result), '--subject', 'S2', '--model', 'lr',
        '--sensors', 'all', '--raw-root', str(out/'raw'))
    per = pd.read_csv(result/'per_subject.csv')
    pred = pd.read_csv(result/'predictions.csv')
    assert len(per) == 6*9
    assert not pred.duplicated(['subject', 'session', 'end_s', 'model', 'sensors']).any()
    assert pred.prob.between(0, 1).all()
    assert len(pd.read_csv(result/'ablation.csv')) == 7
    for split in json.loads((result/'splits.json').read_text()):
        tr, va, te = [set(split[k]) for k in ('train', 'validation', 'test')]
        assert not (tr & va or tr & te or va & te)
    expected = ['confusion_matrix.png', 'subject_performance.png', 'sensor_ablation.png',
                'model_comparison.png', 'quality_flow.png', 'pr_curve.png']
    assert all((result/'figures'/name).is_file() for name in expected)
    print('PASS synthetic_test: extraction, 6-subject LOSO, 9 configurations, 7 sensor sets, outputs, raw replay.')
    print('These are software checks; no real physiological performance was measured.')


if __name__ == '__main__':
    os.chdir(ROOT)
    main()
