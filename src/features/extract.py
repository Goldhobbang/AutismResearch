"""CLI: python -m src.features.extract --root data/raw/WESAD --window 60 --step 15"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import pandas as pd
from src.loaders.wesad import discover, load, EXPECTED_SUBJECTS
from src.features.core import subject_frame


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', default='data/processed/wesad_features.csv')
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--step', type=int, default=15)
    parser.add_argument('--allow-partial', action='store_true',
                        help='Exploratory/test only; default requires all 15 official subjects')
    args = parser.parse_args()
    paths = discover(args.root)
    if not paths:
        raise SystemExit('No original S*/S*.pkl files found. Do not substitute synthetic data.')
    found = {p.stem for p in paths}
    if not args.allow_partial and found != EXPECTED_SUBJECTS:
        raise SystemExit(f'WESAD subject gate failed: missing={sorted(EXPECTED_SUBJECTS-found)}, extra={sorted(found-EXPECTED_SUBJECTS)}')
    frames, audits, quality = [], [], []
    for path in paths:
        tic = time.perf_counter()
        digest = sha256(path)
        data, audit = load(path)
        if audit['duration_spread_s'] > 1.:
            raise ValueError(f'{path.stem}: sensor duration spread >1s; audit alignment first')
        frame = subject_frame(data, path.stem, args.window, args.step)
        if frame.empty:
            raise ValueError(f'{path.stem}: recording shorter than selected window')
        frames.append(frame)
        elapsed = time.perf_counter()-tic
        audits.append(dict(subject=path.stem, path=str(path), sha256=digest,
                           file_bytes=path.stat().st_size, extraction_s=elapsed, **audit))
        base = {'subject':path.stem, 'n_windows':len(frame),
                'n_label_eligible':int((frame.y>=0).sum()), 'n_stress':int((frame.y==1).sum()),
                'n_common_valid':int(frame.qc_common_valid.sum()),
                'feature_ms_p50':float(frame.qc_feature_ms.median()),
                'feature_ms_p95':float(frame.qc_feature_ms.quantile(.95))}
        for sensor in ['eda','ppg','acc']:
            base[f'{sensor}_valid_windows'] = int(frame[f'qc_{sensor}_valid'].sum())
            base[f'{sensor}_flatline_windows'] = int(frame[f'qc_{sensor}_flatline'].sum())
            base[f'{sensor}_feature_nan_fraction'] = float(frame.filter(regex=f'^{sensor}_').isna().mean().mean())
        quality.append(base)
        del data
        print(path.stem, len(frame), 'windows', round(elapsed,2), 's', flush=True)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(target,index=False)
    pd.DataFrame(quality).to_csv(target.with_suffix('.quality.csv'),index=False)
    target.with_suffix('.manifest.json').write_text(json.dumps(dict(
        dataset='WESAD', version='original ICMI 2018 archive', window_s=args.window,
        hop_s=args.step, interval='[start,end)', units='native source units; no automatic ACC g conversion',
        rates_hz={'EDA':4,'BVP':64,'ACC':32,'label':700},
        task='adult laboratory stress vs baseline+amusement', platform=platform.platform(),
        python=platform.python_version(), subjects=audits),indent=2),encoding='utf-8')
    print('Saved',target)


if __name__ == '__main__':
    main()
