import time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from src.loaders.wesad import RATES, LABEL_RATE, validate
from src.quality.checks import signal_quality


def native_slice(values, start, end, rate):
    """Sample timestamp n/rate must satisfy start <= n/rate < end."""
    if start < 0 or end <= start or rate <= 0:
        raise ValueError('Invalid time interval or frequency')
    lo = int(np.ceil(start * rate - 1e-10))
    hi = int(np.ceil(end * rate - 1e-10))
    return np.asarray(values[lo:hi], dtype=float)


def basic(values, rate, prefix):
    x = np.asarray(values, dtype=float).reshape(-1)
    ok = np.isfinite(x)
    z = x[ok]
    names = ['mean', 'median', 'std', 'range', 'q10', 'q90', 'slope', 'absdiff_per_s', 'energy']
    if len(z) < 3:
        return {prefix + k: np.nan for k in names}
    t = np.arange(len(x))[ok] / rate
    tc = t - t.mean()
    delta = np.diff(x)
    delta = delta[np.isfinite(delta)]
    result = [z.mean(), np.median(z), z.std(), np.ptp(z), *np.quantile(z, [.1, .9]),
              np.dot(tc, z-z.mean()) / max(np.dot(tc, tc), 1e-12),
              np.mean(np.abs(delta)) * rate if len(delta) else np.nan, np.mean(z*z)]
    return dict(zip([prefix+k for k in names], map(float, result)))


def spectrum(values, rate, lo, hi):
    x = np.asarray(values).reshape(-1)
    if not np.isfinite(x).all() or len(x) < 16 or x.std() < 1e-8:
        return np.nan, np.nan
    freq, power = welch(x, fs=rate, nperseg=min(len(x), int(rate*32)), detrend='constant')
    band = (freq >= lo) & (freq <= hi)
    if not band.any():
        return np.nan, np.nan
    return float(freq[band][np.argmax(power[band])]), float(power[band].sum()/max(power.sum(), 1e-12))


def window_features(wrist, start, end, rates=None):
    """Only [start,end) observations. Native units; no clinical HR/HRV claims."""
    rates = rates or RATES
    chunks = {k: native_slice(wrist[k], start, end, fs) for k, fs in rates.items()}
    eda, bvp, acc = chunks['EDA'].reshape(-1), chunks['BVP'].reshape(-1), chunks['ACC']
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError('ACC must have three axes')
    out = {}
    for key, values in [('eda', eda), ('ppg', bvp), ('acc', acc)]:
        out.update(signal_quality(values, key))
    out.update(basic(eda, rates['EDA'], 'eda_'))
    delta = np.diff(eda)
    out['eda_positive_diff_sum'] = float(np.maximum(delta[np.isfinite(delta)], 0).sum())
    # Candidate peak count ONLY: no tonic/phasic separation, not a validated SCR detector.
    out['eda_scr_candidate_count'] = float(len(find_peaks(eda, prominence=.05,
        distance=max(1, int(rates['EDA'])))[0])) if np.isfinite(eda).all() else np.nan
    out.update(basic(bvp, rates['BVP'], 'ppg_'))
    out['ppg_peak_hz'], out['ppg_band_fraction'] = spectrum(bvp, rates['BVP'], .7, 3.5)
    mag = np.linalg.norm(acc, axis=1)
    out.update(basic(mag, rates['ACC'], 'acc_mag_'))
    for i, axis in enumerate('xyz'):
        out.update(basic(acc[:, i], rates['ACC'], f'acc_{axis}_'))
    # Remove each axis's within-history mean, not an orientation-aware gravity estimate.
    if np.isfinite(acc).all():
        dynamic = acc - acc.mean(axis=0)
        out['acc_dynamic_energy'] = float(np.mean(np.sum(dynamic**2, axis=1)))
        jerk = np.linalg.norm(np.diff(acc, axis=0), axis=1)
        out['acc_motion_change_per_s'] = float(jerk.mean() * rates['ACC']) if len(jerk) else np.nan
        # Exact unchanged raw samples, not a physical stationary-state classifier.
        out['acc_unchanged_fraction'] = float(np.mean(jerk <= 1e-8)) if len(jerk) else np.nan
    else:
        for key in ['acc_dynamic_energy', 'acc_motion_change_per_s', 'acc_unchanged_fraction']:
            out[key] = np.nan
    out['acc_peak_hz'], out['acc_band_fraction'] = spectrum(mag, rates['ACC'], .1, 10)
    for sensor in ['eda', 'ppg', 'acc']:
        if not out[f'qc_{sensor}_valid']:
            for key in list(out):
                if key.startswith(sensor+'_'):
                    out[key] = np.nan
    out['qc_common_valid'] = all(out[f'qc_{s}_valid'] for s in ['eda', 'ppg', 'acc'])
    return out


def label_window(labels, start, end):
    selected = native_slice(labels, start, end, LABEL_RATE).reshape(-1)
    expected = int(np.ceil(end*LABEL_RATE-1e-10)-np.ceil(start*LABEL_RATE-1e-10))
    phase = int(selected[0]) if (len(selected) == expected and len(selected) and
                               np.all(selected == selected[0])) else 0
    y = 1 if phase == 2 else 0 if phase in (1, 3) else -1
    return phase, y


def subject_frame(data, subject, window=60, step=15):
    if int(window) != window or int(step) != step or window <= 0 or step <= 0:
        raise ValueError('window and step must be positive integer seconds')
    audit = validate(data)
    rows = []
    for end in range(int(window), int(audit['duration_s'])+1, int(step)):
        start = end-window
        phase, y = label_window(data['label'], start, end)
        tic = time.perf_counter()
        features = window_features(data['signal']['wrist'], start, end)
        elapsed = (time.perf_counter()-tic)*1000
        rows.append(dict(dataset='WESAD', subject=subject, session=subject,
                         start_s=start, end_s=end, step_s=step, phase=phase, y=y,
                         qc_label_reason='' if y >= 0 else 'mixed_or_unselected_label',
                         qc_feature_ms=elapsed, **features))
    return pd.DataFrame(rows)
