import numpy as np

MAX_MISSING = 0.05
FLAT_STD = 1e-8


def signal_quality(values, sensor):
    """Exact/nearly-exact flatline is a heuristic, not contact validation.

    ACC flatline may be immobility and is not itself grounds for invalidation.
    No empirical quality cutoff is tuned on test participants.
    """
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    finite = values[good]
    missing = float(1 - good.mean()) if values.size else 1.
    if sensor == 'acc' and values.ndim == 2 and len(values):
        # Different static gravity offsets across axes do not imply motion.
        axis_flat = [np.std(v[np.isfinite(v)]) <= FLAT_STD
                     for v in values.T if np.isfinite(v).any()]
        flat = bool(axis_flat and all(axis_flat))
    else:
        flat = bool(len(finite) > 0 and np.std(finite) <= FLAT_STD)
    valid = bool(values.size >= 3 and missing <= MAX_MISSING + 1e-12
                 and (sensor == 'acc' or not flat))
    reasons = []
    if values.size < 3:
        reasons.append('too_short')
    if missing > MAX_MISSING + 1e-12:
        reasons.append('nonfinite_gt_5pct')
    if flat and sensor != 'acc':
        reasons.append('flatline_heuristic')
    return {f'qc_{sensor}_valid': valid, f'qc_{sensor}_missing_fraction': missing,
            f'qc_{sensor}_flatline': flat, f'qc_{sensor}_reason': ';'.join(reasons)}
