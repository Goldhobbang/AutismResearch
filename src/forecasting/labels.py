"""Prospective future stress event forecasting label generation and censoring logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Task definitions: (lead_min_s, lead_max_s)
FORECAST_HORIZONS = {
    "P0": (0.0, 60.0),      # Onset in [t, t + 60s] (proximity baseline, NOT >=1 min early forecast)
    "P1": (60.0, 180.0),    # Onset in [t + 60s, t + 180s] (PRIMARY: 1 to 3 min early forecast)
    "P2": (180.0, 300.0),   # Onset in [t + 180s, t + 300s] (EXTENDED: 3 to 5 min early forecast)
}

POST_EVENT_REFRACTORY_S = 900.0  # 15 minutes post-event refractory period
NA_PRE_BUFFER_S = 300.0          # Censor 5 min before unconfirmed 'na' candidate events
NA_POST_BUFFER_S = 600.0         # Censor 10 min after unconfirmed 'na' candidate events


def classify_prediction_timestamp(
    t_abs: float,
    confirmed_events: list[tuple[str, float, float]],
    na_events: list[tuple[str, float, float]],
    no_stress_0_events: list[tuple[str, float, float]],
) -> dict:
    """Classify a single prediction timestamp `t_abs` (Unix UTC epoch) for P0, P1, P2 tasks.

     strictly enforces temporal separation:
    - Observation window is [t_abs - W, t_abs), strictly before t_abs.
    - Confirmed stress events have Stress level in {1, 2}.
    - Unconfirmed 'na' events are censored so they are never treated as negative baseline.
    """
    # 1. Check if t_abs is inside an ongoing confirmed stress event [t_start, t_end]
    for ev_id, ts, te in confirmed_events:
        if ts <= t_abs <= te:
            return {
                "state": "ONGOING_STRESS_EVENT",
                "target_event_id": ev_id,
                "lead_time_to_onset_s": t_abs - ts,
                "y_P0": -1,
                "y_P1": -1,
                "y_P2": -1,
            }

    # 2. Check if t_abs is inside or near an unconfirmed ('na') candidate event
    for ev_id, ts, te in na_events:
        if (ts - NA_PRE_BUFFER_S) <= t_abs <= (te + NA_POST_BUFFER_S):
            return {
                "state": "UNKNOWN_NA_CENSORED",
                "target_event_id": ev_id,
                "lead_time_to_onset_s": np.nan,
                "y_P0": -1,
                "y_P1": -1,
                "y_P2": -1,
            }

    # 3. Check future confirmed stress onsets > t_abs
    future_onsets = [(ev_id, ts, ts - t_abs) for ev_id, ts, te in confirmed_events if ts > t_abs]
    if future_onsets:
        # Closest upcoming confirmed stress event
        next_ev_id, next_ts, delta_s = min(future_onsets, key=lambda x: x[2])
    else:
        next_ev_id, next_ts, delta_s = "", np.nan, np.inf

    # Check if in post-event refractory period of a previous confirmed stress event
    in_refractory = False
    for ev_id, ts, te in confirmed_events:
        if te < t_abs <= (te + POST_EVENT_REFRACTORY_S):
            in_refractory = True
            break

    # If a future onset is within 300s, pre-onset takes priority over post-previous refractory
    if delta_s <= 300.0:
        if 0.0 < delta_s <= 60.0:
            state = "PRE_ONSET_P0_0_60S"
            y_p0, y_p1, y_p2 = 1, -1, -1
        elif 60.0 < delta_s <= 180.0:
            state = "PRE_ONSET_P1_60_180S"
            y_p0, y_p1, y_p2 = -1, 1, -1
        else:  # 180.0 < delta_s <= 300.0
            state = "PRE_ONSET_P2_180_300S"
            y_p0, y_p1, y_p2 = -1, -1, 1
        return {
            "state": state,
            "target_event_id": next_ev_id,
            "lead_time_to_onset_s": float(delta_s),
            "y_P0": y_p0,
            "y_P1": y_p1,
            "y_P2": y_p2,
        }

    # If in refractory period after a previous stress event (and not within 300s of a new one)
    if in_refractory:
        return {
            "state": "POST_EVENT_REFRACTORY",
            "target_event_id": "",
            "lead_time_to_onset_s": float(delta_s) if np.isfinite(delta_s) else np.nan,
            "y_P0": -1,
            "y_P1": -1,
            "y_P2": -1,
        }

    # Check if inside a pre-event buffer (300s < delta_s <= 600s) -> transition zone, censor from clean negative
    if 300.0 < delta_s <= 600.0:
        return {
            "state": "PRE_ONSET_BUFFER_300_600S",
            "target_event_id": next_ev_id,
            "lead_time_to_onset_s": float(delta_s),
            "y_P0": -1,
            "y_P1": -1,
            "y_P2": -1,
        }

    # Check if inside a survey-confirmed non-stress interval (Stress level == 0)
    in_confirmed_zero = any(ts <= t_abs <= te for _, ts, te in no_stress_0_events)
    state_name = "CONFIRMED_NO_STRESS_0" if in_confirmed_zero else "VERIFIED_NON_EVENT"

    return {
        "state": state_name,
        "target_event_id": "",
        "lead_time_to_onset_s": float(delta_s) if np.isfinite(delta_s) else np.nan,
        "y_P0": 0,
        "y_P1": 0,
        "y_P2": 0,
    }


def build_session_prediction_grid(
    session_start_ts: float,
    session_end_ts: float,
    subject_events_df: pd.DataFrame,
    window_s: float = 180.0,
    event_Zone_step_s: float = 30.0,
    background_step_s: float = 60.0,
) -> list[dict]:
    """Build causal prediction timestamps `t` for an annotated session.

    Uses 30s step near any survey event (from t_start - 900s to t_end + 900s) for high-resolution
    lead-time tracking, and 60s step across verified non-event background periods.
    """
    if (session_end_ts - session_start_ts) < (window_s + 60.0):
        return []

    confirmed_ev = []
    na_ev = []
    zero_ev = []

    for _, r in subject_events_df.iterrows():
        eid = str(r["event_id"])
        ts = float(r["event_start_ts"])
        te = float(r["event_end_ts"])
        cat = str(r["label_category"])
        if cat in ("confirmed_high_stress", "confirmed_moderate_stress"):
            confirmed_ev.append((eid, ts, te))
        elif cat == "survey_confirmed_no_stress":
            zero_ev.append((eid, ts, te))
        else:
            na_ev.append((eid, ts, te))

    all_ev_bounds = [(ts - 900.0, te + 900.0) for _, ts, te in (confirmed_ev + zero_ev + na_ev)]

    t_cur = session_start_ts + window_s
    t_max = session_end_ts - 5.0
    rows = []

    while t_cur <= t_max:
        rel_s = t_cur - session_start_ts
        info = classify_prediction_timestamp(t_cur, confirmed_ev, na_ev, zero_ev)
        rows.append(
            {
                "pred_ts": float(t_cur),
                "pred_rel_s": float(rel_s),
                "start_s": float(rel_s - window_s),
                "end_s": float(rel_s),
                **info,
            }
        )
        near_any = any(b0 <= t_cur <= b1 for b0, b1 in all_ev_bounds)
        step = event_Zone_step_s if near_any else background_step_s
        t_cur += step

    return rows
