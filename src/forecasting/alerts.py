"""Alert policies, episode merging, and event-level evaluation for Nurses Stress forecasting.

Strictly separates window-level classification metrics from event-level early warning metrics:
- Merges overlapping or adjacent predicted positive windows into discrete Alert Episodes.
- Never double-counts multiple warnings for the same upcoming stress event.
- Excludes censored ('UNKNOWN_NA_CENSORED', 'ONGOING_STRESS_EVENT', 'POST_EVENT_REFRACTORY')
  periods from false alarm denominator and counts.
- Computes Event Recall, False Alarm Episodes, False Alarms/hour (over verified non-event hours),
  and Lead Time statistics (mean, median, min, max, >=60s count, >=180s count).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.advanced import apply_dynamic_motion_gate, compute_causal_exercise_score


def apply_alert_policy_to_session(
    probs: np.ndarray,
    end_s: np.ndarray,
    val_threshold: float,
    policy: str = "val_opt",
    df_session: pd.DataFrame | None = None,
) -> np.ndarray:
    """Apply operational alert policy causally within a single session.

    Policies:
    - 'fixed_05': p >= 0.50
    - 'val_opt': p >= val_threshold
    - 'hysteresis': turn ON when p >= max(val_threshold, 0.55), turn OFF when p < 0.35
    - 'duration_2win': require 2 consecutive windows >= val_threshold
    - 'cooldown_300s': require p >= val_threshold, then suppress new triggers for 300s after episode
    - 'soft_gate_hysteresis': apply soft_dynamic motion gate + hysteresis
    """
    p = np.asarray(probs, dtype=float)
    t_arr = np.asarray(end_s, dtype=float)
    n = len(p)
    out = np.zeros(n, dtype=int)

    if policy == "fixed_05":
        return (p >= 0.50).astype(int)

    if policy == "val_opt":
        return (p >= val_threshold).astype(int)

    if policy == "hysteresis":
        t_on = max(val_threshold, 0.52)
        t_off = min(val_threshold * 0.70, 0.35)
        active = 0
        for i in range(n):
            if active == 0 and p[i] >= t_on:
                active = 1
            elif active == 1 and p[i] < t_off:
                active = 0
            out[i] = active
        return out

    if policy == "duration_2win":
        streak = 0
        for i in range(n):
            if i > 0 and (t_arr[i] - t_arr[i - 1]) > 120.0:
                streak = 0
            if p[i] >= val_threshold:
                streak += 1
            else:
                streak = 0
            out[i] = 1 if streak >= 2 else 0
        return out

    if policy == "cooldown_300s":
        last_alert_t = -1e12
        for i in range(n):
            if p[i] >= val_threshold and (t_arr[i] - last_alert_t) >= 300.0:
                out[i] = 1
                last_alert_t = t_arr[i]
        return out

    if policy == "soft_gate_hysteresis":
        if df_session is not None:
            p_gated, dyn_thr, _ = apply_dynamic_motion_gate(
                p, df_session, mode="soft_dynamic", base_threshold=val_threshold
            )
        else:
            p_gated = p
            dyn_thr = np.full(n, val_threshold)
        active = 0
        for i in range(n):
            t_on = max(dyn_thr[i], 0.50)
            t_off = min(dyn_thr[i] * 0.70, 0.35)
            if active == 0 and p_gated[i] >= t_on:
                active = 1
            elif active == 1 and p_gated[i] < t_off:
                active = 0
            out[i] = active
        return out

    raise ValueError(f"Unknown alert policy: {policy}")


def merge_alert_episodes(
    pred_df: pd.DataFrame,
    pred_col: str = "pred",
    merge_gap_s: float = 180.0,
) -> list[dict]:
    """Merge consecutive or nearby positive window predictions (within `merge_gap_s`) into discrete Alert Episodes."""
    episodes = []
    for (sid, sess), sdf in pred_df.groupby(["subject", "session"], sort=False):
        sdf = sdf.sort_values("pred_ts")
        pos = sdf[sdf[pred_col] == 1]
        if pos.empty:
            continue

        cur_start_ts = None
        cur_end_ts = None
        cur_max_prob = 0.0
        cur_states = []
        cur_n_win = 0

        for _, r in pos.iterrows():
            ts = float(r["pred_ts"])
            pr = float(r.get("prob", 1.0))
            st = str(r.get("state", ""))
            if cur_start_ts is None:
                cur_start_ts = ts
                cur_end_ts = ts
                cur_max_prob = pr
                cur_states = [st]
                cur_n_win = 1
            elif (ts - cur_end_ts) <= merge_gap_s:
                cur_end_ts = ts
                cur_max_prob = max(cur_max_prob, pr)
                cur_states.append(st)
                cur_n_win += 1
            else:
                episodes.append(
                    {
                        "subject": sid,
                        "session": sess,
                        "alert_start_ts": cur_start_ts,
                        "alert_end_ts": cur_end_ts,
                        "n_windows": cur_n_win,
                        "max_prob": cur_max_prob,
                        "states": cur_states,
                    }
                )
                cur_start_ts = ts
                cur_end_ts = ts
                cur_max_prob = pr
                cur_states = [st]
                cur_n_win = 1

        if cur_start_ts is not None:
            episodes.append(
                {
                    "subject": sid,
                    "session": sess,
                    "alert_start_ts": cur_start_ts,
                    "alert_end_ts": cur_end_ts,
                    "n_windows": cur_n_win,
                    "max_prob": cur_max_prob,
                    "states": cur_states,
                }
            )
    return episodes


def evaluate_event_level_forecasting(
    pred_df: pd.DataFrame,
    eligible_events_df: pd.DataFrame,
    pred_col: str = "pred",
    max_valid_lead_s: float = 300.0,
    min_valid_lead_s: float = 0.0,
    merge_gap_s: float = 180.0,
) -> dict:
    """Evaluate event-level early warning metrics on held-out predictions.

    Definitions:
    - An eligible confirmed stress event with onset `t_onset` is DETECTED if at least one positive
      prediction occurs at `t` inside `[t_onset - max_valid_lead_s, t_onset - min_valid_lead_s]`.
    - Lead time for a detected event is `t_onset - t_first_alert`, where `t_first_alert` is the
      earliest positive prediction in `[t_onset - max_valid_lead_s, t_onset]`.
    - Alert episodes occurring entirely inside verified non-event / confirmed no-stress-0 regions
      (and not matching any confirmed event within `[t_onset - 300s, t_end]`) are counted as
      False Alarm Episodes.
    - Denominator for False Alarms/hour (`verified_non_event_hours`) is the sum of non-overlapping
      step durations of evaluated negative windows (`VERIFIED_NON_EVENT` and `CONFIRMED_NO_STRESS_0`).
    """
    evs = eligible_events_df.copy()
    n_total_events = len(evs)

    # Calculate verified non-event observation hours
    neg_rows = pred_df[pred_df["state"].isin(["VERIFIED_NON_EVENT", "CONFIRMED_NO_STRESS_0"])]
    # Each background step represents 60s (or 30s near events); compute exact non-overlapping coverage per session
    non_event_seconds = 0.0
    for (sid, sess), sg in neg_rows.groupby(["subject", "session"], sort=False):
        if len(sg) == 1:
            non_event_seconds += 60.0
        else:
            ts_sorted = np.sort(sg["pred_ts"].to_numpy(dtype=float))
            diffs = np.diff(ts_sorted)
            non_event_seconds += float(np.sum(np.minimum(diffs, 60.0)) + 60.0)
    non_event_hours = max(non_event_seconds / 3600.0, 1e-6)

    # Evaluate each ground-truth event
    detected_count = 0
    lead_times = []
    lead_ge_60 = 0
    lead_ge_180 = 0

    for _, ev in evs.iterrows():
        sid = str(ev["subject"])
        t_onset = float(ev["event_start_ts"])
        sub_p = pred_df[(pred_df["subject"] == sid) & (pred_df[pred_col] == 1)]
        # Windows inside valid early warning horizon [t_onset - max_valid_lead_s, t_onset - min_valid_lead_s]
        in_horizon = sub_p[
            (sub_p["pred_ts"] >= (t_onset - max_valid_lead_s - 1e-3))
            & (sub_p["pred_ts"] <= (t_onset - min_valid_lead_s + 1e-3))
        ]
        if not in_horizon.empty:
            detected_count += 1
            t_first = float(in_horizon["pred_ts"].min())
            lt = float(t_onset - t_first)
            lead_times.append(lt)
            if lt >= 60.0 - 1e-3:
                lead_ge_60 += 1
            if lt >= 180.0 - 1e-3:
                lead_ge_180 += 1

    # Merge positive predictions into discrete alert episodes
    # Filter pred_df to only eligible evaluation windows (exclude UNKNOWN_NA_CENSORED, ONGOING, REFRACTORY)
    eval_pred_df = pred_df[
        pred_df["state"].isin(
            [
                "PRE_ONSET_P0_0_60S",
                "PRE_ONSET_P1_60_180S",
                "PRE_ONSET_P2_180_300S",
                "VERIFIED_NON_EVENT",
                "CONFIRMED_NO_STRESS_0",
            ]
        )
    ]
    episodes = merge_alert_episodes(eval_pred_df, pred_col=pred_col, merge_gap_s=merge_gap_s)

    true_hit_episodes = 0
    false_alarm_episodes = 0

    for ep in episodes:
        sid = ep["subject"]
        a0, a1 = ep["alert_start_ts"], ep["alert_end_ts"]
        sub_evs = evs[evs["subject"] == sid]
        # Does this episode overlap the [t_onset - 300s, t_onset] pre-onset window of any confirmed event?
        hit = False
        for _, ev in sub_evs.iterrows():
            t_onset = float(ev["event_start_ts"])
            if max(a0, t_onset - 300.0) <= min(a1, t_onset):
                hit = True
                break
        if hit:
            true_hit_episodes += 1
        else:
            false_alarm_episodes += 1

    total_alert_episodes = true_hit_episodes + false_alarm_episodes
    alert_precision = (
        float(true_hit_episodes / total_alert_episodes) if total_alert_episodes > 0 else np.nan
    )
    event_recall = float(detected_count / n_total_events) if n_total_events > 0 else np.nan
    missed_count = n_total_events - detected_count
    far_per_hour = float(false_alarm_episodes / non_event_hours)

    lt_arr = np.asarray(lead_times, dtype=float)
    return {
        "n_total_events": int(n_total_events),
        "n_detected_events": int(detected_count),
        "n_missed_events": int(missed_count),
        "event_recall": event_recall,
        "total_alert_episodes": int(total_alert_episodes),
        "true_hit_episodes": int(true_hit_episodes),
        "false_alarm_episodes": int(false_alarm_episodes),
        "verified_non_event_hours": float(round(non_event_hours, 4)),
        "false_alarms_per_hour": far_per_hour,
        "alert_episode_precision": alert_precision,
        "lead_time_mean_s": float(np.mean(lt_arr)) if len(lt_arr) else np.nan,
        "lead_time_median_s": float(np.median(lt_arr)) if len(lt_arr) else np.nan,
        "lead_time_min_s": float(np.min(lt_arr)) if len(lt_arr) else np.nan,
        "lead_time_max_s": float(np.max(lt_arr)) if len(lt_arr) else np.nan,
        "n_events_warned_ge_60s": int(lead_ge_60),
        "n_events_warned_ge_180s": int(lead_ge_180),
        "lead_time_reliability_note": "Minute-rounded survey start timestamp (+-60s resolution)",
    }
