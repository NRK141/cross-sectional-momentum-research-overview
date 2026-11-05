from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import json
import math
import shutil
import sys
import os
import traceback

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data_access import (
    HistoryStore,
    load_config,
    load_feature_history,
    load_normalized_calibration_backfill,
    merge_calibration_history,
    load_spec,
    verify_frozen_production_identity,
)
from research_protocol import verify_holdout_freeze
from reporting import (
    cross_sectional_ic,
    cross_sectional_ic_by_source,
    non_overlapping_forecast_frame,
    equity_risk_diagnostics,
    portfolio_metrics,
    realized_quantile_diagnostics,
    selection_lift_diagnostics,
    trade_concentration_diagnostics,
    trade_metrics,
    write_json,
)
from stochastic_core import (
    Calibration,
    adaptive_entry_forecast,
    adaptive_recheck_forecast,
    calibrate_heston_merton,
    regularize_calibration_cross_section,
    stable_seed,
)


@dataclass
class ModelPosition:
    permno: int
    ticker: str
    sector: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    entry_notional: float
    entry_weight: float
    entry_cost: float
    model_score: float
    forecast_terminal_volatility: float
    forecast_expected_return: float
    signal_source: str = "UNKNOWN"
    held_days: int = 0
    negative_drift_rechecks: int = 0
    last_seen_date: pd.Timestamp | None = None
    last_seen_price: float | None = None
    missing_global_sessions: int = 0


def _calibration_at(
    permno: int,
    date: pd.Timestamp,
    history: HistoryStore,
    spec: dict,
    cache: dict,
) -> Calibration | None:
    key = (int(permno), pd.Timestamp(date))
    if key in cache:
        return cache[key]
    m = spec["model"]
    trailing = history.trailing_returns(int(permno), date, int(m["max_calibration_observations"]))
    cal = calibrate_heston_merton(
        trailing,
        min_observations=int(m["min_calibration_observations"]),
        max_observations=int(m["max_calibration_observations"]),
        variance_window=int(m["variance_window_days"]),
        jump_sigma_threshold=float(m["jump_sigma_threshold"]),
        drift_short_half_life_days=float(m["drift_short_half_life_days"]),
        drift_long_half_life_days=float(m["drift_long_half_life_days"]),
        drift_short_weight=float(m["drift_short_weight"]),
        drift_shrinkage=float(m["drift_shrinkage"]),
        max_abs_annual_drift=float(m["max_abs_annual_drift"]),
    )
    cache[key] = cal
    return cal


def _terminal_forecast(
    permno: int,
    ticker: str,
    date: pd.Timestamp,
    price: float,
    history: HistoryStore,
    spec: dict,
    cal_cache: dict,
    forecast_cache: dict,
    *,
    horizon_days: int,
    n_paths: int,
    purpose: str,
    cal_override: Calibration | None = None,
) -> dict | None:
    """Patch 2.9.3 forecast adapter.

    Entry forecasts use next-close entry + close-stop payoff semantics. Position
    rechecks are already-held mark-to-horizon forecasts and therefore do not
    apply a synthetic next-close entry transformation.
    """
    cal = cal_override if cal_override is not None else _calibration_at(permno, date, history, spec, cal_cache)
    if cal is None:
        return None
    is_recheck = "recheck" in str(purpose).lower()
    stop_pct = float(spec["standalone"].get("emergency_stop_pct", 0.025))
    cal_sig = (
        round(float(cal.mu), 12), round(float(cal.v0), 12), round(float(cal.theta), 12),
        round(float(cal.kappa), 12), round(float(cal.xi), 12), round(float(cal.rho), 12),
        round(float(cal.lambda_jump), 12), round(float(cal.jump_mu_log), 12), round(float(cal.jump_sigma_log), 12),
    )
    key = (
        purpose, int(permno), pd.Timestamp(date), int(horizon_days), int(n_paths),
        "RECHECK" if is_recheck else "ENTRY", round(stop_pct, 8), cal_sig,
    )
    if key in forecast_cache:
        return forecast_cache[key]

    context_key = f"{purpose}|{pd.Timestamp(date).date()}|h{int(horizon_days)}|{'recheck' if is_recheck else 'entry'}"
    if is_recheck:
        stats = adaptive_recheck_forecast(
            float(price), cal, spec, horizon_days=int(horizon_days),
            requested_paths=int(n_paths), purpose=purpose, context_key=context_key,
        )
    else:
        stats = adaptive_entry_forecast(
            float(price), cal, spec, horizon_days=int(horizon_days),
            requested_paths=int(n_paths), stop_pct=stop_pct, purpose=purpose, context_key=context_key,
        )
    seed = stable_seed(int(spec["random_seed"]), purpose, pd.Timestamp(date).date(), int(horizon_days), round(stop_pct, 8))
    out = {**cal.to_dict(), **stats, "simulation_seed": int(seed), "forecast_stop_pct": stop_pct}
    forecast_cache[key] = out
    return out

def _drift_prefilter_pass(cal: Calibration, spec: dict) -> bool:
    """Patch 2.9.3: calibration availability only; drift sign is NOT a hard gate."""
    return cal is not None


def _prefilter_score(cal: Calibration, spec: dict) -> float:
    """Use uncertainty-shrunk physical drift as a soft ranking input, never a sign gate."""
    return float(cal.mu)


def _rank01(values: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if not higher_is_better:
        x = -x
    r = x.rank(method="average", pct=True)
    return r.fillna(0.5).clip(0.0, 1.0)


def _signal_alpha_frame(g: pd.DataFrame) -> pd.DataFrame:
    """Causal cross-sectional momentum/pullback alpha from signal-close features.

    No Heston-Merton forecast statistic enters this score.  All inputs are known
    at the signal close and entry remains the following close.
    """
    x = g.copy()
    def num(name: str) -> pd.Series:
        if name not in x:
            return pd.Series(np.nan, index=x.index, dtype=float)
        return pd.to_numeric(x[name], errors="coerce")
    close = num("close")
    sma50, sma100, sma200 = num("sma50"), num("sma100"), num("sma200")
    ret5, rsi2, dist20 = num("ret5"), num("rsi2"), num("distance_20d_high")
    trend100 = close / sma100.replace(0.0, np.nan) - 1.0
    trend200 = close / sma200.replace(0.0, np.nan) - 1.0
    trend50_200 = sma50 / sma200.replace(0.0, np.nan) - 1.0
    trend = (_rank01(trend100) + _rank01(trend200) + _rank01(trend50_200)) / 3.0

    # Pullback: short-term weakness/oversold behavior inside the established trend.
    # Clipping prevents a crash outlier from winning simply because it is the most negative.
    ret5_pull = -ret5.clip(lower=-0.10, upper=0.03)
    pullback = 0.60 * _rank01(rsi2, higher_is_better=False) + 0.40 * _rank01(ret5_pull)

    # Prefer a moderate retreat from the 20-day high (~5%) rather than either
    # zero pullback or an extreme collapse.  Cross-sectional ranking keeps units robust.
    distance_target = -(dist20.clip(lower=-0.30, upper=0.05) + 0.05).abs()
    location = _rank01(distance_target)

    out = pd.DataFrame(index=x.index)
    out["alpha_trend_component"] = trend
    out["alpha_pullback_component"] = pullback
    out["alpha_location_component"] = location
    out["alpha_balanced"] = (0.55 * trend + 0.30 * pullback + 0.15 * location).clip(0.0, 1.0)
    out["alpha_trend_heavy"] = (0.70 * trend + 0.20 * pullback + 0.10 * location).clip(0.0, 1.0)
    out["alpha_pullback_heavy"] = (0.40 * trend + 0.45 * pullback + 0.15 * location).clip(0.0, 1.0)
    # Development-only hump-shaped challenger. Extreme short-term collapses and
    # extraordinarily low RSI no longer receive maximum desirability.
    moderate_rsi = _rank01(-(rsi2.clip(0, 100) - 23.0).abs())
    moderate_ret = _rank01(-(ret5.clip(-0.40, 0.25) + 0.035).abs())
    moderate_pull = 0.55 * moderate_rsi + 0.45 * moderate_ret
    out["alpha_hump_pullback"] = (0.50 * trend + 0.35 * moderate_pull + 0.15 * location).clip(0.0, 1.0)
    for name in ("balanced", "trend_heavy", "pullback_heavy", "hump_pullback"):
        out[f"alpha_pct_{name}"] = _rank01(out[f"alpha_{name}"])
    out["alpha_prefilter_score"] = out[["alpha_balanced", "alpha_trend_heavy", "alpha_pullback_heavy", "alpha_hump_pullback"]].max(axis=1)
    return out


def _alpha_columns_for_model(model: str) -> tuple[str, str]:
    m = str(model).upper()
    mapping = {
        "BALANCED": ("alpha_balanced", "alpha_pct_balanced"),
        "TREND_HEAVY": ("alpha_trend_heavy", "alpha_pct_trend_heavy"),
        "PULLBACK_HEAVY": ("alpha_pullback_heavy", "alpha_pct_pullback_heavy"),
        "HUMP_PULLBACK": ("alpha_hump_pullback", "alpha_pct_hump_pullback"),
    }
    if m not in mapping:
        raise ValueError(f"Unsupported alpha_model: {model}")
    return mapping[m]

def _ranking_score(persistent_drift: float, f: dict, spec: dict) -> float:
    """Cross-sectional ranking family selected by the development optimizer."""
    metric = str(spec["standalone"].get("ranking_metric", "PERSISTENT_DRIFT_X_SIGNED_SQRT_MEDIAN_RETURN")).upper()
    if metric in {"PERSISTENT_DRIFT_X_SIGNED_SQRT_MEDIAN_RETURN", "CURRENT_SCORE", "MODEL_SCORE"}:
        # Patch 2.9.3 removes the old multiplicative sign trap: once negative drift
        # is allowed into Monte Carlo, negative drift * negative median would
        # otherwise become a falsely positive score. Put both components in the
        # same horizon-return units and average them instead.
        horizon = float(spec["standalone"].get("forecast_horizon_days", 20))
        drift_horizon = float(persistent_drift) * horizon / 252.0
        med = float(f["median_return"])
        return float(0.5 * drift_horizon + 0.5 * med)
    if metric == "MEDIAN_RETURN":
        return float(f["median_return"])
    if metric == "PROBABILITY_POSITIVE":
        return float(f["probability_positive"])
    if metric == "EXPECTED_SHORTFALL_5":
        return float(f["expected_shortfall_5"])
    if metric == "RISK_ALLOCATION_SCORE":
        score, _ = _risk_allocation_score(f, spec)
        return float(score)
    raise ValueError(f"Unsupported ranking_metric: {metric}")


def _entry_pass(f: dict, spec: dict) -> tuple[bool, str]:
    s = spec["standalone"]
    tests: dict[str, bool] = {}
    if bool(s.get("require_mc_converged", True)) and "mc_converged" in f:
        tests["monte_carlo_converged"] = bool(f["mc_converged"])
    max_mc_se = s.get("max_mc_mean_se")
    if max_mc_se is not None and "mc_mean_se" in f:
        tests["maximum_mc_mean_se"] = float(f["mc_mean_se"]) <= float(max_mc_se)
    min_er = s.get("min_expected_return")
    if min_er is not None:
        tests["minimum_expected_return"] = float(f["expected_return"]) >= float(min_er)
    min_pp = s.get("min_probability_positive")
    if min_pp is not None:
        tests["minimum_probability_positive"] = float(f["probability_positive"]) >= float(min_pp)
    min_es = s.get("min_expected_shortfall_5")
    if min_es is not None:
        tests["minimum_expected_shortfall"] = float(f["expected_shortfall_5"]) >= float(min_es)
    max_l10 = s.get("max_probability_loss_10")
    if max_l10 is not None:
        tests["maximum_loss10_probability"] = float(f["probability_loss_10"]) <= float(max_l10)
    failed = [k for k, ok in tests.items() if not ok]
    return len(failed) == 0, ";".join(failed)


def _risk_allocation_score(f: dict, spec: dict) -> tuple[float, float]:
    """Return (relative allocation score, conservative annualized volatility).

    V4 softens the v3 1/sigma rule to 1/sqrt(sigma).  The score is relative; actual
    target weights are normalized across the selected portfolio to a gross target.
    """
    s = spec["standalone"]
    risk_vol = max(
        float(f["terminal_volatility"]),
        math.sqrt(max(float(f["theta"]), 0.0)),
        float(s["position_volatility_floor"]),
    )
    power = float(s.get("risk_allocation_power", 0.5))
    score = 1.0 / max(risk_vol, 1e-8) ** power
    return float(score), float(risk_vol)


def _capped_proportional_weights(scores: list[float], total: float, cap: float) -> list[float]:
    """Proportional weights with iterative per-name caps."""
    n = len(scores)
    if n == 0 or total <= 0.0:
        return [0.0] * n
    total = min(float(total), float(cap) * n)
    weights = [0.0] * n
    active = set(range(n))
    remaining = total
    while active and remaining > 1e-12:
        denom = sum(max(float(scores[i]), 0.0) for i in active)
        if denom <= 0.0:
            alloc = remaining / len(active)
            trial = {i: alloc for i in active}
        else:
            trial = {i: remaining * max(float(scores[i]), 0.0) / denom for i in active}
        capped = [i for i, w in trial.items() if weights[i] + w > cap + 1e-12]
        if not capped:
            for i, w in trial.items():
                weights[i] += w
            remaining = 0.0
            break
        for i in capped:
            room = max(0.0, cap - weights[i])
            weights[i] += room
            remaining -= room
            active.remove(i)
    return [float(w) for w in weights]


def _allocate_signal_weights(
    candidates: list[dict],
    gross_room: float,
    spec: dict,
    *,
    max_new_entries: int | None = None,
) -> list[dict]:
    """Patch 2.9.3 fixed-target allocation helper.

    Live-style baseline: each new position uses the spec-defined fixed equal target weight. Patch 2.9.3
    optimizes between four 25% positions and five 20% positions at 100% gross.
    If appreciation leaves less room, the new position may use remaining room
    provided it is still above the configured minimum weight.
    """
    s = spec["standalone"]
    min_w = float(s["min_position_weight"])
    fixed_w = float(s.get("fixed_position_weight", s["max_position_weight"]))
    max_n = int(s["max_new_entries_per_rebalance"]) if max_new_entries is None else int(max_new_entries)
    room = max(0.0, float(gross_room))
    out: list[dict] = []
    for candidate in candidates:
        if len(out) >= max_n or room < min_w - 1e-12:
            break
        w = min(fixed_w, room)
        if w < min_w - 1e-12:
            continue
        x = dict(candidate)
        x["target_weight"] = float(w)
        out.append(x)
        room -= w
    return out


def _sector_weights(positions: dict[int, ModelPosition], g: pd.DataFrame, equity_now: float) -> dict[str, float]:
    out: dict[str, float] = {}
    if equity_now <= 0:
        return out
    for perm, pos in positions.items():
        if perm in g.index:
            try:
                px = float(g.loc[perm]["close"])
            except Exception:
                px = float("nan")
        else:
            px = float("nan")
        if not math.isfinite(px) or px <= 0:
            px = float(pos.last_seen_price if pos.last_seen_price is not None else pos.entry_price)
        out[pos.sector] = out.get(pos.sector, 0.0) + float(pos.shares * px / equity_now)
    return out


def _construct_signal_batch(
    candidates: list[dict],
    *,
    current_gross: float,
    current_sector_weights: dict[str, float],
    vacancies: int,
    spec: dict,
    max_new_entries: int,
    date: pd.Timestamp | None = None,
    events: list[dict] | None = None,
) -> list[dict]:
    """Construct a capacity-valid ranked entry batch with deterministic backfill.

    Candidates are considered strictly in model-rank order. A candidate that cannot
    fit gross/sector/min-weight constraints is explicitly skipped and the next-ranked
    candidate is considered. This repairs the old behavior where an infeasible top
    candidate consumed a slot and no lower-ranked stock was backfilled.
    """
    s = spec["standalone"]
    min_w = float(s["min_position_weight"])
    fixed_w = float(s.get("fixed_position_weight", s["max_position_weight"]))
    target_gross = float(s["target_gross_exposure"])
    hard_gross = float(s["max_gross_exposure"])
    sector_cap = float(s["max_sector_weight"])
    max_take = max(0, min(int(vacancies), int(max_new_entries)))
    gross = max(0.0, float(current_gross))
    sectors = {str(k): float(v) for k, v in current_sector_weights.items()}
    selected: list[dict] = []

    def log_skip(c: dict, reason: str, planned_weight: float = 0.0) -> None:
        if events is None:
            return
        events.append({
            "date": date,
            "ticker": c.get("ticker"),
            "permno": c.get("permno"),
            "event": "SIGNAL_CANDIDATE_SKIPPED_CAPACITY",
            "signal_source": c.get("signal_source"),
            "model_score": c.get("model_score"),
            "sector": c.get("sector", "Other"),
            "reason": reason,
            "planned_weight": float(planned_weight),
            "current_gross_before_candidate": float(gross),
        })

    for candidate in candidates:
        if len(selected) >= max_take:
            break
        remaining_target = max(0.0, target_gross - gross)
        remaining_hard = max(0.0, hard_gross - gross)
        if min(remaining_target, remaining_hard) < min_w - 1e-12:
            log_skip(candidate, "NO_GROSS_CAPACITY")
            break
        sector = str(candidate.get("sector", "Other"))
        sector_room = max(0.0, sector_cap - sectors.get(sector, 0.0))
        planned = min(fixed_w, remaining_target, remaining_hard, sector_room)
        if planned < min_w - 1e-12:
            reason = "SECTOR_CAP" if sector_room < min_w - 1e-12 else "BELOW_MIN_POSITION_WEIGHT"
            log_skip(candidate, reason, planned)
            continue
        x = dict(candidate)
        x["target_weight"] = float(planned)
        x["order_rank"] = int(len(selected) + 1)
        selected.append(x)
        gross += planned
        sectors[sector] = sectors.get(sector, 0.0) + planned
    return selected


def _risk_exit(f: dict, spec: dict, prior_negative_rechecks: int = 0) -> tuple[bool, str, int]:
    """Stateful recheck with a Monte-Carlo quality veto before model-risk exits."""
    s = spec["standalone"]
    if bool(s.get("require_mc_converged", True)) and not bool(f.get("mc_converged", True)):
        return False, "MC_UNRESOLVED_NO_ACTION", int(prior_negative_rechecks)
    max_mc_se = s.get("max_mc_mean_se")
    if max_mc_se is not None and float(f.get("mc_mean_se", 0.0)) > float(max_mc_se):
        return False, "MC_SE_TOO_LARGE_NO_ACTION", int(prior_negative_rechecks)
    # Patch 2.9.3: Heston-Merton drift is a risk-distribution input, not alpha.
    # Do not create a second alpha decision by exiting merely because re-estimated
    # physical drift crosses zero.  Only genuinely catastrophic tail deterioration
    # may trigger a model-risk exit.
    streak = 0
    reasons = []
    joint_tail = (
        float(f["expected_shortfall_5"]) < float(s["recheck_joint_tail_expected_shortfall_below"])
        and float(f["probability_loss_10"]) > float(s["recheck_joint_tail_probability_loss_10_above"])
    )
    if joint_tail:
        reasons.append("JOINT_CATASTROPHIC_TAIL")
    return bool(reasons), ";".join(reasons), int(streak)


def _mark_to_market(positions: dict[int, ModelPosition], g: pd.DataFrame) -> float:
    """Mark positions using the current close or the last observed close while a quote is missing.

    Patch 2.9.3 never falls back to entry price for a disappeared security because that can
    erase real P&L accumulated before the quote stream ended. The last observed close is an
    explicit accounting mark only; discontinuation exits are separately flagged.
    """
    value = 0.0
    for perm, pos in positions.items():
        if perm in g.index:
            px = float(g.loc[perm]["close"])
        elif pos.last_seen_price is not None and math.isfinite(float(pos.last_seen_price)):
            px = float(pos.last_seen_price)
        else:
            px = float(pos.entry_price)
        value += pos.shares * px
    return float(value)



def _entry_forecast_types() -> set[str]:
    return {"MONTH_END_ENTRY", "WEEKLY_REFILL_ENTRY"}


def _rebalance_date_sets(all_dates: pd.DatetimeIndex) -> tuple[set[pd.Timestamp], set[pd.Timestamp]]:
    """Return last available trading date of each month and week."""
    if len(all_dates) == 0:
        return set(), set()
    s = pd.Series(all_dates, index=all_dates)
    month_end = set(s.groupby(all_dates.to_period("M")).max().tolist())
    week_end = set(s.groupby(all_dates.to_period("W-FRI")).max().tolist())
    return month_end, week_end


def _last_full_horizon_signal_date(
    all_dates: pd.DatetimeIndex,
    max_holding_days: int,
) -> pd.Timestamp | None:
    """Latest global signal date that can fit next-close entry + full max hold.

    Entry executes on the first session after the signal and entry day counts as
    held day 1. Therefore a H-session maximum hold needs at least H global
    sessions after the signal date. This helper uses calendar availability only,
    never future price/return values.
    """
    h = int(max_holding_days)
    if h <= 0:
        raise ValueError("max_holding_days must be positive")
    if len(all_dates) <= h:
        return None
    return pd.Timestamp(all_dates[-(h + 1)])


def _discover_entry_candidates(
    *,
    date: pd.Timestamp,
    g: pd.DataFrame,
    positions: dict[int, ModelPosition],
    history: HistoryStore,
    spec: dict,
    cal_cache: dict,
    forecast_cache: dict,
    blocked_entries: set[tuple[int, pd.Timestamp]],
    forecast_rows: list[dict],
    events: list[dict],
    forecast_type: str,
    prefilter_n: int,
    n_paths: int,
    purpose: str,
) -> list[dict]:
    """Fresh model-owned cross-sectional discovery for a single signal date."""
    s = spec["standalone"]
    raw_candidates: list[tuple[int, pd.Series, Calibration]] = []
    raw_map: dict[int, Calibration] = {}
    for perm, row in g.iterrows():
        perm = int(perm)
        if perm in positions or not bool(row.get("eligible", False)):
            continue
        price = float(row["close"])
        if not math.isfinite(price) or price <= 0:
            continue
        cal = _calibration_at(perm, date, history, spec, cal_cache)
        if cal is None:
            continue
        raw_candidates.append((perm, row, cal))
        raw_map[perm] = cal

    reg_map, _reg_diag = regularize_calibration_cross_section(
        raw_map,
        jump_prior_strength=float(spec.get("model", {}).get("jump_prior_strength", 8.0)),
        max_mu_reliability_weight=float(spec.get("model", {}).get("drift_max_eb_weight", 0.35)),
    )
    eligible_g = g[g.get("eligible", pd.Series(True, index=g.index)).astype(bool)].copy()
    # Match the optimizer's return-consistent point-in-time technical inputs.
    if len(eligible_g):
        repaired = [history.repaired_signal_features(int(perm), date) for perm in eligible_g.index]
        for name in ("close","ret5","rsi2","distance_20d_high","sma20","sma50","sma100","sma200"):
            eligible_g[name] = [r[name] if r is not None else np.nan for r in repaired]
        eligible_g = eligible_g.dropna(subset=["close","sma200","ret5","rsi2"])
    alpha = _signal_alpha_frame(eligible_g)
    alpha_model = str(s.get("alpha_model", "BALANCED"))
    alpha_col, alpha_pct_col = _alpha_columns_for_model(alpha_model)
    close_all = pd.to_numeric(g.get("close"), errors="coerce")
    sma200_all = pd.to_numeric(g.get("sma200"), errors="coerce")
    valid_breadth = close_all.notna() & sma200_all.notna()
    breadth = float((close_all[valid_breadth] > sma200_all[valid_breadth]).mean()) if bool(valid_breadth.any()) else 0.0
    if breadth < float(s.get("min_market_breadth_200", 0.0)):
        events.append({"date": date, "event": "REGIME_ABSTAIN", "breadth_above_sma200": breadth, "threshold": float(s.get("min_market_breadth_200", 0.0))})
        return []

    prefiltered: list[tuple[float, int, str, pd.Series, Calibration, dict]] = []
    for perm, row, raw_cal in raw_candidates:
        cal = reg_map.get(perm, raw_cal)
        cal_cache[(int(perm), pd.Timestamp(date))] = cal
        a = alpha.loc[perm].to_dict() if perm in alpha.index else {}
        ps = float(a.get("alpha_prefilter_score", 0.5))
        prefiltered.append((ps, perm, str(row["ticker"]), row, cal, a))
    prefiltered.sort(key=lambda x: (x[0], x[1]), reverse=True)
    prefiltered = prefiltered[: int(prefilter_n)]

    accepted: list[dict] = []
    for pre_score, perm, ticker, row, cal, a in prefiltered:
        f = _terminal_forecast(
            perm,
            ticker,
            date,
            float(row["close"]),
            history,
            spec,
            cal_cache,
            forecast_cache,
            horizon_days=int(s["forecast_horizon_days"]),
            n_paths=int(n_paths),
            purpose=purpose,
            cal_override=cal,
        )
        if f is None:
            continue
        allocation_score, risk_vol = _risk_allocation_score(f, spec)
        alpha_score = float(a.get(alpha_col, 0.5))
        alpha_pct = float(a.get(alpha_pct_col, 0.5))
        stop_ucb = float(f.get("mc_p_stop", 1.0)) + float(s.get("mc_stop_risk_z", 1.0)) * float(f.get("mc_p_stop_se", 0.0))
        tests = {
            "alpha_percentile": alpha_pct >= float(s.get("min_alpha_percentile", 0.0)),
            "stop_probability_ucb": stop_ucb <= float(s.get("max_mc_stop_probability", 1.0)),
            "risk_volatility": risk_vol <= float(s.get("max_risk_volatility", float("inf"))),
            "monte_carlo_converged": (not bool(s.get("require_mc_converged", True))) or bool(f.get("mc_converged", False)),
            "maximum_mc_mean_se": s.get("max_mc_mean_se") is None or float(f.get("mc_mean_se", float("inf"))) <= float(s["max_mc_mean_se"]),
        }
        failed = [k for k, ok in tests.items() if not ok]
        passed, reason = (not failed), ";".join(failed)
        model_score = alpha_score
        forecast_index = len(forecast_rows)
        forecast_rows.append({
            "date": date,
            "ticker": ticker,
            "permno": perm,
            "sector": str(row.get("sector", "Other")),
            "forecast_type": forecast_type,
            "signal_source": "MONTH_END" if forecast_type == "MONTH_END_ENTRY" else "WEEKLY_REFILL",
            "horizon_days": int(s["forecast_horizon_days"]),
            "prefilter_score": pre_score,
            "alpha_model": alpha_model,
            "alpha_score": alpha_score,
            "alpha_percentile": alpha_pct,
            "breadth_above_sma200": breadth,
            "mc_stop_probability_ucb": stop_ucb,
            **a,
            **f,
            "pass": passed,
            "reason": reason,
            "model_score": model_score,
            "ranking_score": model_score,
            "target_weight": np.nan,
            "risk_volatility": risk_vol,
            "risk_allocation_score": allocation_score,
        })
        events.append({
            "date": date,
            "ticker": ticker,
            "permno": perm,
            "event": "MODEL_FORECAST_PASS" if passed else "MODEL_FORECAST_REJECT",
            "signal_source": "MONTH_END" if forecast_type == "MONTH_END_ENTRY" else "WEEKLY_REFILL",
            "model_score": model_score,
            "risk_allocation_score": allocation_score,
            "reason": reason,
        })
        if passed and (perm, date) not in blocked_entries:
            accepted.append({
                "permno": perm,
                "ticker": ticker,
                "sector": str(row.get("sector", "Other")),
                "signal_date": date,
                "signal_source": "MONTH_END" if forecast_type == "MONTH_END_ENTRY" else "WEEKLY_REFILL",
                "model_score": model_score,
                "ranking_score": model_score,
                "risk_allocation_score": allocation_score,
                "risk_volatility": risk_vol,
                "forecast_index": forecast_index,
                "terminal_volatility": float(f["terminal_volatility"]),
                "expected_return": float(f["expected_return"]),
            })
    accepted.sort(key=lambda x: (x["model_score"], x["permno"]), reverse=True)
    return accepted


def _simulate_once(
    daily: pd.DataFrame,
    cfg: dict,
    spec: dict,
    history: HistoryStore,
    cal_cache: dict,
    forecast_cache: dict,
    blocked_entries: set[tuple[int, pd.Timestamp]],
    *,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp | None,
):
    s = spec["standalone"]
    cost = float(cfg["execution"]["transaction_cost_bps_per_side"]) / 10000.0
    initial_capital = float(s["initial_capital"])
    cash = initial_capital
    positions: dict[int, ModelPosition] = {}
    pending: list[dict] = []
    trades: list[dict] = []
    events: list[dict] = []
    equity: list[dict] = []
    forecast_rows: list[dict] = []

    work = daily[daily["date"] >= pd.Timestamp(score_start)].copy()
    if score_end is not None:
        work = work[work["date"] <= pd.Timestamp(score_end)].copy()
    work = work.sort_values(["date", "permno"])
    all_dates = pd.DatetimeIndex(sorted(work["date"].unique()))
    month_end_dates, week_end_dates = _rebalance_date_sets(all_dates)

    # Patch 2.9.3 right-boundary policy: do not create a new entry signal unless
    # the full maximum holding horizon can fit inside the loaded global trading
    # calendar. With next-close execution and entry day counted as held day 1,
    # a 20-session maximum hold requires at least 20 global sessions after the
    # signal date. This is end-of-sample hygiene only; no future price values are
    # inspected and no terminal position is force-closed.
    max_hold = int(s["max_holding_days"])
    last_full_horizon_signal_date = _last_full_horizon_signal_date(all_dates, max_hold)

    for date, gg in work.groupby("date", sort=True):
        date = pd.Timestamp(date)
        gg = gg.copy()
        if history.corporate_action_events is not None:
            # The standalone and optimizer exact engines now use the SAME
            # return-consistent mark-to-market/stop price basis.
            gg["close"] = [history.economic_price(int(p), date) for p in gg["permno"]]
            gg = gg.dropna(subset=["close"])
        g = gg.set_index("permno", drop=False)

        # Model-owned exits + explicit security-discontinuation handling.
        discontinuation_cfg = spec.get("security_discontinuation", {})
        missing_before_exit = max(1, int(discontinuation_cfg.get("missing_global_sessions_before_exit", 2)))
        for perm, pos in list(positions.items()):
            reason = None
            risk_reason = ""
            exit_price_source = "CURRENT_CLOSE"
            discontinuation_last_seen_date = None
            missing_sessions_at_exit = 0

            if perm not in g.index:
                # Causal missing-quote detection: we do not inspect the security's future
                # end date and we do not retroactively reject its entry. A single missing
                # row is tolerated; persistent absence is treated as a security lifecycle
                # discontinuation and closed using the last observed close as an explicit
                # accounting proxy.
                pos.missing_global_sessions += 1
                events.append({
                    "date": date,
                    "ticker": pos.ticker,
                    "permno": perm,
                    "event": "SECURITY_QUOTE_MISSING",
                    "missing_global_sessions": pos.missing_global_sessions,
                    "last_seen_date": pos.last_seen_date,
                    "last_seen_price": pos.last_seen_price,
                })
                if pos.missing_global_sessions < missing_before_exit:
                    continue
                if pos.last_seen_price is None or not math.isfinite(float(pos.last_seen_price)) or float(pos.last_seen_price) <= 0:
                    raise RuntimeError(
                        f"Cannot resolve security discontinuation for PERMNO {perm} ({pos.ticker}): "
                        f"last_seen_price={pos.last_seen_price!r}"
                    )
                price = float(pos.last_seen_price)
                reason = "SECURITY_DISCONTINUATION"
                risk_reason = "PERSISTENT_MISSING_QUOTE_STREAM"
                exit_price_source = str(discontinuation_cfg.get("exit_price_source", "LAST_OBSERVED_CLOSE_PROXY"))
                discontinuation_last_seen_date = pos.last_seen_date
                missing_sessions_at_exit = pos.missing_global_sessions
            else:
                row = g.loc[perm]
                price = float(row["close"])
                if not math.isfinite(price) or price <= 0:
                    # A present row with an unusable close is treated exactly like a missing
                    # quote for lifecycle accounting, but it does not overwrite last good mark.
                    pos.missing_global_sessions += 1
                    events.append({
                        "date": date,
                        "ticker": pos.ticker,
                        "permno": perm,
                        "event": "SECURITY_CLOSE_INVALID",
                        "missing_global_sessions": pos.missing_global_sessions,
                        "last_seen_date": pos.last_seen_date,
                        "last_seen_price": pos.last_seen_price,
                    })
                    if pos.missing_global_sessions < missing_before_exit:
                        continue
                    if pos.last_seen_price is None or not math.isfinite(float(pos.last_seen_price)) or float(pos.last_seen_price) <= 0:
                        raise RuntimeError(
                            f"Cannot resolve invalid-close discontinuation for PERMNO {perm} ({pos.ticker})"
                        )
                    price = float(pos.last_seen_price)
                    reason = "SECURITY_DISCONTINUATION"
                    risk_reason = "PERSISTENT_INVALID_OR_MISSING_QUOTE_STREAM"
                    exit_price_source = str(discontinuation_cfg.get("exit_price_source", "LAST_OBSERVED_CLOSE_PROXY"))
                    discontinuation_last_seen_date = pos.last_seen_date
                    missing_sessions_at_exit = pos.missing_global_sessions
                else:
                    pos.last_seen_date = date
                    pos.last_seen_price = price
                    pos.missing_global_sessions = 0
                    pos.held_days += 1
                    if price <= pos.entry_price * (1.0 - float(s["emergency_stop_pct"])):
                        reason = "EMERGENCY_STOP"
                    elif pos.held_days >= int(s["max_holding_days"]):
                        reason = "MODEL_HORIZON"
                    elif pos.held_days % int(s["recheck_every_days"]) == 0:
                        remaining = max(1, int(s["max_holding_days"]) - pos.held_days)
                        f = _terminal_forecast(
                            perm,
                            pos.ticker,
                            date,
                            price,
                            history,
                            spec,
                            cal_cache,
                            forecast_cache,
                            horizon_days=remaining,
                            n_paths=int(s["recheck_n_paths"]),
                            purpose="standalone_recheck_v9",
                        )
                        if f is not None:
                            exit_now, risk_reason, new_negative_streak = _risk_exit(
                                f, spec, pos.negative_drift_rechecks
                            )
                            pos.negative_drift_rechecks = new_negative_streak
                            forecast_rows.append({
                                "date": date,
                                "ticker": pos.ticker,
                                "permno": perm,
                                "forecast_type": "POSITION_RECHECK",
                                "signal_source": "POSITION_RECHECK",
                                "horizon_days": remaining,
                                "prefilter_score": np.nan,
                                **f,
                                "pass": not exit_now,
                                "reason": risk_reason,
                                "model_score": np.nan,
                                "target_weight": np.nan,
                                "risk_volatility": np.nan,
                                "negative_drift_rechecks": pos.negative_drift_rechecks,
                            })
                            if exit_now:
                                reason = "MODEL_RISK_EXIT"

            if reason:
                proceeds = pos.shares * price
                fee = proceeds * cost
                cash += proceeds - fee
                invested = pos.entry_notional + pos.entry_cost
                net_pnl = proceeds - fee - invested
                net = net_pnl / invested if invested > 0 else 0.0
                zero_recovery_incremental_loss = -(proceeds - fee) if reason == "SECURITY_DISCONTINUATION" else 0.0
                trades.append({
                    "ticker": pos.ticker,
                    "permno": perm,
                    "sector": pos.sector,
                    "signal_date": pos.signal_date,
                    "signal_source": pos.signal_source,
                    "entry_date": pos.entry_date,
                    "exit_date": date,
                    "entry_price": pos.entry_price,
                    "exit_price": price,
                    "price_basis": "CRSP_RET_REINVESTED_WEALTH_INDEX",
                    "exit_price_source": exit_price_source,
                    "last_observed_quote_date": discontinuation_last_seen_date,
                    "missing_global_sessions_at_exit": missing_sessions_at_exit,
                    "held_days": pos.held_days,
                    "entry_notional_usd": pos.entry_notional,
                    "entry_weight": pos.entry_weight,
                    "entry_cost_usd": pos.entry_cost,
                    "exit_cost_usd": fee,
                    "gross_return": price / pos.entry_price - 1.0,
                    "net_return": net,
                    "net_pnl_usd": net_pnl,
                    "zero_recovery_incremental_loss_usd": zero_recovery_incremental_loss,
                    "entry_model_score": pos.model_score,
                    "entry_forecast_terminal_volatility": pos.forecast_terminal_volatility,
                    "entry_forecast_expected_return": pos.forecast_expected_return,
                    "exit_reason": reason,
                    "model_risk_reason": risk_reason,
                    "negative_drift_rechecks_at_exit": pos.negative_drift_rechecks,
                })
                events.append({
                    "date": date,
                    "ticker": pos.ticker,
                    "permno": perm,
                    "event": "EXIT",
                    "price": price,
                    "price_source": exit_price_source,
                    "reason": reason,
                    "model_risk_reason": risk_reason,
                    "last_observed_quote_date": discontinuation_last_seen_date,
                    "missing_global_sessions_at_exit": missing_sessions_at_exit,
                    "net_pnl_usd": net_pnl,
                })
                del positions[perm]

        # Execute prior-close orders AFTER same-close exits have released cash,
        # gross, sector and position capacity. Orders are valid for this close only:
        # every order receives either ENTRY or ENTRY_ORDER_CANCELLED and is never
        # silently carried to a later day (which would violate next-close semantics).
        for order in pending:
            perm = int(order["permno"])
            key = (perm, pd.Timestamp(order["signal_date"]))
            order_id = str(order.get("order_id", f"{pd.Timestamp(order['signal_date']).strftime('%Y%m%d')}-{perm}"))
            cancel_reason = None
            row = None
            price = None

            if key in blocked_entries:
                cancel_reason = "BOUNDARY_BLOCKED"
            elif perm in positions:
                cancel_reason = "ALREADY_HELD"
            elif len(positions) >= int(s["max_active_positions"]):
                cancel_reason = "MAX_POSITIONS_REACHED"
            elif perm not in g.index:
                cancel_reason = "NO_NEXT_CLOSE_QUOTE"
            else:
                row = g.loc[perm]
                price = float(row["close"])
                if not math.isfinite(price) or price <= 0:
                    cancel_reason = "INVALID_NEXT_CLOSE"

            if cancel_reason is None:
                open_value = _mark_to_market(positions, g)
                eq_now = cash + open_value
                if eq_now <= 0:
                    cancel_reason = "NONPOSITIVE_EQUITY"
                else:
                    sector = str(row.get("sector", order.get("sector", "Other")))
                    current_sector = sum(
                        pos.shares * (
                            float(g.loc[k]["close"])
                            if k in g.index and math.isfinite(float(g.loc[k]["close"]))
                            else float(pos.last_seen_price if pos.last_seen_price is not None else pos.entry_price)
                        )
                        for k, pos in positions.items()
                        if pos.sector == sector
                    )
                    gross_room = max(0.0, eq_now * float(s["max_gross_exposure"]) - open_value)
                    sector_room = max(0.0, eq_now * float(s["max_sector_weight"]) - current_sector)
                    target_weight = float(order["target_weight"])
                    target = eq_now * target_weight
                    cash_room = max(0.0, cash / (1.0 + cost))
                    notional = min(target, gross_room, sector_room, cash_room)
                    actual_weight = notional / eq_now if eq_now > 0 else 0.0
                    if notional <= 0.0:
                        if gross_room <= 1e-12:
                            cancel_reason = "NO_GROSS_ROOM"
                        elif sector_room <= 1e-12:
                            cancel_reason = "NO_SECTOR_ROOM"
                        elif cash_room <= 1e-12:
                            cancel_reason = "NO_CASH_ROOM"
                        else:
                            cancel_reason = "ZERO_NOTIONAL"
                    elif actual_weight < float(s["min_position_weight"]):
                        cancel_reason = "BELOW_MIN_POSITION_WEIGHT"

            if cancel_reason is not None:
                events.append({
                    "date": date,
                    "ticker": order["ticker"],
                    "permno": perm,
                    "event": "ENTRY_ORDER_CANCELLED",
                    "order_id": order_id,
                    "signal_date": order["signal_date"],
                    "signal_source": order.get("signal_source"),
                    "target_weight": float(order.get("target_weight", np.nan)),
                    "reason": cancel_reason,
                })
                continue

            sector = str(row.get("sector", order.get("sector", "Other")))
            shares = notional / price
            fee = notional * cost
            cash -= notional + fee
            positions[perm] = ModelPosition(
                permno=perm,
                ticker=str(row["ticker"]),
                sector=sector,
                signal_date=pd.Timestamp(order["signal_date"]),
                entry_date=date,
                entry_price=price,
                shares=shares,
                entry_notional=notional,
                entry_weight=actual_weight,
                entry_cost=fee,
                model_score=float(order["model_score"]),
                forecast_terminal_volatility=float(order["terminal_volatility"]),
                forecast_expected_return=float(order["expected_return"]),
                signal_source=str(order.get("signal_source", "UNKNOWN")),
                # Entry close is held session 1, matching the 20-session forecast
                # horizon and forward-diagnostic convention.
                held_days=1,
                last_seen_date=date,
                last_seen_price=price,
                missing_global_sessions=0,
            )
            events.append({
                "date": date,
                "ticker": str(row["ticker"]),
                "permno": perm,
                "event": "ENTRY",
                "order_id": order_id,
                "signal_date": order["signal_date"],
                "signal_source": order.get("signal_source"),
                "price": price,
                "model_score": order["model_score"],
                "target_weight": target_weight,
                "actual_weight": actual_weight,
                "forecast_terminal_volatility": order["terminal_volatility"],
            })
        pending = []

        # Entry discovery. Month-end does a full rebalance scan. On non-month-end
        # week-ends, a fresh model scan is allowed only to refill vacancies.
        discovery = None
        entry_schedule = str(s.get("entry_schedule", "BOTH")).upper()
        use_month = entry_schedule in {"MONTH_END_ONLY", "BOTH"}
        use_week = entry_schedule in {"WEEKLY_ONLY", "BOTH"}
        if use_month and date in month_end_dates:
            discovery = {
                "forecast_type": "MONTH_END_ENTRY",
                "prefilter_n": int(s["prefilter_n"]),
                "n_paths": int(s["entry_n_paths"]),
                "purpose": "standalone_month_end_v9",
                "max_new_entries": int(s["max_new_entries_per_rebalance"]),
            }
        elif use_week and bool(s.get("weekly_vacancy_refill_enabled", True)) and date in week_end_dates:
            open_value = _mark_to_market(positions, g)
            eq_now = cash + open_value
            current_gross = open_value / eq_now if eq_now > 0 else 0.0
            vacancies = max(0, int(s["max_active_positions"]) - len(positions))
            enough_room = float(s["target_gross_exposure"]) - current_gross >= float(s["min_position_weight"]) - 1e-12
            if vacancies > 0 and enough_room:
                discovery = {
                    "forecast_type": "WEEKLY_REFILL_ENTRY",
                    "prefilter_n": int(s["weekly_refill_prefilter_n"]),
                    "n_paths": int(s["weekly_refill_n_paths"]),
                    "purpose": "standalone_weekly_refill_v9",
                    "max_new_entries": min(int(s["weekly_refill_max_new_entries"]), vacancies),
                }

        if discovery is not None:
            accepted = _discover_entry_candidates(
                date=date,
                g=g,
                positions=positions,
                history=history,
                spec=spec,
                cal_cache=cal_cache,
                forecast_cache=forecast_cache,
                blocked_entries=blocked_entries,
                forecast_rows=forecast_rows,
                events=events,
                forecast_type=str(discovery["forecast_type"]),
                prefilter_n=int(discovery["prefilter_n"]),
                n_paths=int(discovery["n_paths"]),
                purpose=str(discovery["purpose"]),
            )
            open_value = _mark_to_market(positions, g)
            eq_now = cash + open_value
            current_gross = open_value / eq_now if eq_now > 0 else 0.0
            vacancies = max(0, int(s["max_active_positions"]) - len(positions))
            max_batch = min(int(discovery["max_new_entries"]), vacancies)
            current_sector_weights = _sector_weights(positions, g, eq_now)
            allocated = _construct_signal_batch(
                accepted,
                current_gross=current_gross,
                current_sector_weights=current_sector_weights,
                vacancies=vacancies,
                spec=spec,
                max_new_entries=max_batch,
                date=date,
                events=events,
            )

            # Deterministic dataset-end no-entry buffer. We deliberately run the
            # normal discovery/ranking first so diagnostics still show what the
            # model wanted to own, but signals that cannot fit a complete natural
            # holding horizon are never placed into ``pending``. This replaces the
            # non-convergent reject-and-rerun loop introduced by weekly refills.
            boundary_block = (
                last_full_horizon_signal_date is None
                or date > last_full_horizon_signal_date
            )
            for order in allocated:
                forecast_rows[int(order["forecast_index"])]["target_weight"] = float(order["target_weight"])
                if boundary_block:
                    events.append({
                        "date": date,
                        "ticker": order["ticker"],
                        "permno": order["permno"],
                        "event": "BOUNDARY_SIGNAL_REJECTED",
                        "signal_source": order.get("signal_source"),
                        "model_score": order["model_score"],
                        "risk_allocation_score": order["risk_allocation_score"],
                        "target_weight": order["target_weight"],
                        "current_gross_before_signal": current_gross,
                        "last_full_horizon_signal_date": last_full_horizon_signal_date,
                        "boundary_policy": "FULL_HORIZON_PREENTRY_REJECTION",
                    })
                    continue
                order["order_id"] = f"{pd.Timestamp(date).strftime('%Y%m%d')}-{order.get('signal_source','UNKNOWN')}-{int(order['permno'])}"
                pending.append(order)
                events.append({
                    "date": date,
                    "ticker": order["ticker"],
                    "permno": order["permno"],
                    "event": "SIGNAL",
                    "order_id": order["order_id"],
                    "signal_source": order.get("signal_source"),
                    "model_score": order["model_score"],
                    "risk_allocation_score": order["risk_allocation_score"],
                    "target_weight": order["target_weight"],
                    "current_gross_before_signal": current_gross,
                })

        open_value = _mark_to_market(positions, g)
        total = cash + open_value
        equity.append({
            "date": date,
            "equity": total,
            "cash": cash,
            "gross_exposure": open_value / total if total > 0 else 0.0,
            "positions": len(positions),
        })

    unresolved = [
        {"permno": p.permno, "ticker": p.ticker, "signal_date": p.signal_date, "entry_date": p.entry_date}
        for p in positions.values()
    ]
    return pd.DataFrame(equity), pd.DataFrame(trades), pd.DataFrame(events), pd.DataFrame(forecast_rows), unresolved

def _attach_forward_diagnostics(entry_forecasts: pd.DataFrame, history: HistoryStore, horizon: int) -> pd.DataFrame:
    if entry_forecasts.empty:
        return entry_forecasts.copy()
    rows = []
    for rec in entry_forecasts.to_dict("records"):
        d = history.realized_forward_path(int(rec["permno"]), pd.Timestamp(rec["date"]), int(horizon))
        row = dict(rec)
        if d is None:
            row.update({
                "realized_entry_date": pd.NaT,
                "realized_terminal_date": pd.NaT,
                "realized_forward_return": np.nan,
                "realized_mfe": np.nan,
                "realized_mae": np.nan,
            })
        else:
            row.update(d)
        rows.append(row)
    return pd.DataFrame(rows)




def _window_for_mode(spec: dict, mode: str) -> dict:
    mode = str(mode).lower()
    if mode not in {"development", "holdout"}:
        raise ValueError(f"Unsupported research mode: {mode}")
    return dict(spec["experiment_windows"][mode])


def _checkpoint_dir(mode: str) -> Path:
    return ROOT / "outputs" / "research" / f".stochastic_standalone_patch_2_9_3_{mode}_checkpoint"


def _write_simulation_checkpoint(
    *,
    mode: str,
    window: dict,
    run_id: str,
    run_started: dict,
    identity: dict,
    eq: pd.DataFrame,
    trades: pd.DataFrame,
    events: pd.DataFrame,
    forecasts: pd.DataFrame,
    forward: pd.DataFrame,
    cal_cache_entries: int,
    forecast_cache_entries: int,
    boundary_entries_rejected: int,
    rows_loaded: int,
    min_loaded_date: str | None,
    max_loaded_date: str | None,
    warmup_global_sessions: int,
    required_warmup_sessions: int,
) -> Path:
    """Persist the expensive simulation before reporting diagnostics."""
    cp = _checkpoint_dir(mode)
    if cp.exists():
        shutil.rmtree(cp)
    cp.mkdir(parents=True, exist_ok=True)
    eq.to_csv(cp / "equity_curve.csv", index=False)
    trades.to_csv(cp / "trade_log.csv", index=False)
    events.to_csv(cp / "event_log.csv", index=False)
    forecasts.to_csv(cp / "model_forecasts.csv", index=False)
    forward.to_csv(cp / "forecast_forward_outcomes.csv", index=False)
    write_json(cp / "CHECKPOINT.json", {
        "patch": "2.9.3",
        "research_mode": mode,
        "run_id": run_id,
        "status": "SIMULATION_COMPLETE_DIAGNOSTICS_PENDING",
        "run_started": run_started,
        "evaluation_window": window,
        "production_identity": identity,
        "calibration_cache_entries": int(cal_cache_entries),
        "forecast_cache_entries": int(forecast_cache_entries),
        "boundary_entries_rejected": int(boundary_entries_rejected),
        "rows_loaded": int(rows_loaded),
        "min_loaded_date": min_loaded_date,
        "max_loaded_date": max_loaded_date,
        "warmup_global_sessions": int(warmup_global_sessions),
        "required_warmup_sessions": int(required_warmup_sessions),
    })
    return cp


def _publish_output_bundle(
    out: Path,
    frames: dict[str, pd.DataFrame],
    payloads: dict[str, dict],
    *,
    patch: str,
    run_id: str,
) -> None:
    """Stage a complete bundle and promote RUN_COMPLETE.json last."""
    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    tag = str(patch).replace(".", "_")
    staging = parent / f".{out.name}_staging_{tag}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    try:
        for name, frame in frames.items():
            frame.to_csv(staging / name, index=False)
        for name, payload in payloads.items():
            write_json(staging / name, payload)
        write_json(staging / "RUN_COMPLETE.json", {"patch": patch, "run_id": run_id, "status": "PASS"})

        summary_check = json.loads((staging / "summary.json").read_text(encoding="utf-8"))
        manifest_check = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        if summary_check.get("patch") != patch or manifest_check.get("spec_patch") != patch:
            raise RuntimeError("Staged stochastic outputs failed patch-version consistency check")
        if summary_check.get("run_id") != run_id or manifest_check.get("run_id") != run_id:
            raise RuntimeError("Staged stochastic outputs failed run-id consistency check")

        complete_path = out / "RUN_COMPLETE.json"
        if complete_path.exists():
            complete_path.unlink()

        names = [p.name for p in staging.iterdir() if p.is_file() and p.name != "RUN_COMPLETE.json"]
        priority = {"summary.json": 90, "manifest.json": 91, "RUN_STARTED.json": 92}
        names.sort(key=lambda n: (priority.get(n, 10), n))
        for name in names:
            os.replace(staging / name, out / name)
        os.replace(staging / "RUN_COMPLETE.json", out / "RUN_COMPLETE.json")

        for name in ("summary.json", "manifest.json", "RUN_STARTED.json", "RUN_COMPLETE.json"):
            if not (out / name).exists():
                raise RuntimeError(f"Published stochastic output bundle missing {name}")
        final_summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        final_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        final_started = json.loads((out / "RUN_STARTED.json").read_text(encoding="utf-8"))
        final_complete = json.loads((out / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        ids = {
            final_summary.get("run_id"),
            final_manifest.get("run_id"),
            final_started.get("run_id"),
            final_complete.get("run_id"),
        }
        if final_summary.get("patch") != patch or final_manifest.get("spec_patch") != patch:
            raise RuntimeError("Published stochastic output bundle has stale version metadata")
        if final_started.get("patch") != patch or final_complete.get("patch") != patch:
            raise RuntimeError("Published run markers have stale version metadata")
        if None in ids or len(ids) != 1:
            raise RuntimeError(f"Published stochastic output bundle run-id mismatch: {ids}")
    finally:
        if (out / "RUN_COMPLETE.json").exists():
            try:
                final_complete = json.loads((out / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
            except Exception:
                final_complete = {}
            if final_complete.get("patch") == patch and final_complete.get("run_id") == run_id:
                shutil.rmtree(staging, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Heston-Merton research runner")
    parser.add_argument(
        "--mode",
        choices=("development", "holdout"),
        default="development",
        help="development=2015-2024 only; holdout=2025-present and requires frozen config",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mode = str(args.mode)
    if mode == "holdout":
        raise RuntimeError("Patch 2.9.3 is an optimizer-development patch. 2025+ holdout execution is intentionally disabled until a later top-3 promotion/freeze patch.")
    identity = verify_frozen_production_identity(ROOT)
    cfg = load_config(ROOT)
    spec = load_spec(ROOT)
    if str(spec.get("patch")) != "2.9.3":
        raise RuntimeError(f"Expected Patch 2.9.3 spec, found {spec.get('patch')}")
    s = spec["standalone"]
    window = _window_for_mode(spec, mode)

    freeze = None
    if mode == "holdout":
        freeze = verify_holdout_freeze(ROOT, spec)

    print("=" * 118)
    print(f"PATCH 2.9.3 - STANDALONE HESTON-MERTON V8 - {window['label']}")
    if mode == "development":
        print("DEVELOPMENT ONLY: rows after 2024-12-31 are not loaded into this process.")
    else:
        print("HOLDOUT EVALUATION: exact frozen development code/spec fingerprint verified.")
    print("=" * 118)

    out = ROOT / str(window["output_dir"])
    if out.exists():
        shutil.rmtree(out)
    cp = _checkpoint_dir(mode)
    if cp.exists():
        shutil.rmtree(cp)

    run_id = f"{spec['patch']}-{mode}-{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    run_started = {
        "patch": str(spec["patch"]),
        "research_mode": mode,
        "run_id": run_id,
        "status": "RUNNING",
        "runner": str(Path(__file__).resolve()),
        "spec": str((HERE / "STOCHASTIC_RESEARCH_SPEC.json").resolve()),
        "output_dir": str(out.resolve()),
        "evaluation_window": window,
        "development_freeze": freeze,
        "started_utc": pd.Timestamp.utcnow().isoformat(),
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "RUN_STARTED.json", run_started)
    print(f"[LIVE PATCH] {spec['patch']} runner: {Path(__file__).resolve()}")
    print(f"[MODE] {mode}")
    print(f"[SCORE WINDOW] {window['score_start']} through {window.get('score_end') or 'latest available'}")
    print(f"[OUTPUT] {out.resolve()}")

    history_start = window["history_start"]
    history_end = window.get("score_end") if mode == "development" else None
    if mode == "development":
        print(f"[DATA FIREWALL] Loading only {history_start} through {history_end}. 2025+ cannot enter memory.")
    else:
        print(f"[HISTORY] Loading {history_start} through latest available for trailing calibration + holdout scoring.")
    daily = load_feature_history(cfg, history_start, history_end)
    if daily.empty:
        raise RuntimeError("No feature rows loaded for requested research window.")
    min_loaded = pd.Timestamp(daily["date"].min())
    max_loaded = pd.Timestamp(daily["date"].max())
    if mode == "development" and max_loaded > pd.Timestamp("2024-12-31"):
        raise RuntimeError(f"DEVELOPMENT FIREWALL BREACH: max loaded date is {max_loaded}")
    print(
        f"Loaded {len(daily):,} feature rows across {daily['permno'].nunique():,} PERMNOs "
        f"({min_loaded.date()} -> {max_loaded.date()})."
    )

    score_start = pd.Timestamp(window["score_start"])
    score_end = pd.Timestamp(window["score_end"]) if window.get("score_end") else None
    required_warmup_sessions = int(spec["model"]["max_calibration_observations"])
    feature_warmup_sessions = int(daily.loc[daily["date"] < score_start, "date"].nunique())

    # The canonical feature store intentionally begins in 2014. For a true 2015
    # development start, supplement ONLY the calibration HistoryStore from the
    # older normalized CRSP parquet store. These rows never enter signal discovery,
    # eligibility, portfolio construction, exits, or forward diagnostics.
    missing_warmup_sessions = max(0, required_warmup_sessions - feature_warmup_sessions)
    normalized_backfill = pd.DataFrame(columns=["date", "permno", "ret1", "close"])
    if missing_warmup_sessions > 0:
        normalized_backfill = load_normalized_calibration_backfill(
            cfg,
            before_date=min_loaded,
            global_sessions_needed=missing_warmup_sessions,
        )
        print(
            f"[CALIBRATION BACKFILL] Feature store supplies {feature_warmup_sessions} pre-score sessions; "
            f"loaded {normalized_backfill['date'].nunique()} older normalized-CRSP sessions "
            f"({normalized_backfill['date'].min().date()} -> {normalized_backfill['date'].max().date()}) "
            "for Heston-Merton calibration only."
        )

    normalized_backfill_sessions = int(normalized_backfill["date"].nunique()) if len(normalized_backfill) else 0
    calibration_frame = merge_calibration_history(
        daily[["date", "permno", "ret1", "close"]],
        normalized_backfill,
    )
    history = HistoryStore.from_daily(calibration_frame)
    warmup_global_sessions = int(calibration_frame.loc[calibration_frame["date"] < score_start, "date"].nunique())
    calibration_min_loaded = pd.Timestamp(calibration_frame["date"].min())
    if mode == "development" and warmup_global_sessions < required_warmup_sessions:
        raise RuntimeError(
            f"Insufficient combined calibration warm-up: {warmup_global_sessions} global sessions; "
            f"need at least {required_warmup_sessions}. Feature store contributed {feature_warmup_sessions}; "
            f"normalized CRSP backfill contributed {normalized_backfill['date'].nunique() if len(normalized_backfill) else 0}."
        )
    print(
        f"[WARM-UP] Combined calibration history has {warmup_global_sessions} global sessions before "
        f"{score_start.date()} (required >= {required_warmup_sessions}); earliest calibration row "
        f"{calibration_min_loaded.date()}."
    )

    cal_cache: dict = {}
    forecast_cache: dict = {}
    # Patch 2.9.3 uses a deterministic pre-entry boundary buffer, so the expensive
    # simulation runs exactly once. ``blocked`` remains only for API compatibility
    # with the discovery helper and is intentionally empty.
    blocked: set[tuple[int, pd.Timestamp]] = set()
    result = _simulate_once(
        daily,
        cfg,
        spec,
        history,
        cal_cache,
        forecast_cache,
        blocked,
        score_start=score_start,
        score_end=score_end,
    )
    eq, trades, events, forecasts, unresolved = result
    if unresolved:
        detail = [
            {
                "permno": int(u["permno"]),
                "ticker": str(u["ticker"]),
                "signal_date": str(pd.Timestamp(u["signal_date"]).date()),
                "entry_date": str(pd.Timestamp(u["entry_date"]).date()),
            }
            for u in unresolved[:20]
        ]
        raise RuntimeError(
            "Unexpected unresolved position(s) despite full-horizon pre-entry boundary policy. "
            "This indicates a security-specific data discontinuity/delisting rather than the "
            f"dataset-end boundary and must not be silently removed: {detail}"
        )

    boundary_entries_rejected = 0
    if len(events) and "event" in events.columns:
        boundary_entries_rejected = int((events["event"] == "BOUNDARY_SIGNAL_REJECTED").sum())
    if boundary_entries_rejected:
        print(
            f"[RIGHT BOUNDARY] Rejected {boundary_entries_rejected} new signal(s) that could not "
            "fit the full natural holding horizon before the loaded dataset end. No rerun required."
        )

    if len(eq):
        if pd.Timestamp(eq["date"].min()) < score_start:
            raise RuntimeError("Equity curve starts before configured score_start.")
        if score_end is not None and pd.Timestamp(eq["date"].max()) > score_end:
            raise RuntimeError("Equity curve extends beyond configured score_end.")
    if len(trades) and score_end is not None and pd.Timestamp(trades["exit_date"].max()) > score_end:
        raise RuntimeError("Development trade log contains post-window exits.")

    # Mark entry forecasts that became actual signals.
    if len(forecasts):
        forecasts = forecasts.copy()
        signal_keys = {
            (int(r.permno), pd.Timestamp(r.date))
            for r in events.itertuples(index=False)
            if getattr(r, "event", None) == "SIGNAL"
        }
        entry_types = _entry_forecast_types()
        forecasts["selected_for_signal"] = [
            (int(p), pd.Timestamp(d)) in signal_keys if ft in entry_types else False
            for p, d, ft in zip(forecasts["permno"], forecasts["date"], forecasts["forecast_type"])
        ]

    entry_forecasts = (
        forecasts[forecasts["forecast_type"].isin(_entry_forecast_types())].copy()
        if len(forecasts)
        else pd.DataFrame()
    )
    forward = _attach_forward_diagnostics(entry_forecasts, history, int(s["forecast_horizon_days"]))
    if mode == "development" and len(forward) and "realized_terminal_date" in forward.columns:
        terminal = pd.to_datetime(forward["realized_terminal_date"], errors="coerce").dropna()
        if len(terminal) and terminal.max() > pd.Timestamp("2024-12-31"):
            raise RuntimeError("DEVELOPMENT FIREWALL BREACH: forward diagnostics used post-2024 outcomes.")

    checkpoint = _write_simulation_checkpoint(
        mode=mode,
        window=window,
        run_id=run_id,
        run_started=run_started,
        identity=identity,
        eq=eq,
        trades=trades,
        events=events,
        forecasts=forecasts,
        forward=forward,
        cal_cache_entries=len(cal_cache),
        forecast_cache_entries=len(forecast_cache),
        boundary_entries_rejected=boundary_entries_rejected,
        rows_loaded=len(daily),
        min_loaded_date=min_loaded.strftime("%Y-%m-%d"),
        max_loaded_date=max_loaded.strftime("%Y-%m-%d"),
        warmup_global_sessions=warmup_global_sessions,
        required_warmup_sessions=required_warmup_sessions,
    )
    print(f"[CHECKPOINT] Expensive simulation saved before diagnostics: {checkpoint.resolve()}")

    diag_cfg = spec.get("diagnostics", {})
    metrics = list(diag_cfg.get("forecast_metrics", []))
    qdiag = realized_quantile_diagnostics(
        forward,
        metrics,
        realized_col="realized_forward_return",
        buckets=int(diag_cfg.get("quantile_buckets", 5)),
    )
    ic_detail, ic_summary = cross_sectional_ic(
        forward,
        metrics,
        date_col="date",
        realized_col="realized_forward_return",
        min_names=int(diag_cfg.get("ic_min_names", 20)),
    )
    ic_by_source, ic_summary_by_source = cross_sectional_ic_by_source(
        forward,
        metrics,
        source_col="signal_source",
        date_col="date",
        realized_col="realized_forward_return",
        min_names=int(diag_cfg.get("ic_min_names", 20)),
    )
    nonoverlap_forward = non_overlapping_forecast_frame(
        forward,
        source_col="signal_source",
        date_col="date",
        entry_col="realized_entry_date",
        terminal_col="realized_terminal_date",
    )
    ic_nonoverlap, ic_nonoverlap_summary = cross_sectional_ic_by_source(
        nonoverlap_forward,
        metrics,
        source_col="signal_source",
        date_col="date",
        realized_col="realized_forward_return",
        min_names=int(diag_cfg.get("ic_min_names", 20)),
    )
    selection_lift = selection_lift_diagnostics(forward)
    path_parts = []
    for realized_metric in ("realized_mae", "realized_mfe"):
        d = realized_quantile_diagnostics(
            forward,
            ["risk_volatility", "terminal_volatility", "expected_shortfall_5"],
            realized_col=realized_metric,
            buckets=int(diag_cfg.get("quantile_buckets", 5)),
        )
        if len(d):
            d.insert(0, "realized_metric", realized_metric)
            path_parts.append(d)
    path_risk_diag = pd.concat(path_parts, ignore_index=True) if path_parts else pd.DataFrame()

    pm = (
        portfolio_metrics(eq["equity"], int(cfg["execution"]["annualization_days"]))
        if len(eq)
        else portfolio_metrics(pd.Series(dtype=float))
    )
    tm = trade_metrics(trades)
    risk_diag = equity_risk_diagnostics(eq)
    concentration = trade_concentration_diagnostics(trades)

    # Patch 2.9.3: isolate every security-discontinuation accounting assumption.
    # The base backtest uses the last observed close only after persistent quote
    # disappearance is observed causally. A zero-recovery stress delta is published
    # so performance dependence on that proxy is transparent rather than hidden.
    if len(trades) and "exit_reason" in trades.columns:
        discontinuation_trades = trades[trades["exit_reason"] == "SECURITY_DISCONTINUATION"].copy()
    else:
        discontinuation_trades = pd.DataFrame()
    if len(discontinuation_trades):
        stress_delta = float(pd.to_numeric(
            discontinuation_trades.get("zero_recovery_incremental_loss_usd", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0).sum())
        observed_pnl = float(pd.to_numeric(discontinuation_trades["net_pnl_usd"], errors="coerce").fillna(0.0).sum())
        final_equity = float(eq["equity"].iloc[-1]) if len(eq) else float(s["initial_capital"])
        discontinuation_summary = {
            "exits": int(len(discontinuation_trades)),
            "observed_proxy_net_pnl_usd": observed_pnl,
            "zero_recovery_incremental_loss_usd": stress_delta,
            "zero_recovery_stress_ending_equity_usd": float(final_equity + stress_delta),
            "tickers": sorted({str(x) for x in discontinuation_trades["ticker"].dropna().tolist()}),
            "policy": spec.get("security_discontinuation", {}),
        }
    else:
        discontinuation_summary = {
            "exits": 0,
            "observed_proxy_net_pnl_usd": 0.0,
            "zero_recovery_incremental_loss_usd": 0.0,
            "zero_recovery_stress_ending_equity_usd": float(eq["equity"].iloc[-1]) if len(eq) else float(s["initial_capital"]),
            "tickers": [],
            "policy": spec.get("security_discontinuation", {}),
        }

    signal_source_counts = {}
    if len(forecasts) and "signal_source" in forecasts.columns:
        ef = forecasts[forecasts["forecast_type"].isin(_entry_forecast_types())]
        if len(ef):
            signal_source_counts = {
                str(k): int(v)
                for k, v in ef[ef.get("selected_for_signal", False).astype(bool)]
                .groupby("signal_source")
                .size()
                .to_dict()
                .items()
            }

    entry_order_audit = pd.DataFrame()
    entry_order_summary = {"submitted": 0, "filled": 0, "cancelled": 0, "unresolved": 0, "cancel_reasons": {}}
    if len(events) and "event" in events.columns:
        audit_events = events[events["event"].isin(["SIGNAL", "ENTRY", "ENTRY_ORDER_CANCELLED"])].copy()
        if len(audit_events) and "order_id" in audit_events.columns:
            audit_rows = []
            for order_id, og in audit_events.dropna(subset=["order_id"]).groupby("order_id", sort=True):
                sig = og[og["event"] == "SIGNAL"]
                filled = og[og["event"] == "ENTRY"]
                cancelled = og[og["event"] == "ENTRY_ORDER_CANCELLED"]
                if len(filled) > 1 or len(cancelled) > 1 or (len(filled) and len(cancelled)):
                    raise RuntimeError(f"Entry-order audit invariant failed for {order_id}")
                status = "FILLED" if len(filled) else ("CANCELLED" if len(cancelled) else "UNRESOLVED")
                base = sig.iloc[0] if len(sig) else og.iloc[0]
                cancel_reason = cancelled.iloc[0].get("reason") if len(cancelled) else None
                audit_rows.append({
                    "order_id": str(order_id),
                    "signal_date": base.get("date"),
                    "ticker": base.get("ticker"),
                    "permno": base.get("permno"),
                    "signal_source": base.get("signal_source"),
                    "target_weight": base.get("target_weight"),
                    "status": status,
                    "execution_date": (filled.iloc[0].get("date") if len(filled) else (cancelled.iloc[0].get("date") if len(cancelled) else pd.NaT)),
                    "actual_weight": filled.iloc[0].get("actual_weight") if len(filled) else np.nan,
                    "cancel_reason": cancel_reason,
                })
            entry_order_audit = pd.DataFrame(audit_rows)
            if len(entry_order_audit):
                entry_order_summary["submitted"] = int(len(entry_order_audit))
                entry_order_summary["filled"] = int((entry_order_audit["status"] == "FILLED").sum())
                entry_order_summary["cancelled"] = int((entry_order_audit["status"] == "CANCELLED").sum())
                entry_order_summary["unresolved"] = int((entry_order_audit["status"] == "UNRESOLVED").sum())
                entry_order_summary["cancel_reasons"] = {
                    str(k): int(v) for k, v in entry_order_audit.loc[entry_order_audit["status"] == "CANCELLED", "cancel_reason"].value_counts(dropna=False).to_dict().items()
                }
                if entry_order_summary["unresolved"]:
                    raise RuntimeError(f"Entry-order audit found unresolved next-close orders: {entry_order_summary}")

    summary = {
        "patch": spec["patch"],
        "research_mode": mode,
        "run_id": run_id,
        "system": "Momentum-Pullback Alpha + Heston-Merton QE-M/RQMC Risk Control v9",
        "role": "RESEARCH_ONLY_INDEPENDENT_STRATEGY",
        "evaluation_window": {
            "history_start": str(window["history_start"]),
            "score_start": str(window["score_start"]),
            "score_end": str(window.get("score_end")) if window.get("score_end") else None,
            "min_loaded_date": min_loaded.strftime("%Y-%m-%d"),
            "max_loaded_date": max_loaded.strftime("%Y-%m-%d"),
        },
        "production_identity": identity,
        "development_freeze": freeze,
        "warmup_diagnostics": {
            "feature_history_requested_start": str(window["history_start"]),
            "feature_min_loaded_date": min_loaded.strftime("%Y-%m-%d"),
            "calibration_min_loaded_date": calibration_min_loaded.strftime("%Y-%m-%d"),
            "score_start": str(window["score_start"]),
            "feature_global_sessions_before_score_start": int(feature_warmup_sessions),
            "normalized_crsp_backfill_global_sessions": int(normalized_backfill_sessions),
            "combined_global_sessions_before_score_start": int(warmup_global_sessions),
            "global_sessions_before_score_start": int(warmup_global_sessions),
            "required_max_calibration_observations": int(required_warmup_sessions),
            "calibration_backfill_role": "CALIBRATION_ONLY_NEVER_TRADABLE",
            "first_entry_date": (pd.Timestamp(trades["entry_date"].min()).isoformat() if len(trades) else None),
            "first_signal_date": (pd.Timestamp(events.loc[events["event"] == "SIGNAL", "date"].min()).isoformat() if len(events) and (events["event"] == "SIGNAL").any() else None),
        },
        "model_interpretation": "Historical physical-measure Heston-Merton proxy; no option-implied surface is used.",
        "portfolio_metrics": pm,
        "trade_metrics": tm,
        "risk_diagnostics": risk_diag,
        "trade_concentration": concentration,
        "security_discontinuation": discontinuation_summary,
        "forecast_summary": {
            "entry_forecasts": int(len(entry_forecasts)),
            "entry_forecast_passes": int(entry_forecasts["pass"].sum()) if len(entry_forecasts) else 0,
            "signals_selected": int(entry_forecasts.get("selected_for_signal", pd.Series(dtype=bool)).sum()) if len(entry_forecasts) else 0,
            "signals_by_source": signal_source_counts,
            "forward_outcomes_available": int(
                pd.to_numeric(
                    forward.get("realized_forward_return", pd.Series(dtype=float)),
                    errors="coerce",
                ).notna().sum()
            ) if len(forward) else 0,
            "calibration_cache_entries": int(len(cal_cache)),
            "forecast_cache_entries": int(len(forecast_cache)),
            "boundary_entries_rejected": int(boundary_entries_rejected),
        },
        "standalone_spec": s,
        "model_spec": spec["model"],
        "diagnostic_ic_summary": ic_summary.to_dict("records") if len(ic_summary) else [],
        "diagnostic_ic_summary_by_source": ic_summary_by_source.to_dict("records") if len(ic_summary_by_source) else [],
        "diagnostic_ic_nonoverlap_summary": ic_nonoverlap_summary.to_dict("records") if len(ic_nonoverlap_summary) else [],
        "entry_order_audit": entry_order_summary,
        "selection_lift_summary": {
            "forecast_dates": int(len(selection_lift)),
            "mean_lift": float(
                pd.to_numeric(
                    selection_lift.get("selection_lift_vs_other_passing", pd.Series(dtype=float)),
                    errors="coerce",
                ).mean()
            ) if len(selection_lift) else None,
            "median_lift": float(
                pd.to_numeric(
                    selection_lift.get("selection_lift_vs_other_passing", pd.Series(dtype=float)),
                    errors="coerce",
                ).median()
            ) if len(selection_lift) else None,
            "positive_lift_fraction": float(
                (
                    pd.to_numeric(
                        selection_lift.get("selection_lift_vs_other_passing", pd.Series(dtype=float)),
                        errors="coerce",
                    ) > 0
                ).mean()
            ) if len(selection_lift) else None,
        },
        "right_boundary": {"policy": spec["boundary"]["policy"], "entries_rejected": int(boundary_entries_rejected), "reruns": 0},
        "research_guardrail": (
            "DEVELOPMENT ONLY. Patch 2.9.3 loaded no rows after 2024-12-31. 2025-present is sealed from further tuning, but prior patches already inspected 2025-2026 so it is not a pristine retroactive holdout. Security-discontinuation exits are explicitly flagged and zero-recovery stressed."
            if mode == "development"
            else "SEALED CONFIRMATION WINDOW. Do not tune or revise the frozen strategy in response to these 2025-present results; prior patches had already inspected part of this period."
        ),
    }
    manifest = {
        "production_identity": identity,
        "spec_patch": spec["patch"],
        "research_mode": mode,
        "run_id": run_id,
        "history_start": str(window["history_start"]),
        "score_start_date": str(window["score_start"]),
        "score_end_date": str(window.get("score_end")) if window.get("score_end") else None,
        "min_loaded_date": min_loaded.strftime("%Y-%m-%d"),
        "max_loaded_date": max_loaded.strftime("%Y-%m-%d"),
        "feature_min_loaded_date": min_loaded.strftime("%Y-%m-%d"),
        "calibration_min_loaded_date": calibration_min_loaded.strftime("%Y-%m-%d"),
        "feature_warmup_global_sessions": int(feature_warmup_sessions),
        "normalized_crsp_backfill_global_sessions": int(normalized_backfill_sessions),
        "combined_warmup_global_sessions": int(warmup_global_sessions),
        "rows_loaded": len(daily),
        "output_dir": str(out),
        "retired_shadow": True,
        "development_holdout_protocol": spec["research_protocol"],
        "publication_policy": "STAGE_COMPLETE_BUNDLE_THEN_REPLACE_PREVIOUS_OUTPUT_DIRECTORY",
    }
    _publish_output_bundle(
        out,
        {
            "equity_curve.csv": eq,
            "trade_log.csv": trades,
            "event_log.csv": events,
            "model_forecasts.csv": forecasts,
            "forecast_forward_outcomes.csv": forward,
            "forecast_quantile_diagnostics.csv": qdiag,
            "forecast_date_ic.csv": ic_detail,
            "forecast_monthly_ic.csv": ic_detail,  # legacy alias; rows are exact forecast dates, not calendar months
            "forecast_ic_summary.csv": ic_summary,
            "forecast_ic_by_source.csv": ic_by_source,
            "forecast_ic_summary_by_source.csv": ic_summary_by_source,
            "forecast_ic_nonoverlap.csv": ic_nonoverlap,
            "forecast_ic_nonoverlap_summary.csv": ic_nonoverlap_summary,
            "forecast_selection_lift.csv": selection_lift,
            "entry_order_audit.csv": entry_order_audit,
            "forecast_path_risk_diagnostics.csv": path_risk_diag,
            "security_discontinuation_diagnostics.csv": discontinuation_trades,
        },
        {
            "summary.json": summary,
            "manifest.json": manifest,
            "RUN_STARTED.json": {**run_started, "status": "COMPLETED"},
        },
        patch=str(spec["patch"]),
        run_id=run_id,
    )
    shutil.rmtree(_checkpoint_dir(mode), ignore_errors=True)

    print("-" * 118)
    print(
        f"CAGR={pm['cagr']:.2%} | Sharpe={pm['sharpe']:.3f} | "
        f"MaxDD={pm['max_drawdown']:.2%} | Trades={tm['closed_trades']}"
    )
    print(
        f"Entry forecasts passed: {summary['forecast_summary']['entry_forecast_passes']}/"
        f"{summary['forecast_summary']['entry_forecasts']}"
    )
    print(
        f"Signals selected: {summary['forecast_summary']['signals_selected']} "
        f"{summary['forecast_summary']['signals_by_source']} | "
        f"Avg gross exposure={risk_diag.get('average_gross_exposure', 0.0):.2%}"
    )
    print(f"Outputs: {out}")
    if mode == "development":
        print("[HOLDOUT STATUS] 2025-present remains LOCKED. Do not run holdout until development tuning is final.")
    print("=" * 118)


def _failure_context() -> tuple[str, Path]:
    mode = "holdout" if "--mode" in sys.argv and "holdout" in sys.argv else "development"
    try:
        spec = load_spec(ROOT)
        window = _window_for_mode(spec, mode)
        out = ROOT / str(window["output_dir"])
    except Exception:
        out = ROOT / "outputs" / "research" / f"stochastic_standalone_patch_2_9_3_{mode}_failed"
    return mode, out


def _write_failure_artifacts(exc: BaseException) -> None:
    mode, out = _failure_context()
    try:
        out.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        (out / "ERROR_TRACEBACK.txt").write_text(tb, encoding="utf-8")
        started_path = out / "RUN_STARTED.json"
        run_id = None
        if started_path.exists():
            try:
                run_id = json.loads(started_path.read_text(encoding="utf-8")).get("run_id")
            except Exception:
                pass
        write_json(out / "RUN_FAILED.json", {
            "patch": "2.9.3",
            "research_mode": mode,
            "run_id": run_id,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_file": str((out / "ERROR_TRACEBACK.txt").resolve()),
        })
        print(f"[FAILURE TRACE] {(out / 'ERROR_TRACEBACK.txt').resolve()}")
        print(f"[FAILURE MARKER] {(out / 'RUN_FAILED.json').resolve()}")
        cp = _checkpoint_dir(mode)
        if (cp / "CHECKPOINT.json").exists():
            print(f"[RECOVERABLE CHECKPOINT] {cp.resolve()}")
            print("[RECOVERY] Run the matching Patch 2.9.3 checkpoint-resume launcher after installing a reporting-only fix.")
    except Exception as artifact_error:
        print(f"[WARN] Could not write failure artifacts: {artifact_error}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _write_failure_artifacts(exc)
        raise
