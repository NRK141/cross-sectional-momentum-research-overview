from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

RS = Path(__file__).resolve().parents[1]
if str(RS) not in sys.path:
    sys.path.insert(0, str(RS))

from data_access import HistoryStore
from reporting import cross_sectional_ic, cross_sectional_ic_by_source, non_overlapping_forecast_frame, realized_quantile_diagnostics
from run_stochastic_standalone import (
    _allocate_signal_weights,
    _construct_signal_batch,
    _calibration_at,
    _drift_prefilter_pass,
    _entry_pass,
    _prefilter_score,
    _publish_output_bundle,
    _ranking_score,
    _risk_allocation_score,
    _risk_exit,
    _rebalance_date_sets,
    _last_full_horizon_signal_date,
    _window_for_mode,
)
from stochastic_core import (
    Calibration,
    calibrate_heston_merton,
    regularize_calibration_cross_section,
    simulate_price_paths,
    simulate_terminal_distribution,
    stable_seed,
)


def synthetic_returns(n=504, seed=1):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.012, size=n)
    r[100] = -0.09
    r[260] = 0.11
    r[420] = -0.075
    return r


def standalone_spec():
    return {
        "standalone": {
            "require_positive_short_drift": True,
            "require_positive_long_drift": True,
            "min_expected_return": 0.0,
            "min_expected_shortfall_5": -0.45,
            "max_probability_loss_10": 0.40,
            "position_volatility_floor": 0.12,
            "risk_allocation_power": 0.50,
            "position_sizing": "FIXED_TARGET_WEIGHT",
            "fixed_position_weight": 0.50,
            "target_gross_exposure": 1.0,
            "max_gross_exposure": 1.0,
            "min_position_weight": 0.25,
            "max_position_weight": 0.50,
            "max_new_entries_per_rebalance": 2,
            "max_active_positions": 2,
            "max_sector_weight": 1.0,
            "weekly_vacancy_refill_enabled": True,
            "weekly_refill_max_new_entries": 2,
            "weekly_refill_prefilter_n": 100,
            "weekly_refill_n_paths": 512,
            "exit_consensus_drift_below": 0.0,
            "negative_drift_rechecks_required": 2,
            "recheck_joint_tail_expected_shortfall_below": -0.45,
            "recheck_joint_tail_probability_loss_10_above": 0.40,
        }
    }


def test_calibration_has_regularized_drift_components():
    c = calibrate_heston_merton(synthetic_returns())
    assert c is not None
    assert c.n_obs == 504
    assert np.isfinite(c.mu)
    assert np.isfinite(c.mu_short)
    assert np.isfinite(c.mu_long)
    assert -1.0 <= c.drift_agreement <= 1.0
    assert abs(c.mu) <= 0.75 + 1e-12
    assert c.v0 > 0 and c.theta > 0


def test_drift_consensus_is_between_short_and_long_components():
    c = calibrate_heston_merton(synthetic_returns(seed=3))
    assert c is not None
    lo, hi = sorted([c.mu_short, c.mu_long])
    assert lo - 1e-12 <= c.mu <= hi + 1e-12




def test_cross_sectional_regularization_shrinks_noisy_drift_and_sparse_jumps():
    a = _cal(mu_short=0.30, mu_long=0.25, mu=0.28)
    b = _cal(mu_short=0.02, mu_long=0.01, mu=0.015)
    # supply uncertainty diagnostics and sparse jump counts
    from dataclasses import replace
    a = replace(a, mu_se=0.30, mu_raw=a.mu, jump_count=1, lambda_jump=8.0, lambda_jump_raw=8.0)
    b = replace(b, mu_se=0.03, mu_raw=b.mu, jump_count=10, lambda_jump=1.0, lambda_jump_raw=1.0)
    reg, diag = regularize_calibration_cross_section({1: a, 2: b}, jump_prior_strength=8.0)
    assert diag["n"] == 2
    assert abs(reg[1].mu - diag["mu_prior"]) < abs(a.mu - diag["mu_prior"])
    assert reg[1].mu_shrink_weight <= reg[2].mu_shrink_weight
    assert reg[1].lambda_jump < a.lambda_jump


def test_simulation_is_reproducible_by_seed():
    c = calibrate_heston_merton(synthetic_returns())
    assert c is not None
    p1 = simulate_price_paths(100.0, c, horizon_days=10, n_paths=128, seed=42)
    p2 = simulate_price_paths(100.0, c, horizon_days=10, n_paths=128, seed=42)
    assert np.array_equal(p1, p2)
    assert np.all(np.isfinite(p1))
    assert np.all(p1 > 0)


def test_terminal_distribution_probabilities_are_valid():
    c = calibrate_heston_merton(synthetic_returns())
    assert c is not None
    s = simulate_terminal_distribution(50.0, c, horizon_days=20, n_paths=512, seed=7)
    assert 0 <= s.probability_positive <= 1
    assert 0 <= s.probability_loss_10 <= 1
    assert 0 <= s.jump_probability_horizon <= 1
    assert np.isfinite(s.expected_return)
    assert np.isfinite(s.expected_shortfall_5)


def test_stable_seed_changes_with_date_or_permno():
    a = stable_seed(2300, "standalone", 12345, "2025-01-02")
    b = stable_seed(2300, "standalone", 12345, "2025-01-03")
    c = stable_seed(2300, "standalone", 12346, "2025-01-02")
    assert a != b and a != c
    assert a == stable_seed(2300, "standalone", 12345, "2025-01-02")


def _cal(mu_short=0.20, mu_long=0.10, mu=0.15, theta=0.04):
    return Calibration(
        n_obs=504,
        mu=mu,
        mu_short=mu_short,
        mu_long=mu_long,
        drift_agreement=0.5,
        v0=0.05,
        theta=theta,
        kappa=2.0,
        xi=0.5,
        rho=-0.3,
        lambda_jump=2.0,
        jump_mu_log=-0.01,
        jump_sigma_log=0.05,
        jump_count=4,
        robust_daily_sigma=0.015,
    )


def test_prefilter_no_longer_rejects_negative_drift_signs():
    spec = standalone_spec()
    assert _drift_prefilter_pass(_cal(0.2, 0.1), spec)
    assert _drift_prefilter_pass(_cal(-0.01, 0.1, mu=0.04), spec)
    assert _drift_prefilter_pass(_cal(0.1, -0.01, mu=0.04), spec)


def test_prefilter_score_uses_regularized_consensus_mu():
    assert _prefilter_score(_cal(0.30, 0.12, mu=0.07), standalone_spec()) == 0.07
    assert _prefilter_score(_cal(0.08, 0.25, mu=-0.02), standalone_spec()) == -0.02


def test_entry_pass_uses_broad_distribution_guards_not_probability_positive():
    spec = standalone_spec()
    f = {
        "expected_return": 0.01,
        "probability_positive": 0.10,
        "expected_shortfall_5": -0.30,
        "probability_loss_10": 0.20,
    }
    passed, reason = _entry_pass(f, spec)
    assert passed and reason == ""


def test_entry_pass_rejects_negative_mean_or_catastrophic_tail():
    spec = standalone_spec()
    passed, reason = _entry_pass({"expected_return": -0.001, "expected_shortfall_5": -0.2, "probability_loss_10": 0.1}, spec)
    assert not passed and "minimum_expected_return" in reason
    passed, reason = _entry_pass({"expected_return": 0.02, "expected_shortfall_5": -0.46, "probability_loss_10": 0.1}, spec)
    assert not passed and "minimum_expected_shortfall" in reason


def test_ranking_score_combines_horizon_drift_and_median_without_double_negative_sign_trap():
    spec = standalone_spec()
    spec["standalone"]["forecast_horizon_days"] = 20
    pos = _ranking_score(0.20, {"median_return": 0.04}, spec)
    neg = _ranking_score(-0.20, {"median_return": -0.04}, spec)
    expected_pos = 0.5 * (0.20 * 20 / 252) + 0.5 * 0.04
    assert abs(pos - expected_pos) < 1e-12
    assert neg < 0.0  # old multiplicative formula incorrectly made this positive


def test_inverse_sqrt_vol_allocation_score_is_softer_for_high_volatility():
    spec = standalone_spec()
    low_score, rv_low = _risk_allocation_score({"terminal_volatility": 0.16, "theta": 0.0256}, spec)
    high_score, rv_high = _risk_allocation_score({"terminal_volatility": 0.64, "theta": 0.4096}, spec)
    assert rv_high > rv_low
    assert high_score < low_score
    # 4x volatility becomes 2x lower score under inverse-square-root sizing, not 4x.
    assert abs((low_score / high_score) - 2.0) < 1e-12


def test_signal_weights_target_two_half_equity_positions():
    spec = standalone_spec()
    candidates = [
        {"model_score": 9-i, "risk_allocation_score": score, "permno": i}
        for i, score in enumerate([3.0, 2.8, 2.5, 2.2], start=1)
    ]
    out = _allocate_signal_weights(candidates, 1.0, spec)
    weights = [x["target_weight"] for x in out]
    assert len(out) == 2
    assert weights == [0.5, 0.5]
    assert abs(sum(weights) - 1.0) < 1e-12


def test_signal_weights_use_remaining_room_if_existing_position_appreciated():
    spec = standalone_spec()
    candidates = [{"model_score": 2-i, "risk_allocation_score": 1.0, "permno": i} for i in range(2)]
    out = _allocate_signal_weights(candidates, 0.40, spec, max_new_entries=1)
    assert len(out) == 1
    assert abs(out[0]["target_weight"] - 0.40) < 1e-12


def test_recheck_does_not_use_heston_drift_as_alpha_exit_signal():
    spec = standalone_spec()
    f = {
        "mu": -0.50,
        "probability_positive": 0.10,
        "expected_shortfall_5": -0.20,
        "probability_loss_10": 0.15,
        "mc_converged": True,
        "mc_mean_se": 0.0001,
    }
    exit_now, reason, streak = _risk_exit(f, spec, 0)
    assert not exit_now and reason == "" and streak == 0
    positive = dict(f, mu=0.01)
    exit_now, reason, streak = _risk_exit(positive, spec, streak)
    assert not exit_now and streak == 0


def test_recheck_catastrophic_tail_exits_immediately():
    spec = standalone_spec()
    exit_now, reason, streak = _risk_exit({
        "mu": 0.02,
        "expected_shortfall_5": -0.50,
        "probability_loss_10": 0.50,
    }, spec, 0)
    assert exit_now and "JOINT_CATASTROPHIC_TAIL" in reason and streak == 0


def test_output_publication_replaces_stale_bundle_only_after_complete(tmp_path):
    out = tmp_path / "stochastic_standalone"
    out.mkdir()
    (out / "summary.json").write_text('{"patch":"2.2.1"}', encoding="utf-8")
    frames = {"equity_curve.csv": pd.DataFrame({"equity": [100.0, 101.0]})}
    summary = {"patch": "2.3.5", "run_id": "r1"}
    manifest = {"spec_patch": "2.3.5", "run_id": "r1"}
    started = {"patch": "2.3.5", "run_id": "r1", "status": "COMPLETED"}
    _publish_output_bundle(
        out,
        frames,
        {"summary.json": summary, "manifest.json": manifest, "RUN_STARTED.json": started},
        patch="2.3.5",
        run_id="r1",
    )
    import json
    assert json.loads((out / "summary.json").read_text())["patch"] == "2.3.5"
    assert json.loads((out / "manifest.json").read_text())["spec_patch"] == "2.3.5"
    assert json.loads((out / "RUN_COMPLETE.json").read_text())["status"] == "PASS"
    assert (out / "equity_curve.csv").exists()


def test_history_forward_path_matches_next_close_and_held_session_count():
    d = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=6, freq="D"),
        "permno": [1] * 6,
        "ret1": [np.nan, .01, .01, .01, .01, .01],
        "close": [100, 101, 102, 103, 104, 105],
    })
    h = HistoryStore.from_daily(d)
    out = h.realized_forward_path(1, pd.Timestamp("2025-01-01"), 3)
    assert out is not None
    assert out["realized_entry_date"] == pd.Timestamp("2025-01-02")
    assert out["realized_terminal_date"] == pd.Timestamp("2025-01-04")
    assert abs(out["realized_forward_return"] - (1.01**2 - 1)) < 1e-12


def test_cross_sectional_ic_reports_positive_monotonic_signal():
    dates = [pd.Timestamp("2025-01-31")] * 25 + [pd.Timestamp("2025-02-28")] * 25
    signal = np.tile(np.arange(25, dtype=float), 2)
    realized = signal * 0.01
    frame = pd.DataFrame({"date": dates, "signal": signal, "realized_forward_return": realized})
    detail, summary = cross_sectional_ic(frame, ["signal"], min_names=20)
    assert len(detail) == 2
    assert np.allclose(detail["spearman_ic"], 1.0)
    assert float(summary.iloc[0]["mean_ic"]) == 1.0


def test_quantile_diagnostics_can_use_forward_return_column():
    frame = pd.DataFrame({"metric": np.arange(20), "realized_forward_return": np.arange(20) / 100.0})
    out = realized_quantile_diagnostics(frame, ["metric"], realized_col="realized_forward_return", buckets=5)
    assert len(out) == 5
    assert out.iloc[-1]["mean_realized_return"] > out.iloc[0]["mean_realized_return"]


def test_cross_sectional_ic_is_scipy_free_and_handles_ties():
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-31")] * 6,
        "metric": [1.0, 1.0, 2.0, 3.0, 3.0, 4.0],
        "realized_forward_return": [10.0, 10.0, 20.0, 30.0, 30.0, 40.0],
    })
    detail, summary = cross_sectional_ic(frame, ["metric"], min_names=2)
    assert len(detail) == 1
    assert np.isclose(float(detail.iloc[0]["spearman_ic"]), 1.0)
    assert np.isclose(float(summary.iloc[0]["mean_ic"]), 1.0)


def test_rebalance_dates_include_week_end_and_month_end():
    dates = pd.DatetimeIndex(pd.to_datetime([
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
        "2024-01-29", "2024-01-30", "2024-01-31",
    ]))
    month_end, week_end = _rebalance_date_sets(dates)
    assert pd.Timestamp("2024-01-31") in month_end
    assert pd.Timestamp("2024-01-05") in week_end
    assert pd.Timestamp("2024-01-12") in week_end
    assert pd.Timestamp("2024-01-31") in week_end


def test_patch_2_9_3_windows_use_pre2015_warmup_and_keep_2025_out_of_development():
    import json
    spec = json.loads((RS / "STOCHASTIC_RESEARCH_SPEC.json").read_text())
    dev = _window_for_mode(spec, "development")
    hold = _window_for_mode(spec, "holdout")
    assert dev["history_start"] == "2014-01-01"
    assert dev["score_start"] == "2015-01-01"
    assert dev["score_end"] == "2024-12-31"
    assert hold["score_start"] == "2025-01-01"
    assert hold["score_end"] is None
    assert spec["standalone"]["max_active_positions"] == 5
    assert spec["standalone"]["fixed_position_weight"] == 0.2
    assert spec["standalone"]["target_gross_exposure"] == 1.0
    assert spec["standalone"]["weekly_vacancy_refill_enabled"] is False
    assert spec["standalone"]["entry_schedule"] == "MONTH_END_ONLY"


def test_weekly_refill_allocation_can_fill_two_live_style_slots():
    spec = standalone_spec()
    candidates = [{"model_score": 9-i, "risk_allocation_score": 1.0, "permno": i} for i in range(8)]
    out = _allocate_signal_weights(candidates, 1.0, spec, max_new_entries=2)
    assert len(out) == 2
    assert [x["target_weight"] for x in out] == [0.5, 0.5]


def test_simulation_score_end_excludes_later_dates():
    # Minimal no-signal frame is enough to prove the simulation clock is hard-bounded.
    d = pd.DataFrame({
        "date": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02"]),
        "permno": [1, 1, 1],
        "ticker": ["X", "X", "X"],
        "sector": ["Test", "Test", "Test"],
        "eligible": [False, False, False],
        "close": [100.0, 101.0, 102.0],
        "ret1": [0.0, 0.01, 0.01],
    })
    for col in ["ret5", "rsi2", "distance_20d_high", "sma20", "sma50", "sma100", "sma200"]:
        d[col] = np.nan
    from run_stochastic_standalone import _simulate_once
    h = HistoryStore.from_daily(d[["date", "permno", "ret1", "close"]])
    cfg = {"execution": {"transaction_cost_bps_per_side": 5, "annualization_days": 252}}
    import json
    full_spec = json.loads((RS / "STOCHASTIC_RESEARCH_SPEC.json").read_text())
    eq, trades, events, forecasts, unresolved = _simulate_once(
        d, cfg, full_spec,
        h, {}, {}, set(),
        score_start=pd.Timestamp("2024-12-30"),
        score_end=pd.Timestamp("2024-12-31"),
    )
    assert pd.Timestamp(eq["date"].max()) == pd.Timestamp("2024-12-31")
    assert len(trades) == 0 and len(unresolved) == 0


def test_full_horizon_boundary_cutoff_for_next_close_entry():
    dates = pd.bdate_range("2024-11-01", periods=40)
    cutoff = _last_full_horizon_signal_date(pd.DatetimeIndex(dates), 20)
    assert cutoff == pd.Timestamp(dates[-21])
    # Exactly 20 global sessions remain after the cutoff signal date.
    assert int((pd.DatetimeIndex(dates) > cutoff).sum()) == 20


def test_full_horizon_boundary_returns_none_when_window_too_short():
    dates = pd.bdate_range("2024-12-01", periods=20)
    assert _last_full_horizon_signal_date(pd.DatetimeIndex(dates), 20) is None


def test_patch_2_9_3_boundary_policy_has_no_iterative_reruns():
    import json
    spec = json.loads((RS / "STOCHASTIC_RESEARCH_SPEC.json").read_text())
    b = spec["boundary"]
    assert b["policy"] == "PREENTRY_REJECT_IF_FULL_MAX_HOLD_HORIZON_CANNOT_FIT_BEFORE_DATASET_END"
    assert b["full_horizon_required"] is True
    assert b["required_future_global_sessions_after_signal"] == 20
    assert b["max_reruns"] == 0
    assert b["forced_terminal_exit"] is False


def test_simulation_preentry_boundary_rejects_late_weekly_refill_without_rerun(monkeypatch):
    import json
    import run_stochastic_standalone as runner

    dates = pd.bdate_range("2024-12-02", periods=10)
    d = pd.DataFrame({
        "date": dates,
        "permno": 1,
        "ticker": "X",
        "sector": "Test",
        "eligible": True,
        "close": np.linspace(100.0, 109.0, len(dates)),
        "ret1": 0.001,
    })
    for col in ["ret5", "rsi2", "distance_20d_high", "sma20", "sma50", "sma100", "sma200"]:
        d[col] = np.nan

    spec = json.loads((RS / "STOCHASTIC_RESEARCH_SPEC.json").read_text())
    # Small horizon makes the synthetic sequence easy to reason about.
    spec["standalone"]["max_holding_days"] = 5
    spec["standalone"]["forecast_horizon_days"] = 5
    spec["standalone"]["recheck_every_days"] = 99
    spec["standalone"]["max_active_positions"] = 1
    spec["standalone"]["max_new_entries_per_rebalance"] = 1
    spec["standalone"]["weekly_refill_max_new_entries"] = 1
    spec["standalone"]["target_gross_exposure"] = 0.20
    spec["standalone"]["max_gross_exposure"] = 0.80
    spec["standalone"]["max_sector_weight"] = 0.80
    spec["standalone"]["min_position_weight"] = 0.01
    spec["standalone"]["max_position_weight"] = 0.20

    def fake_discover(**kwargs):
        fr = kwargs["forecast_rows"]
        idx = len(fr)
        fr.append({
            "date": kwargs["date"], "ticker": "X", "permno": 1,
            "forecast_type": kwargs["forecast_type"], "signal_source": "TEST",
            "horizon_days": 5, "prefilter_score": 1.0, "pass": True,
            "reason": "", "model_score": 1.0, "ranking_score": 1.0,
            "target_weight": np.nan, "risk_volatility": 0.20,
            "risk_allocation_score": 1.0,
        })
        return [{
            "permno": 1, "ticker": "X", "signal_date": kwargs["date"],
            "signal_source": "TEST", "model_score": 1.0, "ranking_score": 1.0,
            "risk_allocation_score": 1.0, "risk_volatility": 0.20,
            "forecast_index": idx, "terminal_volatility": 0.20,
            "expected_return": 0.01,
        }]

    monkeypatch.setattr(runner, "_discover_entry_candidates", fake_discover)
    h = HistoryStore.from_daily(d[["date", "permno", "ret1", "close"]])
    cfg = {"execution": {"transaction_cost_bps_per_side": 0, "annualization_days": 252}}
    eq, trades, events, forecasts, unresolved = runner._simulate_once(
        d, cfg, spec, h, {}, {}, set(),
        score_start=dates[0], score_end=dates[-1],
    )
    assert unresolved == []
    assert (events["event"] == "BOUNDARY_SIGNAL_REJECTED").any()
    # No forced terminal exit is manufactured.
    assert not ((events["event"] == "EXIT") & events.get("reason", pd.Series(index=events.index)).eq("FORCED_TERMINAL")).any()


def test_security_discontinuation_exits_after_two_missing_global_sessions(monkeypatch):
    import json
    import run_stochastic_standalone as runner

    dates = pd.bdate_range("2024-01-31", periods=8)
    rows = []
    # Tradable security exists for signal + two held closes, then disappears.
    for i, d in enumerate(dates):
        rows.append({
            "date": d, "permno": 2, "ticker": "DUMMY", "sector": "Test",
            "eligible": False, "close": 50.0 + i, "ret1": 0.0,
        })
        if i <= 2:
            rows.append({
                "date": d, "permno": 1, "ticker": "GONE", "sector": "Test",
                "eligible": True, "close": 100.0 + i, "ret1": 0.001,
            })
    d = pd.DataFrame(rows)
    for col in ["ret5", "rsi2", "distance_20d_high", "sma20", "sma50", "sma100", "sma200"]:
        d[col] = np.nan

    spec = json.loads((RS / "STOCHASTIC_RESEARCH_SPEC.json").read_text())
    spec["standalone"]["max_holding_days"] = 3
    spec["standalone"]["forecast_horizon_days"] = 3
    spec["standalone"]["recheck_every_days"] = 99
    spec["standalone"]["max_active_positions"] = 1
    spec["standalone"]["max_new_entries_per_rebalance"] = 1
    spec["standalone"]["weekly_vacancy_refill_enabled"] = False
    spec["standalone"]["target_gross_exposure"] = 0.20
    spec["standalone"]["max_gross_exposure"] = 0.80
    spec["standalone"]["max_sector_weight"] = 0.80
    spec["standalone"]["min_position_weight"] = 0.01
    spec["standalone"]["max_position_weight"] = 0.20
    spec["security_discontinuation"]["missing_global_sessions_before_exit"] = 2

    def fake_discover(**kwargs):
        if pd.Timestamp(kwargs["date"]) != pd.Timestamp(dates[0]):
            return []
        fr = kwargs["forecast_rows"]
        idx = len(fr)
        fr.append({
            "date": kwargs["date"], "ticker": "GONE", "permno": 1,
            "forecast_type": kwargs["forecast_type"], "signal_source": "TEST",
            "horizon_days": 3, "prefilter_score": 1.0, "pass": True,
            "reason": "", "model_score": 1.0, "ranking_score": 1.0,
            "target_weight": np.nan, "risk_volatility": 0.20,
            "risk_allocation_score": 1.0,
        })
        return [{
            "permno": 1, "ticker": "GONE", "signal_date": kwargs["date"],
            "signal_source": "TEST", "model_score": 1.0, "ranking_score": 1.0,
            "risk_allocation_score": 1.0, "risk_volatility": 0.20,
            "forecast_index": idx, "terminal_volatility": 0.20,
            "expected_return": 0.01,
        }]

    monkeypatch.setattr(runner, "_discover_entry_candidates", fake_discover)
    h = HistoryStore.from_daily(d[["date", "permno", "ret1", "close"]])
    cfg = {"execution": {"transaction_cost_bps_per_side": 0, "annualization_days": 252}}
    eq, trades, events, forecasts, unresolved = runner._simulate_once(
        d, cfg, spec, h, {}, {}, set(),
        score_start=dates[0], score_end=dates[-1],
    )
    assert unresolved == []
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "SECURITY_DISCONTINUATION"
    assert t["exit_price_source"] == "LAST_OBSERVED_CLOSE_PROXY"
    assert pd.Timestamp(t["last_observed_quote_date"]) == pd.Timestamp(dates[2])
    assert int(t["missing_global_sessions_at_exit"]) == 2
    assert abs(float(t["exit_price"]) - (100.0 * 1.001**2)) < 1e-9
    assert float(t["zero_recovery_incremental_loss_usd"]) < 0.0
    missing_events = events[events["event"] == "SECURITY_QUOTE_MISSING"]
    assert len(missing_events) == 2


def test_single_missing_quote_resets_when_security_returns(monkeypatch):
    import json
    import run_stochastic_standalone as runner

    dates = pd.bdate_range("2024-01-31", periods=7)
    rows = []
    for i, dt in enumerate(dates):
        rows.append({"date": dt, "permno": 2, "ticker": "DUMMY", "sector": "Test", "eligible": False, "close": 50.0, "ret1": 0.0})
        if i != 2:  # one-session gap only
            rows.append({"date": dt, "permno": 1, "ticker": "BACK", "sector": "Test", "eligible": True, "close": 100.0 + i, "ret1": 0.001})
    d = pd.DataFrame(rows)
    for col in ["ret5", "rsi2", "distance_20d_high", "sma20", "sma50", "sma100", "sma200"]:
        d[col] = np.nan

    spec = json.loads((RS / "STOCHASTIC_RESEARCH_SPEC.json").read_text())
    spec["standalone"]["max_holding_days"] = 4
    spec["standalone"]["forecast_horizon_days"] = 4
    spec["standalone"]["recheck_every_days"] = 99
    spec["standalone"]["max_active_positions"] = 1
    spec["standalone"]["max_new_entries_per_rebalance"] = 1
    spec["standalone"]["weekly_vacancy_refill_enabled"] = False
    spec["standalone"]["target_gross_exposure"] = 0.20
    spec["standalone"]["max_gross_exposure"] = 0.80
    spec["standalone"]["max_sector_weight"] = 0.80
    spec["standalone"]["min_position_weight"] = 0.01
    spec["standalone"]["max_position_weight"] = 0.20
    spec["security_discontinuation"]["missing_global_sessions_before_exit"] = 2

    def fake_discover(**kwargs):
        if pd.Timestamp(kwargs["date"]) != pd.Timestamp(dates[0]):
            return []
        idx = len(kwargs["forecast_rows"])
        kwargs["forecast_rows"].append({
            "date": kwargs["date"], "ticker": "BACK", "permno": 1,
            "forecast_type": kwargs["forecast_type"], "signal_source": "TEST", "horizon_days": 4,
            "prefilter_score": 1.0, "pass": True, "reason": "", "model_score": 1.0,
            "ranking_score": 1.0, "target_weight": np.nan, "risk_volatility": 0.2,
            "risk_allocation_score": 1.0,
        })
        return [{"permno": 1, "ticker": "BACK", "signal_date": kwargs["date"], "signal_source": "TEST",
                 "model_score": 1.0, "ranking_score": 1.0, "risk_allocation_score": 1.0,
                 "risk_volatility": 0.2, "forecast_index": idx, "terminal_volatility": 0.2,
                 "expected_return": 0.01}]

    monkeypatch.setattr(runner, "_discover_entry_candidates", fake_discover)
    h = HistoryStore.from_daily(d[["date", "permno", "ret1", "close"]])
    cfg = {"execution": {"transaction_cost_bps_per_side": 0, "annualization_days": 252}}
    eq, trades, events, forecasts, unresolved = runner._simulate_once(
        d, cfg, spec, h, {}, {}, set(), score_start=dates[0], score_end=dates[-1]
    )
    assert unresolved == []
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "MODEL_HORIZON"
    assert not (trades["exit_reason"] == "SECURITY_DISCONTINUATION").any()


def test_pre2015_history_allows_calibration_at_start_of_2015():
    dates = pd.bdate_range("2012-01-03", "2015-01-02")
    rng = np.random.default_rng(44)
    rets = rng.normal(0.0003, 0.01, len(dates))
    closes = 100.0 * np.cumprod(1.0 + rets)
    h = HistoryStore.from_daily(pd.DataFrame({"date": dates, "permno": 1, "ret1": rets, "close": closes}))
    import json
    spec = json.loads((RS / "STOCHASTIC_RESEARCH_SPEC.json").read_text())
    cal = _calibration_at(1, pd.Timestamp("2015-01-02"), h, spec, {})
    assert cal is not None
    assert cal.n_obs == 504


def test_sector_aware_signal_builder_backfills_next_ranked_candidate():
    spec = standalone_spec()
    spec["standalone"]["max_sector_weight"] = 0.50
    events = []
    candidates = [
        {"permno": 1, "ticker": "TECH1", "sector": "Tech", "model_score": 3.0},
        {"permno": 2, "ticker": "ENERGY1", "sector": "Energy", "model_score": 2.0},
    ]
    out = _construct_signal_batch(
        candidates,
        current_gross=0.50,
        current_sector_weights={"Tech": 0.50},
        vacancies=1,
        spec=spec,
        max_new_entries=1,
        date=pd.Timestamp("2024-01-31"),
        events=events,
    )
    assert len(out) == 1
    assert out[0]["ticker"] == "ENERGY1"
    assert any(e["event"] == "SIGNAL_CANDIDATE_SKIPPED_CAPACITY" and e["reason"] == "SECTOR_CAP" for e in events)


def test_source_split_and_nonoverlap_ic_diagnostics():
    rows = []
    # Weekly dates overlap a 20-session horizon; only a subset should survive non-overlap selection.
    for source, dates in {
        "MONTH_END": pd.to_datetime(["2024-01-31", "2024-02-29"]),
        "WEEKLY_REFILL": pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26", "2024-02-02"]),
    }.items():
        for dt in dates:
            for i in range(25):
                rows.append({
                    "date": dt,
                    "signal_source": source,
                    "metric": float(i),
                    "realized_forward_return": float(i) / 100.0,
                    "realized_entry_date": dt + pd.offsets.BDay(1),
                    "realized_terminal_date": dt + pd.offsets.BDay(20),
                })
    frame = pd.DataFrame(rows)
    detail, summary = cross_sectional_ic_by_source(frame, ["metric"], min_names=20)
    assert set(summary["signal_source"]) == {"MONTH_END", "WEEKLY_REFILL"}
    assert "forecast_dates" in summary.columns
    non = non_overlapping_forecast_frame(frame)
    assert non[non["signal_source"] == "WEEKLY_REFILL"]["date"].nunique() < 5
    assert non[non["signal_source"] == "MONTH_END"]["date"].nunique() <= 2


def test_same_close_exit_releases_slot_before_pending_entry(monkeypatch):
    import json
    import run_stochastic_standalone as runner

    dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08"])
    rows = []
    for dt in dates:
        a_px = 80.0 if dt == pd.Timestamp("2024-02-05") else 100.0
        for perm, ticker, px in [(1, "A", a_px), (2, "B", 50.0)]:
            prev_price = (100.0 if perm == 1 else 50.0)
            ret = (px / prev_price - 1.0) if dt == pd.Timestamp("2024-02-05") and perm == 1 else ((100.0/80.0 - 1.0) if dt == pd.Timestamp("2024-02-06") and perm == 1 else 0.0)
            rows.append({"date": dt, "permno": perm, "ticker": ticker, "sector": "Test", "eligible": True, "close": px, "ret1": ret})
    d = pd.DataFrame(rows)
    for col in ["ret5", "rsi2", "distance_20d_high", "sma20", "sma50", "sma100", "sma200"]:
        d[col] = np.nan
    spec = json.loads((RS / "STOCHASTIC_RESEARCH_SPEC.json").read_text())
    spec["standalone"].update({
        "max_holding_days": 4,
        "forecast_horizon_days": 4,
        "recheck_every_days": 99,
        "max_active_positions": 2,
        "max_new_entries_per_rebalance": 1,
        "weekly_refill_max_new_entries": 1,
        "target_gross_exposure": 1.0,
        "max_gross_exposure": 1.0,
        "fixed_position_weight": 0.5,
        "min_position_weight": 0.25,
        "max_position_weight": 0.5,
        "max_sector_weight": 1.0,
        "entry_schedule": "BOTH",
        "weekly_vacancy_refill_enabled": True,
    })

    def fake_discover(**kwargs):
        dt = pd.Timestamp(kwargs["date"])
        target = 1 if dt == pd.Timestamp("2024-01-31") else (2 if dt == pd.Timestamp("2024-02-02") else None)
        if target is None:
            return []
        ticker = "A" if target == 1 else "B"
        idx = len(kwargs["forecast_rows"])
        kwargs["forecast_rows"].append({
            "date": dt, "ticker": ticker, "permno": target, "sector": "Test",
            "forecast_type": kwargs["forecast_type"], "signal_source": "TEST", "horizon_days": 4,
            "prefilter_score": 1.0, "pass": True, "reason": "", "model_score": 1.0,
            "ranking_score": 1.0, "target_weight": np.nan, "risk_volatility": 0.2,
            "risk_allocation_score": 1.0,
        })
        return [{"permno": target, "ticker": ticker, "sector": "Test", "signal_date": dt, "signal_source": "TEST",
                 "model_score": 1.0, "ranking_score": 1.0, "risk_allocation_score": 1.0,
                 "risk_volatility": 0.2, "forecast_index": idx, "terminal_volatility": 0.2,
                 "expected_return": 0.01}]

    monkeypatch.setattr(runner, "_discover_entry_candidates", fake_discover)
    h = HistoryStore.from_daily(d[["date", "permno", "ret1", "close"]])
    cfg = {"execution": {"transaction_cost_bps_per_side": 0, "annualization_days": 252}}
    eq, trades, events, forecasts, unresolved = runner._simulate_once(
        d, cfg, spec, h, {}, {}, set(), score_start=dates[0], score_end=dates[-1]
    )
    assert unresolved == []
    # A exits at the Feb-05 close and B must be able to enter at that same close.
    exit_a = events[(events["event"] == "EXIT") & (events["ticker"] == "A")]
    entry_b = events[(events["event"] == "ENTRY") & (events["ticker"] == "B")]
    assert len(exit_a) == 1 and len(entry_b) == 1
    assert pd.Timestamp(exit_a.iloc[0]["date"]) == pd.Timestamp(entry_b.iloc[0]["date"]) == pd.Timestamp("2024-02-05")


def test_missing_next_close_order_is_explicitly_cancelled(monkeypatch):
    import json
    import run_stochastic_standalone as runner

    dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02"])
    rows = [
        {"date": dates[0], "permno": 1, "ticker": "GONE", "sector": "Test", "eligible": True, "close": 100.0, "ret1": 0.0},
        {"date": dates[0], "permno": 2, "ticker": "DUMMY", "sector": "Test", "eligible": False, "close": 50.0, "ret1": 0.0},
        {"date": dates[1], "permno": 2, "ticker": "DUMMY", "sector": "Test", "eligible": False, "close": 50.0, "ret1": 0.0},
        {"date": dates[2], "permno": 2, "ticker": "DUMMY", "sector": "Test", "eligible": False, "close": 50.0, "ret1": 0.0},
    ]
    d = pd.DataFrame(rows)
    for col in ["ret5", "rsi2", "distance_20d_high", "sma20", "sma50", "sma100", "sma200"]:
        d[col] = np.nan
    spec = json.loads((RS / "STOCHASTIC_RESEARCH_SPEC.json").read_text())
    spec["standalone"].update({"max_holding_days": 1, "forecast_horizon_days": 1, "weekly_vacancy_refill_enabled": False})

    def fake_discover(**kwargs):
        if pd.Timestamp(kwargs["date"]) != dates[0]:
            return []
        idx = len(kwargs["forecast_rows"])
        kwargs["forecast_rows"].append({"date": dates[0], "ticker": "GONE", "permno": 1, "sector": "Test", "forecast_type": kwargs["forecast_type"], "signal_source": "TEST", "horizon_days": 1, "prefilter_score": 1.0, "pass": True, "reason": "", "model_score": 1.0, "ranking_score": 1.0, "target_weight": np.nan, "risk_volatility": 0.2, "risk_allocation_score": 1.0})
        return [{"permno": 1, "ticker": "GONE", "sector": "Test", "signal_date": dates[0], "signal_source": "TEST", "model_score": 1.0, "ranking_score": 1.0, "risk_allocation_score": 1.0, "risk_volatility": 0.2, "forecast_index": idx, "terminal_volatility": 0.2, "expected_return": 0.01}]

    monkeypatch.setattr(runner, "_discover_entry_candidates", fake_discover)
    h = HistoryStore.from_daily(d[["date", "permno", "ret1", "close"]])
    cfg = {"execution": {"transaction_cost_bps_per_side": 0, "annualization_days": 252}}
    eq, trades, events, forecasts, unresolved = runner._simulate_once(d, cfg, spec, h, {}, {}, set(), score_start=dates[0], score_end=dates[-1])
    cancels = events[events["event"] == "ENTRY_ORDER_CANCELLED"]
    assert len(cancels) == 1
    assert cancels.iloc[0]["reason"] == "NO_NEXT_CLOSE_QUOTE"
    assert len(trades) == 0 and unresolved == []


def test_calibration_history_merge_prefers_feature_rows_and_extends_backwards():
    from data_access import merge_calibration_history

    feature = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-02", "2014-01-03"]),
        "permno": [10001, 10001],
        "ret1": [0.02, 0.03],
        "close": [101.0, 104.0],
    })
    backfill = pd.DataFrame({
        "date": pd.to_datetime(["2013-12-31", "2014-01-02"]),
        "permno": [10001, 10001],
        "ret1": [0.01, 9.99],
        "close": [99.0, 999.0],
    })
    out = merge_calibration_history(feature, backfill)
    assert out["date"].min() == pd.Timestamp("2013-12-31")
    dup = out[(out["permno"] == 10001) & (out["date"] == pd.Timestamp("2014-01-02"))].iloc[0]
    assert dup["ret1"] == 0.02
    assert dup["close"] == 101.0
    assert len(out) == 3


def test_patch_2_9_3_spec_keeps_backfill_calibration_only():
    import json
    spec = json.loads((RS / "STOCHASTIC_RESEARCH_SPEC.json").read_text(encoding="utf-8"))
    assert spec["patch"] == "2.9.3"
    assert spec["experiment_windows"]["development"]["history_start"] == "2014-01-01"
    policy = spec["calibration_history_policy"]
    assert policy["normalized_crsp_role"] == "CALIBRATION_ONLY_PRE_FEATURE_RETURN_CLOSE_BACKFILL"
    assert policy["required_global_sessions_before_score_start"] == spec["model"]["max_calibration_observations"]
    s = spec["standalone"]
    assert s["max_active_positions"] == 5
    assert s["fixed_position_weight"] == 0.2
    assert s["target_gross_exposure"] == 1.0


def test_normalized_calibration_backfill_loads_exact_prior_global_sessions(tmp_path):
    import pytest
    duckdb = pytest.importorskip("duckdb")
    from data_access import load_normalized_calibration_backfill

    norm = tmp_path / "normalized"
    norm.mkdir()
    db = duckdb.connect()
    try:
        db.execute("""
            CREATE TABLE t AS
            SELECT * FROM (VALUES
                (DATE '2013-12-27', 1, 10.0, 0.01),
                (DATE '2013-12-30', 1, 10.1, 0.01),
                (DATE '2013-12-31', 1, 10.2, 0.01),
                (DATE '2014-01-02', 1, 10.3, 0.01)
            ) v(date, permno, close, ret)
        """)
        db.execute(f"COPY t TO '{(norm / 'data.parquet').as_posix()}' (FORMAT PARQUET)")
    finally:
        db.close()

    cfg = {"_root": str(tmp_path), "data": {"normalized_dir": "normalized"}}
    out = load_normalized_calibration_backfill(
        cfg,
        before_date="2014-01-02",
        global_sessions_needed=2,
    )
    assert list(sorted(out["date"].dt.strftime("%Y-%m-%d").unique())) == ["2013-12-30", "2013-12-31"]
    assert out["date"].max() < pd.Timestamp("2014-01-02")


def test_patch_2_9_3_optimizer_grid_has_1458_informative_combinations():
    import json, math
    grid = json.loads((RS / "DEVELOPMENT_OPTIMIZER_GRID.json").read_text(encoding="utf-8"))
    n = math.prod(len(v) for v in grid["variables"].values())
    assert n == 1458
    assert grid["combination_count"] == 1458
    assert "min_expected_shortfall_5" not in grid["variables"]
    assert "max_probability_loss_10" not in grid["variables"]
    assert grid["fixed"]["entry_schedule"] == "MONTH_END_ONLY"
    assert grid["variables"]["max_active_positions"] == [5, 6, 8]
    assert grid["variables"]["daily_close_stop_pct"] == [0.04, 0.05, 0.06]


def test_optimizer_ranking_families_are_supported():
    f = {
        "median_return": 0.04,
        "probability_positive": 0.61,
        "expected_shortfall_5": -0.20,
        "terminal_volatility": 0.30,
        "theta": 0.09,
    }
    spec = standalone_spec()
    for metric in [
        "PERSISTENT_DRIFT_X_SIGNED_SQRT_MEDIAN_RETURN",
        "MEDIAN_RETURN",
        "PROBABILITY_POSITIVE",
        "EXPECTED_SHORTFALL_5",
        "RISK_ALLOCATION_SCORE",
    ]:
        spec["standalone"]["ranking_metric"] = metric
        score = _ranking_score(0.2, f, spec)
        assert np.isfinite(score)


def test_optimizer_entry_probability_gate_is_active_when_configured():
    spec = standalone_spec()
    spec["standalone"]["min_expected_return"] = None
    spec["standalone"]["min_probability_positive"] = 0.60
    f = {"expected_return": -0.5, "probability_positive": 0.59, "expected_shortfall_5": -0.2, "probability_loss_10": 0.1}
    ok, reason = _entry_pass(f, spec)
    assert not ok and "minimum_probability_positive" in reason
    f["probability_positive"] = 0.61
    ok, _ = _entry_pass(f, spec)
    assert ok


def test_patch_2_9_3_history_store_builds_from_bounded_frames_and_later_frame_wins():
    d1 = pd.DataFrame({
        "date": pd.to_datetime(["2014-12-30", "2014-12-31"]),
        "permno": [1, 1],
        "ret1": [0.01, 0.02],
        "close": [100.0, 102.0],
    })
    d2 = pd.DataFrame({
        "date": pd.to_datetime(["2014-12-31", "2015-01-02"]),
        "permno": [1, 1],
        "ret1": [0.03, 0.04],
        "close": [103.0, 107.12],
    })
    h = HistoryStore.from_frames(iter([d1, d2]))
    assert len(h.dates[1]) == 3
    x = h.trailing_returns(1, pd.Timestamp("2014-12-31"), 5)
    assert np.isclose(x[-1], 0.03)
    px = h.prior_closes(1, pd.Timestamp("2014-12-31"), 1)
    assert np.isclose(px[-1], 103.0)


def test_patch_2_9_3_optimizer_uses_memory_bounded_loader_not_full_feature_frame():
    import inspect
    import run_development_optimizer as opt
    src = inspect.getsource(opt)
    assert "load_feature_calendar" in src
    assert "iter_feature_quote_history_chunks" in src
    assert "load_feature_rows_for_dates" in src
    assert "load_feature_history(" not in src
    assert "MEMORY_BOUNDED_RETURN_CONSISTENT_HISTORY_PLUS_SIGNAL_DATE_FEATURE_ROWS" in src


def test_patch_2_9_3_alpha_prefers_uptrend_pullback_to_weak_downtrend():
    import run_stochastic_standalone as runner
    idx = pd.Index([1, 2], name="permno")
    g = pd.DataFrame({
        "close": [105.0, 80.0],
        "sma50": [100.0, 90.0],
        "sma100": [98.0, 95.0],
        "sma200": [90.0, 100.0],
        "ret5": [-0.02, -0.08],
        "rsi2": [20.0, 10.0],
        "distance_20d_high": [-0.05, -0.20],
    }, index=idx)
    a = runner._signal_alpha_frame(g)
    assert a.loc[1, "alpha_balanced"] > a.loc[2, "alpha_balanced"]
    assert a.loc[1, "alpha_trend_heavy"] > a.loc[2, "alpha_trend_heavy"]


def test_patch_2_9_3_drift_reliability_cap_is_enforced():
    from stochastic_core import Calibration, regularize_calibration_cross_section
    def c(mu):
        return Calibration(n_obs=504, mu=mu, mu_short=mu, mu_long=mu, drift_agreement=1.0,
                           v0=.04, theta=.04, kappa=2.0, xi=.4, rho=-.5,
                           lambda_jump=2.0, jump_mu_log=-.02, jump_sigma_log=.08,
                           jump_count=8, robust_daily_sigma=.01, mu_short_se=.001,
                           mu_long_se=.001, mu_se=.001, mu_raw=mu,
                           lambda_jump_raw=2.0, jump_mu_log_raw=-.02, jump_sigma_log_raw=.08)
    out, diag = regularize_calibration_cross_section({1:c(0.60),2:c(0.05),3:c(0.04)}, max_mu_reliability_weight=0.35)
    assert max(v.mu_shrink_weight for v in out.values()) <= 0.3500001
    assert diag["mu_reliability_cap"] == 0.35
    assert out[1].mu < 0.60


def test_patch_2_9_3_diverse_finalists_drop_duplicate_trade_paths():
    import run_development_optimizer as opt
    rows=[]
    for i,(fam,sig,score) in enumerate([
        ("BALANCED","A",.99),("BALANCED","A",.98),("TREND_HEAVY","B",.97),
        ("PULLBACK_HEAVY","C",.96),("TREND_HEAVY","D",.95)
    ],1):
        rows.append({"combo_id":f"G{i:04d}","alpha_model":fam,"trade_signature":sig,"robust_score":score})
    got=opt._select_diverse_finalists(pd.DataFrame(rows),4)
    assert len(got)==4
    assert got["trade_signature"].nunique()==4
    assert set(got["alpha_model"]) == {"BALANCED","TREND_HEAVY","PULLBACK_HEAVY"}


def test_patch_2_9_3_storage_root_and_compact_checkpoint_policy():
    spec = json.loads((RS / "STOCHASTIC_RESEARCH_SPEC.json").read_text(encoding="utf-8"))
    grid = json.loads((RS / "DEVELOPMENT_OPTIMIZER_GRID.json").read_text(encoding="utf-8"))
    assert spec["patch"] == "2.9.3"
    assert spec["outputs"]["optimizer"].startswith(r"outputs\research")
    assert spec["storage"]["minimum_free_gb_start"] >= 20.0
    assert spec["storage"]["minimum_free_gb_checkpoint"] >= 5.0
    assert grid["common_forecast_panel"]["prefilter_checkpoint_buffer"] == 16


def test_patch_2_9_3_atomic_csv_write_and_validators(tmp_path):
    import run_development_optimizer as opt
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "x.csv"
    opt._atomic_to_csv(df, path)
    got = pd.read_csv(path)
    assert got.equals(df)
    assert not list(tmp_path.glob(".x.csv.tmp-*"))


def test_patch_2_9_3_imports_only_complete_291_date_parts_and_compacts_prefilter(tmp_path):
    import run_development_optimizer as opt
    source = tmp_path / "moved_291"
    panels = source / "optimizer_panel_parts"
    pres = source / "optimizer_prefilter_parts"
    panels.mkdir(parents=True)
    pres.mkdir(parents=True)
    (source / "OPTIMIZER_STARTED.json").write_text(json.dumps({"patch": "2.9.1"}), encoding="utf-8")

    good_date = "20200131"
    bad_date = "20200228"
    stops = [0.02, 0.025, 0.03, 0.04]
    rows = []
    for perm in range(1000, 1200):
        for stop in stops:
            rows.append({
                "date": "2020-01-31", "permno": perm, "forecast_stop_pct": stop,
                "alpha_balanced": .5, "alpha_trend_heavy": .5, "alpha_pullback_heavy": .5,
                "risk_volatility": .2, "mc_p_stop": .1, "mc_p_stop_se": .01,
                "mc_converged": True, "forecast_spec_hash": "abc",
            })
    pd.DataFrame(rows).to_csv(panels / f"{good_date}.csv", index=False)
    # Incomplete disk-full style panel: only one stop level.
    pd.DataFrame(rows[:2]).to_csv(panels / f"{bad_date}.csv", index=False)

    pre_rows = []
    for i in range(500):
        pre_rows.append({
            "date": "2020-01-31", "permno": 1000+i, "ticker": f"T{i}", "sector": "X", "close": 100+i/10,
            "persistent_drift": .01, "alpha_prefilter_score": 1.0-i/1000,
            "alpha_balanced": .5, "alpha_trend_heavy": .5, "alpha_pullback_heavy": .5,
            "breadth_above_sma200": .6,
        })
    pd.DataFrame(pre_rows).to_csv(pres / f"{good_date}.csv", index=False)
    pd.DataFrame(pre_rows).to_csv(pres / f"{bad_date}.csv", index=False)

    spec = json.loads((RS / "STOCHASTIC_RESEARCH_SPEC.json").read_text(encoding="utf-8"))
    spec["storage"]["legacy_patch_2_9_1_checkpoint_candidates"] = [str(source)]
    grid = json.loads((RS / "DEVELOPMENT_OPTIMIZER_GRID.json").read_text(encoding="utf-8"))
    out = tmp_path / "new292"
    out.mkdir()
    import pytest
    with pytest.raises(RuntimeError, match="prohibits importing"):
        opt._import_legacy_291_checkpoints(out, spec, grid)
    assert not (out / "optimizer_panel_parts").exists()
