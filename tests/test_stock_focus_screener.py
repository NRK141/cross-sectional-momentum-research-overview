import json
from pathlib import Path

import pandas as pd

from src.screener.stock_focus import _filter_and_rank, simulate_stock_frame

ROOT = Path(__file__).resolve().parents[1]


def cfg():
    return {
        "entry": {
            "min_5d_pullback": -0.12,
            "max_5d_pullback": -0.015,
            "max_rsi2": 40.0,
            "min_distance_from_20d_high": -0.15,
            "max_distance_from_20d_high": -0.015,
            "require_positive_1d_resumption": True,
            "require_close_above_sma50": True,
            "candidate_cooldown_days": 3,
            "max_new_entries_per_day": 3,
        },
        "portfolio": {
            "initial_capital": 100000.0,
            "base_position_weight": 0.125,
            "max_position_weight": 0.15,
            "max_gross_exposure": 0.8,
            "max_sector_weight": 0.3,
        },
        "exit": {
            "initial_stop_pct": 0.04,
            "trailing_activates_at_gain": 0.05,
            "trailing_stop_pct_from_peak": 0.03,
            "trend_break_sma": 20,
            "minimum_holding_days_before_trend_exit": 3,
            "maximum_holding_days": 30,
        },
        "execution": {
            "mode": "next_close",
            "transaction_cost_bps_per_side": 5.0,
            "annualization_days": 252,
        },
    }


def capital_model():
    return {
        "mode": "fixed_dollar_standalone",
        "initial_capital_usd": 5000.0,
        "position_dollars_per_trade": 2000.0,
        "fraction_of_initial_capital": 0.4,
        "share_model": "fractional_equivalent",
        "one_position_per_stock": True,
    }


def row(date, close, *, signal=False, sma20=90.0):
    return {
        "date": date,
        "permno": 1,
        "ticker": "AAA",
        "sector": "Tech",
        "eligible": True,
        "close": close,
        "ret1": 0.01 if signal else -0.01,
        "ret5": -0.05,
        "rsi2": 30.0,
        "distance_20d_high": -0.05,
        "sma20": sma20,
        "sma50": close - 10.0,
    }


def test_fixed_5000_account_uses_2000_fractional_equivalent_notional():
    df = pd.DataFrame([
        row("2026-01-02", 99.0, signal=True),
        row("2026-01-05", 123.45, signal=False),
        row("2026-01-06", 117.00, signal=False),  # stop
    ])
    state = simulate_stock_frame(
        df, cfg(), permno=1, latest_ticker="AAA", latest_sector="Tech", capital_model=capital_model()
    )
    assert state.position is None
    assert len(state.trades) == 1
    trade = state.trades[0]
    assert abs(trade["entry_notional"] - 2000.0) < 1e-9
    result = state.result()
    assert result["screener_initial_capital_usd"] == 5000.0
    assert result["screener_position_dollars_per_trade"] == 2000.0
    assert result["screener_position_fraction_of_initial"] == 0.4


def test_next_close_entry_and_same_close_stop_exit():
    df = pd.DataFrame([
        row("2026-01-02", 99.0, signal=True),
        row("2026-01-05", 100.0, signal=False),  # next-security-session close entry
        row("2026-01-06", 95.0, signal=False),   # same-close 4% stop exit
        row("2026-01-07", 96.0, signal=False),
    ])
    state = simulate_stock_frame(
        df, cfg(), permno=1, latest_ticker="AAA", latest_sector="Tech", capital_model=capital_model()
    )
    assert state.position is None
    assert len(state.trades) == 1
    trade = state.trades[0]
    assert pd.Timestamp(trade["entry_date"]) == pd.Timestamp("2026-01-05")
    assert pd.Timestamp(trade["exit_date"]) == pd.Timestamp("2026-01-06")
    assert trade["exit_reason"] == "STOP"
    assert trade["net_return"] < -0.04


def test_unresolved_terminal_entry_can_be_vetoed_without_forced_exit():
    df = pd.DataFrame([
        row("2026-01-02", 99.0, signal=True),
        row("2026-01-05", 100.0, signal=False, sma20=90.0),
        row("2026-01-06", 101.0, signal=False, sma20=90.0),
        row("2026-01-07", 102.0, signal=False, sma20=90.0),
    ])
    raw = simulate_stock_frame(
        df, cfg(), permno=1, latest_ticker="AAA", latest_sector="Tech", capital_model=capital_model()
    )
    assert raw.position is not None
    assert raw.trades == []

    clean = simulate_stock_frame(
        df,
        cfg(),
        permno=1,
        latest_ticker="AAA",
        latest_sector="Tech",
        veto_entry_dates=["2026-01-05"],
        capital_model=capital_model(),
    )
    assert clean.position is None
    assert clean.trades == []
    assert clean.boundary_entry_rejections == ["2026-01-05"]
    assert clean.result()["closed_trades"] == 0


def test_single_trade_concentration_over_75_percent_is_neutralized_once_for_ranking_metrics():
    df = pd.DataFrame([
        row("2026-01-02", 99.0, signal=True),
        row("2026-01-05", 100.0, signal=False),  # first trade entry
        row("2026-01-06", 120.0, signal=False),  # arm trail
        row("2026-01-07", 115.0, signal=False),  # trail exit, large winner
        row("2026-01-12", 99.0, signal=True),
        row("2026-01-13", 100.0, signal=False),  # second trade entry
        row("2026-01-14", 95.0, signal=False),   # stop, small loser
        row("2026-01-15", 96.0, signal=False),
    ])
    raw = simulate_stock_frame(
        df, cfg(), permno=1, latest_ticker="AAA", latest_sector="Tech", capital_model=capital_model()
    )
    info = raw.concentration_info(0.75)
    assert info["triggered"] is True
    assert float(info["largest_trade_pnl_share"]) > 0.75
    assert pd.Timestamp(info["largest_trade_entry_date"]) == pd.Timestamp("2026-01-05")

    adjusted = simulate_stock_frame(
        df,
        cfg(),
        permno=1,
        latest_ticker="AAA",
        latest_sector="Tech",
        capital_model=capital_model(),
        ranking_excluded_entry_date=info["largest_trade_entry_date"],
    )
    result = adjusted.result()
    assert result["concentration_adjusted"] is True
    assert result["raw_closed_trades"] == 2
    assert result["closed_trades"] == 1
    assert result["excluded_trade_entry_date"] == "2026-01-05"
    assert result["raw_total_return"] > 0
    assert result["total_return"] < 0
    assert sum(bool(t["excluded_from_ranking"]) for t in adjusted.trades) == 1
    # The raw trade chronology remains unchanged; the winner is only neutralized from ranking metrics.
    assert [(t["entry_date"], t["exit_date"], t["exit_reason"]) for t in raw.trades] == [
        (t["entry_date"], t["exit_date"], t["exit_reason"]) for t in adjusted.trades
    ]


def test_single_trade_concentration_cutoff_is_strictly_greater_than_75_percent():
    state = simulate_stock_frame(
        pd.DataFrame([row("2026-01-02", 100.0, signal=False), row("2026-01-05", 100.0, signal=False)]),
        cfg(),
        permno=1,
        latest_ticker="AAA",
        latest_sector="Tech",
        capital_model=capital_model(),
    )
    state.trades = [
        {"entry_date": pd.Timestamp("2026-01-02"), "exit_date": pd.Timestamp("2026-01-03"), "net_return": 0.075, "net_pnl_usd": 75.0},
        {"entry_date": pd.Timestamp("2026-01-04"), "exit_date": pd.Timestamp("2026-01-05"), "net_return": 0.025, "net_pnl_usd": 25.0},
    ]
    assert state.concentration_info(0.75)["triggered"] is False
    state.trades[0]["net_pnl_usd"] = 75.01
    assert state.concentration_info(0.75)["triggered"] is True

def test_original_methodology_filters_remove_low_sharpe_and_low_trade_count_without_pf_gate():
    spec = json.loads((ROOT / "STOCK_SCREENER_SPEC.json").read_text(encoding="utf-8"))
    rows = []
    # 20 valid rows so Top-15 is possible.
    for i in range(20):
        rows.append({
            "permno": 1000 + i,
            "ticker": f"T{i:02d}",
            "sector": "Other",
            "current_close": 50.0 + i,
            "history_start": "2019-01-02",
            "history_end": "2026-08-31",
            "history_sessions": 1500,
            "cagr": 0.05 + i * 0.002,
            "sharpe": 0.90 + i * 0.03,
            "annualized_volatility": 0.1,
            "max_drawdown": -0.10,
            "total_return": 0.5,
            "calmar": 0.5 + i * 0.02,
            "closed_trades": 20 + i,
            "profit_factor": 0.50 if i == 0 else 1.1 + i * 0.02,  # PF must NOT be a gate anymore.
            "average_net_trade_return": 0.01,
            "median_net_trade_return": 0.005,
            "win_rate": 0.45,
            "boundary_entry_rejections": 0,
        })
    # Extra failures.
    rows.append({**rows[0], "permno": 9991, "ticker": "LOW_SHARPE", "sharpe": 0.70})
    rows.append({**rows[0], "permno": 9992, "ticker": "LOW_TRADES", "closed_trades": 19})

    all_results, ranked = _filter_and_rank(pd.DataFrame(rows), spec)
    assert len(ranked) == 20
    assert all_results.loc[all_results["ticker"].eq("LOW_SHARPE"), "filter_failures"].iloc[0] == "sharpe"
    assert all_results.loc[all_results["ticker"].eq("LOW_TRADES"), "filter_failures"].iloc[0] == "closed_trades"
    assert "T00" in set(ranked["ticker"])  # proves PF=0.5 did not filter it out


def test_ranker_uses_original_0_1_minmax_30_30_30_10_composite():
    spec = json.loads((ROOT / "STOCK_SCREENER_SPEC.json").read_text(encoding="utf-8"))
    rows = []
    for i in range(20):
        rows.append({
            "permno": 1000 + i,
            "ticker": f"T{i:02d}",
            "sector": "Other",
            "current_close": 50.0 + i,
            "history_start": "2019-01-02",
            "history_end": "2026-08-31",
            "history_sessions": 1500,
            "cagr": 0.05 + i * 0.002,
            "sharpe": 0.90 + i * 0.03,
            "annualized_volatility": 0.1,
            "max_drawdown": -0.10,
            "total_return": 0.5,
            "calmar": 0.5 + i * 0.02,
            "closed_trades": 20 + i,
            "profit_factor": 1.1 + i * 0.02,
            "average_net_trade_return": 0.01,
            "median_net_trade_return": 0.005,
            "win_rate": 0.45,
            "boundary_entry_rejections": 0,
        })
    _, ranked = _filter_and_rank(pd.DataFrame(rows), spec)
    assert ranked.iloc[0]["ticker"] == "T19"
    assert ranked.iloc[0]["rank"] == 1
    assert abs(float(ranked.iloc[0]["composite_score"]) - 1.0) < 1e-12
    assert abs(float(ranked.iloc[-1]["composite_score"]) - 0.0) < 1e-12
    assert set(["sharpe_norm", "calmar_norm", "cagr_norm", "closed_trades_norm"]).issubset(ranked.columns)
    assert ranked.head(15)["rank"].tolist() == list(range(1, 16))


def test_screener_spec_is_original_methodology_selection_only_and_fixed_capital():
    spec = json.loads((ROOT / "STOCK_SCREENER_SPEC.json").read_text(encoding="utf-8"))
    assert spec["patch"] == "2.1.4"
    assert spec["role"] == "STOCK_SELECTION_ONLY"
    assert spec["top_n"] == 15
    assert spec["monthly_leadership_gate"] is False
    assert spec["minimum_evidence"]["sharpe_strictly_greater_than"] == 0.70
    assert spec["minimum_evidence"]["closed_trades_minimum"] == 20
    assert "profit_factor_strictly_greater_than" not in spec["minimum_evidence"]
    assert spec["score"]["method"].startswith("0-1 min-max")
    assert spec["score"]["trade_count_cap"] is None
    assert spec["capital_model"]["initial_capital_usd"] == 5000.0
    assert spec["capital_model"]["position_dollars_per_trade"] == 2000.0
    assert spec["capital_model"]["fraction_of_initial_capital"] == 0.4
    assert spec["single_trade_concentration"]["enabled"] is True
    assert spec["single_trade_concentration"]["threshold_strictly_greater_than"] == 0.75
    assert spec["single_trade_concentration"]["recursive"] is False
    assert spec["single_trade_concentration"]["preserve_trade_in_audit_log"] is True
    assert "No entry signals" in spec["non_goals"]
    assert "No exit signals" in spec["non_goals"]


def test_insufficient_top15_error_states_previous_outputs_are_preserved():
    spec = json.loads((ROOT / "STOCK_SCREENER_SPEC.json").read_text(encoding="utf-8"))
    rows = []
    for i in range(14):
        rows.append({
            "permno": 2000 + i,
            "ticker": f"Q{i:02d}",
            "sector": "Other",
            "current_close": 50.0,
            "history_start": "2019-01-02",
            "history_end": "2026-08-31",
            "history_sessions": 1500,
            "cagr": 0.05,
            "sharpe": 0.71,
            "annualized_volatility": 0.1,
            "max_drawdown": -0.10,
            "total_return": 0.5,
            "calmar": 0.5,
            "closed_trades": 20,
            "profit_factor": 1.2,
            "average_net_trade_return": 0.01,
            "median_net_trade_return": 0.005,
            "win_rate": 0.45,
            "boundary_entry_rejections": 0,
        })
    import pytest
    with pytest.raises(RuntimeError, match="previous successful outputs remain intact"):
        _filter_and_rank(pd.DataFrame(rows), spec)
