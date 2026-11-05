from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common.metrics import metrics_from_equity, period_metrics, profit_factor, yearly_returns

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
LIVE_BACKTEST = OUTPUTS / "backtest"
LIVE_SCREENER = OUTPUTS / "screener"


def _find(name: str) -> Path:
    """Resolve only from the live backtest namespace, never archives."""
    candidates = list(LIVE_BACKTEST.rglob(name)) if LIVE_BACKTEST.exists() else []
    if not candidates:
        raise FileNotFoundError(f"Could not find live {name} under {LIVE_BACKTEST}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_rankings() -> Path:
    """Resolve only from the live screener namespace, never archives."""
    stable = list(LIVE_SCREENER.rglob("monthly_rankings.csv")) if LIVE_SCREENER.exists() else []
    if stable:
        return max(stable, key=lambda p: p.stat().st_mtime)
    legacy = list(LIVE_SCREENER.rglob("monthly_top*.csv")) if LIVE_SCREENER.exists() else []
    if not legacy:
        raise FileNotFoundError(f"Could not find live monthly rankings under {LIVE_SCREENER}")
    return max(legacy, key=lambda p: p.stat().st_mtime)


def _rank_bucket(rank: float) -> str:
    if pd.isna(rank):
        return "UNMATCHED"
    r = int(rank)
    if r <= 5: return "01-05"
    if r <= 10: return "06-10"
    if r <= 15: return "11-15"
    if r <= 20: return "16-20"
    return "21+"


def _attach_signal_and_rank(trades: pd.DataFrame, signals: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy().reset_index(drop=True)
    t["_trade_id"] = np.arange(len(t))
    sig = signals[["date", "permno", "leadership_score"]].copy().rename(
        columns={"date": "signal_date", "leadership_score": "signal_leadership_score"}
    )
    t = t.sort_values(["entry_date", "permno"])
    sig = sig.sort_values(["signal_date", "permno"])
    matched = pd.merge_asof(
        t, sig,
        left_on="entry_date", right_on="signal_date", by="permno",
        direction="backward", allow_exact_matches=False,
        tolerance=pd.to_timedelta(7, unit="D"),
    )
    matched["rank_effective_month"] = matched["signal_date"].dt.to_period("M").astype(str)
    rankmap = top[["effective_month", "permno", "rank", "leadership_score", "sector"]].drop_duplicates(
        ["effective_month", "permno"]
    )
    rankmap = rankmap.rename(columns={"effective_month": "rank_effective_month", "sector": "screen_sector"})
    matched = matched.merge(rankmap, on=["rank_effective_month", "permno"], how="left", suffixes=("", "_screen"))
    matched["rank_bucket"] = matched["rank"].map(_rank_bucket)
    return matched.sort_values("_trade_id").drop(columns=["_trade_id"])


def _intended_next_day(signal_date: pd.Timestamp, trading_days: list[pd.Timestamp]) -> pd.Timestamp | pd.NaT:
    i = bisect_right(trading_days, signal_date)
    return trading_days[i] if i < len(trading_days) else pd.NaT


def main() -> None:
    summary_p = _find("summary.json")
    trade_p = _find("trade_log.csv")
    event_p = _find("event_log.csv")
    equity_p = _find("equity_curve.csv")
    top_p = _find_rankings()

    summary = json.loads(summary_p.read_text(encoding="utf-8"))
    trades = pd.read_csv(trade_p)
    events = pd.read_csv(event_p)
    equity = pd.read_csv(equity_p)
    top = pd.read_csv(top_p)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8-sig")) or {}
    release = cfg.get("project", {}).get("release", summary.get("release", "UNKNOWN"))

    for c in ["entry_date", "exit_date"]:
        trades[c] = pd.to_datetime(trades[c], errors="raise")
    events["date"] = pd.to_datetime(events["date"], errors="raise")
    equity["date"] = pd.to_datetime(equity["date"], errors="raise")
    top["ranking_date"] = pd.to_datetime(top["ranking_date"], errors="raise")
    top["effective_month"] = top["effective_month"].astype(str)

    entries = events[events.event.eq("ENTRY")].copy()
    exits = events[events.event.eq("EXIT")].copy()
    signals = events[events.event.eq("SIGNAL")].copy()

    ranked_trades = _attach_signal_and_rank(trades, signals, top)
    unmatched_signal = int(ranked_trades["signal_date"].isna().sum())
    unmatched_rank = int(ranked_trades["rank"].isna().sum())

    checks = {
        "closed_trades": int(len(trades)),
        "entry_events": int(len(entries)),
        "exit_events": int(len(exits)),
        "signal_events": int(len(signals)),
        "trade_entry_count_match": bool(len(trades) == len(entries)),
        "trade_exit_count_match": bool(len(trades) == len(exits)),
        "duplicate_trade_entry_keys": int(trades.duplicated(["permno", "entry_date"]).sum()),
        "trades_without_preceding_signal": unmatched_signal,
        "trades_without_rank_match": unmatched_rank,
        "rank_match_rate": float(1.0 - unmatched_rank / len(trades)) if len(trades) else 1.0,
    }

    rank_rows = []
    for bucket, g in ranked_trades.groupby("rank_bucket", dropna=False):
        rank_rows.append({
            "rank_bucket": bucket,
            "trades": int(len(g)),
            "win_rate": float((g.net_return > 0).mean()) if len(g) else np.nan,
            "avg_net_return": float(g.net_return.mean()) if len(g) else np.nan,
            "median_net_return": float(g.net_return.median()) if len(g) else np.nan,
            "profit_factor": profit_factor(g.net_return),
        })
    rank_df = pd.DataFrame(rank_rows).sort_values("rank_bucket")
    yearly_df = yearly_returns(equity)

    trading_days = list(pd.to_datetime(equity.date).sort_values())
    entry_exact = set(zip(entries.permno.astype(int), entries.date))
    rankmap = top[["effective_month", "permno", "rank", "sector"]].drop_duplicates(["effective_month", "permno"])
    base_w = float(cfg.get("portfolio", {}).get("base_position_weight", 0.10))
    sector_cap = float(cfg.get("portfolio", {}).get("max_sector_weight", 0.30))
    max_pos = int(cfg.get("portfolio", {}).get("max_active_positions", 8))
    max_new = int(cfg.get("entry", {}).get("max_new_entries_per_day", 2))

    orphan_rows = []
    for _, s in signals.iterrows():
        intended = _intended_next_day(s.date, trading_days)
        if pd.notna(intended) and (int(s.permno), intended) in entry_exact:
            continue
        effective_month = s.date.to_period("M").strftime("%Y-%m")
        meta = rankmap[(rankmap.effective_month == effective_month) & (rankmap.permno == s.permno)]
        sector = meta.sector.iloc[0] if len(meta) else None
        active = trades[(trades.entry_date <= intended) & (trades.exit_date >= intended)] if pd.notna(intended) else trades.iloc[0:0]
        same_sector = active[active.sector == sector] if sector is not None else active.iloc[0:0]
        entries_that_day = entries[entries.date == intended] if pd.notna(intended) else entries.iloc[0:0]
        reason = "UNEXPLAINED_REQUIRES_ENGINE_REASON"
        confidence = "LOW"
        orphan_status = "CLOSED_UNENTERED"
        if pd.isna(intended):
            # A signal on the final observed trading day has not yet had its next-close
            # execution opportunity.  It is pending evidence, not an unexplained rejection.
            reason, confidence, orphan_status = "PENDING_NEXT_TRADING_DAY_NOT_YET_OBSERVED", "HIGH", "PENDING"
        elif len(active) >= max_pos:
            reason, confidence = "PROBABLE_MAX_ACTIVE_POSITIONS", "MEDIUM"
        elif len(entries_that_day) >= max_new:
            reason, confidence = "PROBABLE_MAX_NEW_ENTRIES", "MEDIUM"
        elif sector is not None and (len(same_sector) + 1) * base_w > sector_cap + 1e-12:
            reason, confidence = "PROBABLE_SECTOR_CAP", "HIGH"
        orphan_rows.append({
            "signal_date": s.date.date().isoformat(),
            "intended_entry_date": intended.date().isoformat() if pd.notna(intended) else None,
            "ticker": s.ticker,
            "permno": int(s.permno),
            "sector": sector,
            "probable_reason": reason,
            "inference_confidence": confidence,
            "orphan_status": orphan_status,
            "active_positions": int(len(active)),
            "same_sector_positions": int(len(same_sector)),
            "entries_on_intended_day": int(len(entries_that_day)),
        })
    orphan_df = pd.DataFrame(orphan_rows)

    stops = trades[trades.exit_reason.eq("STOP")]
    stop_threshold = float(cfg.get("exit", {}).get("initial_stop_pct", 0.025))
    stop_diag = {
        "semantics": "daily exit thresholds are evaluated on the current security-session close and filled at that same close; execution.mode=next_close applies to signal-driven entry timing",
        "stop_trades": int(len(stops)),
        "configured_stop_pct": stop_threshold,
        "average_realized_stop_return": float(stops.net_return.mean()) if len(stops) else None,
        "median_realized_stop_return": float(stops.net_return.median()) if len(stops) else None,
        "stop_losses_worse_than_5pct": int((stops.net_return < -0.05).sum()),
        "stop_losses_worse_than_10pct": int((stops.net_return < -0.10).sum()),
        "worst_realized_stop_return": float(stops.net_return.min()) if len(stops) else None,
    }

    util = {
        "avg_gross_exposure": float(pd.to_numeric(equity.gross_exposure).mean()),
        "max_gross_exposure": float(pd.to_numeric(equity.gross_exposure).max()),
        "avg_positions": float(pd.to_numeric(equity.positions).mean()),
        "max_positions": int(pd.to_numeric(equity.positions).max()),
        "zero_position_days_pct": float((pd.to_numeric(equity.positions) == 0).mean()),
    }

    profits = trades.net_return[trades.net_return > 0].sort_values(ascending=False)
    profit_sum = float(profits.sum())
    concentration = {
        "top1_share_of_positive_trade_returns": float(profits.iloc[:1].sum() / profit_sum) if profit_sum else None,
        "top2_share_of_positive_trade_returns": float(profits.iloc[:2].sum() / profit_sum) if profit_sum else None,
        "top5_share_of_positive_trade_returns": float(profits.iloc[:5].sum() / profit_sum) if profit_sum else None,
        "top10_share_of_positive_trade_returns": float(profits.iloc[:10].sum() / profit_sum) if profit_sum else None,
    }

    canonical = metrics_from_equity(equity)
    reported_cagr = summary.get("strategy_metrics", {}).get("cagr_backtester_original", summary.get("strategy_metrics", {}).get("cagr"))
    unexplained = int((orphan_df.get("probable_reason", pd.Series(dtype=str)) == "UNEXPLAINED_REQUIRES_ENGINE_REASON").sum()) if len(orphan_df) else 0
    pending = int((orphan_df.get("orphan_status", pd.Series(dtype=str)) == "PENDING").sum()) if len(orphan_df) else 0

    hard_integrity_pass = bool(
        checks["trade_entry_count_match"] and
        checks["trade_exit_count_match"] and
        checks["duplicate_trade_entry_keys"] == 0 and
        checks["trades_without_preceding_signal"] == 0 and
        checks["trades_without_rank_match"] == 0
    )

    # Native backtester metadata is preserved separately; canonical release comes from active config.
    native_reporting = {"raw_release": None, "config_snapshot_release": None, "release_matches_snapshot": None}
    raw_candidate = summary_p.parent / "summary.backtester.json"
    if raw_candidate.exists():
        rawj = json.loads(raw_candidate.read_text(encoding="utf-8"))
        native_reporting["raw_release"] = rawj.get("release")
        native_reporting["config_snapshot_release"] = rawj.get("config_snapshot", {}).get("project", {}).get("release")
        native_reporting["release_matches_snapshot"] = native_reporting["raw_release"] == native_reporting["config_snapshot_release"]

    audit = {
        "release": release,
        "hard_integrity_pass": hard_integrity_pass,
        "metric_convention": "CAGR uses calendar elapsed time; Sharpe/volatility use 252 trading days.",
        "integrity_checks": checks,
        "utilization": util,
        "stop_execution_diagnostics": stop_diag,
        "winner_concentration": concentration,
        "full_period_metrics": canonical,
        "backtester_original_cagr": reported_cagr,
        "2019_2021_metrics": period_metrics(equity, end="2021-12-31"),
        "2022_2025_metrics": period_metrics(equity, start="2022-01-01"),
        "orphan_signal_count": int(len(orphan_df)),
        "pending_signal_count": pending,
        "unexplained_orphan_signal_count": unexplained,
        "native_backtester_reporting": native_reporting,
        "note": "Closed unentered signal reasons remain post-run inference until the core engine emits explicit rejection events. Signals on the final observed trading day are classified as pending, not unexplained. Raw backtester metadata is provenance only; canonical release comes from config.yaml.",
    }

    out_dir = summary_p.parent
    (out_dir / "audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    rank_df.to_csv(out_dir / "rank_bucket_performance.csv", index=False)
    yearly_df.to_csv(out_dir / "yearly_performance.csv", index=False)
    orphan_df.to_csv(out_dir / "orphan_signals.csv", index=False)
    ranked_trades.to_csv(out_dir / "trade_rank_attribution.csv", index=False)

    print("=" * 78)
    print("F2.2 PRODUCTION INTEGRITY + STRATEGY AUDIT")
    print("=" * 78)
    print(f"Release:                      {release}")
    print(f"Hard integrity gate:          {'PASS' if hard_integrity_pass else 'FAIL'}")
    print(f"Trades / ENTRY / EXIT / SIGNAL: {len(trades)} / {len(entries)} / {len(exits)} / {len(signals)}")
    print(f"Rank match rate:              {checks['rank_match_rate']:.2%}")
    print(f"Average gross exposure:       {util['avg_gross_exposure']:.2%}")
    print(f"Zero-position days:           {util['zero_position_days_pct']:.2%}")
    print(f"Unentered/pending signals:     {len(orphan_df)} ({pending} pending, {unexplained} unexplained)")
    print(f"Calendar-time CAGR:           {canonical['cagr']:.2%}")
    print(f"STOP avg realized:            {stop_diag['average_realized_stop_return']:.2%}" if stop_diag['average_realized_stop_return'] is not None else "STOP avg realized: n/a")
    print("-" * 78)
    print(f"Audit report:                 {out_dir / 'audit_report.json'}")
    print(f"Trade rank attribution:       {out_dir / 'trade_rank_attribution.csv'}")
    print(f"Rank buckets:                 {out_dir / 'rank_bucket_performance.csv'}")
    print(f"Yearly performance:           {out_dir / 'yearly_performance.csv'}")
    print(f"Orphan signals:               {out_dir / 'orphan_signals.csv'}")
    print("=" * 78)
    if not hard_integrity_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
