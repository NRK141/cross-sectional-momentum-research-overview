from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from src.common.metrics import metrics_from_equity

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
LIVE_BACKTEST = OUTPUTS / "backtest"
LIVE_SCREENER = OUTPUTS / "screener"


def _find_latest(name: str) -> Path:
    """Resolve only from the live backtest namespace.

    Immutable baseline archives live under outputs/baselines;
    those must never be treated as the mutable live run.
    """
    candidates = list(LIVE_BACKTEST.rglob(name)) if LIVE_BACKTEST.exists() else []
    if not candidates:
        raise FileNotFoundError(f"Could not find live {name} under {LIVE_BACKTEST}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_screener_source() -> Path:
    """Resolve the canonical live F2.2 ranking artifact only."""
    path = LIVE_SCREENER / "monthly_rankings.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find canonical live monthly rankings: {path}")
    return path


def canonicalize() -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8-sig")) or {}
    release = cfg.get("project", {}).get("release", "UNKNOWN")

    summary_p = _find_latest("summary.json")
    equity_p = _find_latest("equity_curve.csv")
    summary = json.loads(summary_p.read_text(encoding="utf-8"))
    raw_p = summary_p.with_name("summary.backtester.json")
    if not summary.get("_canonicalization", {}).get("canonicalized"):
        shutil.copy2(summary_p, raw_p)
    raw_summary = json.loads(raw_p.read_text(encoding="utf-8")) if raw_p.exists() else summary
    raw_release = raw_summary.get("release")
    raw_snapshot_release = raw_summary.get("config_snapshot", {}).get("project", {}).get("release")
    equity = pd.read_csv(equity_p)
    canonical_metrics = metrics_from_equity(equity)

    strategy_metrics = summary.setdefault("strategy_metrics", {})
    original_cagr = strategy_metrics.get("cagr_backtester_original", strategy_metrics.get("cagr"))
    strategy_metrics["cagr_backtester_original"] = original_cagr
    strategy_metrics["cagr"] = canonical_metrics["cagr"]
    strategy_metrics["cagr_method"] = "calendar_elapsed_365.25"
    strategy_metrics["sharpe"] = canonical_metrics["sharpe"]
    strategy_metrics["annualized_volatility"] = canonical_metrics["annualized_volatility"]
    strategy_metrics["max_drawdown"] = canonical_metrics["max_drawdown"]
    summary["release"] = release
    summary["metric_conventions"] = {
        "cagr": "calendar elapsed time using 365.25 days/year",
        "sharpe": "daily arithmetic return mean/std scaled by sqrt(252), zero risk-free rate",
        "annualized_volatility": "daily return std scaled by sqrt(252)",
        "max_drawdown": "equity / running equity peak - 1",
    }
    summary["_canonicalization"] = {
        "canonicalized": True,
        "source": "src.production.canonicalize_outputs",
        "release_from_config": release,
        "preserved_backtester_summary": "summary.backtester.json",
        "raw_backtester_release": raw_release,
        "raw_config_snapshot_release": raw_snapshot_release,
        "raw_release_matches_config_snapshot": raw_release == raw_snapshot_release,
    }
    summary_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    src = _find_screener_source()
    stable_p = src.parent / "monthly_rankings.csv"
    if src.resolve() != stable_p.resolve():
        shutil.copy2(src, stable_p)
    top = pd.read_csv(stable_p)
    if "effective_month" not in top or "rank" not in top:
        raise ValueError("Screener output must contain effective_month and rank")
    top_n_cfg = int(cfg.get("screener", {}).get("top_n", int(pd.to_numeric(top["rank"]).max())))
    max_rank = int(pd.to_numeric(top["rank"]).max()) if len(top) else 0
    months = int(top["effective_month"].astype(str).nunique()) if len(top) else 0
    counts = top.groupby(top["effective_month"].astype(str)).size() if len(top) else pd.Series(dtype=int)
    min_rows_per_month = int(counts.min()) if len(counts) else 0
    max_rows_per_month = int(counts.max()) if len(counts) else 0
    meta = {
        "release": release,
        "canonical_file": str(stable_p.relative_to(ROOT)),
        "legacy_source_file": str(src.relative_to(ROOT)),
        "configured_top_n": top_n_cfg,
        "observed_max_rank": max_rank,
        "rows": int(len(top)),
        "months": months,
        "min_rows_per_month": min_rows_per_month,
        "max_rows_per_month": max_rows_per_month,
        "all_months_full_top_n": bool(len(counts) and min_rows_per_month == top_n_cfg and max_rows_per_month == top_n_cfg),
        "top_n_matches_output": bool(max_rank <= top_n_cfg),
    }
    (stable_p.parent / "monthly_rankings.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "release": release,
        "summary": str(summary_p),
        "screener": str(stable_p),
        "cagr": canonical_metrics["cagr"],
        "original_backtester_cagr": original_cagr,
        "top_n": top_n_cfg,
        "rows": int(len(top)),
        "months": months,
    }


def main() -> None:
    result = canonicalize()
    print("=" * 78)
    print("F2.2 PRODUCTION OUTPUT CANONICALIZATION")
    print("=" * 78)
    print(f"Release:                     {result['release']}")
    print(f"Canonical CAGR:              {result['cagr']:.4%}")
    if result["original_backtester_cagr"] is not None:
        print(f"Original backtester CAGR:    {float(result['original_backtester_cagr']):.4%}")
    print(f"Stable screener artifact:    {result['screener']}")
    print(f"Configured Top-N:            {result['top_n']}")
    print(f"Screener rows / months:      {result['rows']} / {result['months']}")
    print(f"Canonical summary:           {result['summary']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
