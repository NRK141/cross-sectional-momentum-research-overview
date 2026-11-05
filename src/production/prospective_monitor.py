from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from .identity import ROOT, LIVE_BACKTEST, RELEASE, load_spec

OUT = ROOT / "outputs" / "diagnostics" / "foundation2_2"
REPORT = OUT / "prospective_monitor.json"


def _equity_col(df: pd.DataFrame) -> str:
    for c in ["equity", "portfolio_value", "nav", "value"]:
        if c in df.columns:
            return c
    nums = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
    if not nums:
        raise RuntimeError("No equity column found")
    return nums[-1]


def _metrics(x: pd.DataFrame) -> dict[str, Any]:
    if len(x) < 2:
        return {"status": "NOT_READY", "trading_days": len(x)}
    c = _equity_col(x)
    v = pd.to_numeric(x[c], errors="coerce").astype(float)
    r = v.pct_change().dropna()
    days = len(x)
    total = float(v.iloc[-1] / v.iloc[0] - 1)
    dd = float((v / v.cummax() - 1).min())
    years = (pd.Timestamp(x.date.iloc[-1]) - pd.Timestamp(x.date.iloc[0])).days / 365.25
    cagr = float((v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1) if years > 0 else None
    sh = float(r.mean() / r.std(ddof=1) * math.sqrt(252)) if len(r) > 1 and r.std(ddof=1) > 0 else None
    return {
        "status": "PASS",
        "trading_days": days,
        "total_return": total,
        "cagr": cagr if days >= 20 else None,
        "sharpe": sh if days >= 20 else None,
        "max_drawdown": dd,
        "cagr_sharpe_suppressed": days < 20,
    }


def main() -> None:
    spec = load_spec()
    cutoff = pd.Timestamp(spec["decision_data_cutoff"])
    eq = pd.read_csv(LIVE_BACKTEST / "equity_curve.csv")
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date")
    tr = pd.read_csv(LIVE_BACKTEST / "trade_log.csv")
    tr["exit_date"] = pd.to_datetime(tr["exit_date"])

    baseline = eq[(eq.date >= pd.Timestamp("2019-01-01")) & (eq.date <= cutoff)]
    prospective = eq[eq.date > cutoff]
    windows = {
        "frozen_baseline_through_decision_cutoff": _metrics(baseline),
        "post_decision_prospective": _metrics(prospective),
    }
    for name, df in [("frozen_baseline_through_decision_cutoff", baseline), ("post_decision_prospective", prospective)]:
        if len(df):
            windows[name]["closed_trades"] = int(((tr.exit_date >= df.date.min()) & (tr.exit_date <= df.date.max())).sum())
        else:
            windows[name]["closed_trades"] = 0

    payload = {
        "schema_version": 1,
        "release": RELEASE,
        "decision_data_cutoff": str(cutoff.date()),
        "windows": windows,
        "guardrail": "Only dates after the final production decision cutoff count as new prospective evidence. No monitoring result changes F2.2 parameters automatically.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print("=" * 78)
    print("FOUNDATION 2.2 PRODUCTION PROSPECTIVE MONITOR")
    print("=" * 78)
    for k, v in windows.items():
        print(f"{k}: {v['status']} days={v.get('trading_days')} trades={v.get('closed_trades')} CAGR={v.get('cagr')} Sharpe={v.get('sharpe')} MaxDD={v.get('max_drawdown')}")
    print(f"Report: {REPORT}")
    print("=" * 78)


if __name__ == "__main__":
    main()
