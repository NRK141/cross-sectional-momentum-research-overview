from __future__ import annotations

import json

from .identity import ROOT, LIVE_BACKTEST, RELEASE, load_spec, validate_active_identity

OUT = ROOT / "outputs" / "diagnostics" / "foundation2_2"
REPORT = OUT / "production_health.json"


def main() -> None:
    ident = validate_active_identity()
    if not ident.get("pass"):
        raise RuntimeError(f"F2.2 active identity failed: {ident}")
    summary = json.loads((LIVE_BACKTEST / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((LIVE_BACKTEST / "audit_report.json").read_text(encoding="utf-8"))
    spec = load_spec()
    sm = summary.get("strategy_metrics", {})
    td = summary.get("trade_diagnostics", {})
    util = audit.get("utilization", {})
    stop = audit.get("stop_execution_diagnostics", {})
    warnings: list[str] = []
    if float(util.get("zero_position_days_pct", 0) or 0) > 0.5:
        warnings.append("Portfolio is flat on more than half of trading days; this is informational and does not authorize parameter changes.")
    payload = {
        "release": RELEASE,
        "status": "PASS" if audit.get("hard_integrity_pass") is True else "FAIL",
        "metrics": sm,
        "trade_diagnostics": td,
        "utilization": util,
        "stop_execution": stop,
        "execution_semantics": spec.get("execution_semantics"),
        "warnings": warnings,
        "production_policy": "Foundation 2.2 is the sole supported production strategy; this health report never mutates parameters.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("=" * 78)
    print("FOUNDATION 2.2 PRODUCTION HEALTH")
    print("=" * 78)
    print(f"Status: {payload['status']}")
    print(f"CAGR: {sm.get('cagr')}")
    print(f"Sharpe: {sm.get('sharpe')}")
    print(f"MaxDD: {sm.get('max_drawdown')}")
    print(f"Trades: {td.get('closed_trades')}")
    print(f"PF: {td.get('profit_factor')}")
    print(f"Report: {REPORT}")
    print("=" * 78)
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
