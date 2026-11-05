from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .forward_prefix_lock import verify_lock
from .identity import BASELINE_DIR, LIVE_BACKTEST, LIVE_SCREENER, RELEASE, ROOT, load_spec, validate_active_identity
from .surface import (
    BASELINE_ALLOWED_DIRS,
    DIAGNOSTIC_ALLOWED_DIRS,
    MANIFEST_ALLOWED_FILES,
    OUTPUT_ALLOWED_TOP_DIRS,
    ROOT_ALLOWED_FILES,
    SRC_ALLOWED_FILES,
    TEST_ALLOWED_FILES,
    TRANSIENT_CLEANUP_FILES,
)

OUT = ROOT / "outputs" / "diagnostics" / "foundation2_2" / "state_verification.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_checks(spec: dict[str, Any]) -> dict[str, bool]:
    expected = spec.get("immutable_baseline", {})
    summary_p = BASELINE_DIR / "backtest" / "summary.json"
    trades_p = BASELINE_DIR / "backtest" / "trade_log.csv"
    audit_p = BASELINE_DIR / "backtest" / "audit_report.json"
    rankings_p = BASELINE_DIR / "screener" / "monthly_rankings.csv"
    checks: dict[str, bool] = {
        "baseline_dir_exists": BASELINE_DIR.exists(),
        "baseline_summary_exists": summary_p.exists(),
        "baseline_trade_log_exists": trades_p.exists(),
        "baseline_audit_exists": audit_p.exists(),
        "baseline_rankings_exists": rankings_p.exists(),
    }
    if not all(checks.values()):
        return checks
    summary = _json(summary_p)
    audit = _json(audit_p)
    trades = pd.read_csv(trades_p)
    sm = summary.get("strategy_metrics", {})
    td = summary.get("trade_diagnostics", {})
    reason_col = "exit_reason" if "exit_reason" in trades.columns else None
    stop_count = int(trades[reason_col].astype(str).str.upper().eq("STOP").sum()) if reason_col else -1
    checks.update({
        "baseline_release": summary.get("release") == RELEASE,
        "baseline_closed_trades": int(td.get("closed_trades", -1)) == int(expected.get("closed_trades", -2)),
        "baseline_stop_trades": stop_count == int(expected.get("stop_trades", -2)),
        "baseline_profit_factor": abs(float(td.get("profit_factor", -999)) - float(expected.get("profit_factor", -998))) <= 1e-10,
        "baseline_sharpe": abs(float(sm.get("sharpe", -999)) - float(expected.get("sharpe", -998))) <= 1e-10,
        "baseline_max_drawdown": abs(float(sm.get("max_drawdown", -999)) - float(expected.get("max_drawdown", -998))) <= 1e-10,
        "baseline_hard_integrity": audit.get("hard_integrity_pass") is True,
        "baseline_rank_match_100pct": float(audit.get("integrity_checks", {}).get("rank_match_rate", 0.0)) == 1.0,
        "baseline_unexplained_signals_zero": int(audit.get("unexplained_orphan_signal_count", 0)) == 0,
    })
    return checks


def _project_files(base: Path) -> set[str]:
    if not base.exists():
        return set()
    return {
        p.relative_to(base).as_posix()
        for p in base.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }


def _surface_checks() -> dict[str, bool]:
    root_files = {p.name for p in ROOT.iterdir() if p.is_file()}
    allowed_root = ROOT_ALLOWED_FILES | TRANSIENT_CLEANUP_FILES
    src_files = _project_files(ROOT / "src")
    test_files = _project_files(ROOT / "tests")

    manifests_dir = ROOT / "data" / "manifests"
    manifest_files = {p.name for p in manifests_dir.iterdir() if p.is_file()} if manifests_dir.exists() else set()

    outputs = ROOT / "outputs"
    output_dirs = {p.name for p in outputs.iterdir() if p.is_dir()} if outputs.exists() else set()
    baselines = outputs / "baselines"
    baseline_dirs = {p.name for p in baselines.iterdir() if p.is_dir()} if baselines.exists() else set()
    diagnostics = outputs / "diagnostics"
    diagnostic_dirs = {p.name for p in diagnostics.iterdir() if p.is_dir()} if diagnostics.exists() else set()

    return {
        "surface_root_exact": root_files.issubset(allowed_root) and ROOT_ALLOWED_FILES.issubset(root_files),
        "surface_src_exact": src_files == SRC_ALLOWED_FILES,
        "surface_tests_exact": test_files == TEST_ALLOWED_FILES,
        "surface_manifests_production_only": manifest_files.issubset(MANIFEST_ALLOWED_FILES),
        "surface_outputs_production_only": output_dirs.issubset(OUTPUT_ALLOWED_TOP_DIRS) and {"backtest", "baselines", "screener"}.issubset(output_dirs),
        "surface_baseline_exact": baseline_dirs == BASELINE_ALLOWED_DIRS,
        "surface_diagnostics_production_only": diagnostic_dirs.issubset(DIAGNOSTIC_ALLOWED_DIRS),
        "archive_absent": not (ROOT / "archive").exists(),
        "research_state_absent": not (ROOT / "data" / "research_state").exists(),
        "legacy_monthly_top25_absent": not (LIVE_SCREENER / "monthly_top25.csv").exists(),
        "research_package_absent": not (ROOT / "src" / "research").exists(),
        "maintenance_package_absent": not (ROOT / "src" / "maintenance").exists(),
    }


def main() -> None:
    spec = load_spec()
    ident = validate_active_identity()
    lock = verify_lock()
    live_summary = _json(LIVE_BACKTEST / "summary.json") if (LIVE_BACKTEST / "summary.json").exists() else {}
    native_summary = _json(LIVE_BACKTEST / "summary.backtester.json") if (LIVE_BACKTEST / "summary.backtester.json").exists() else {}
    live_audit = _json(LIVE_BACKTEST / "audit_report.json") if (LIVE_BACKTEST / "audit_report.json").exists() else {}
    screener_manifest_path = ROOT / "data" / "manifests" / "screener_success.json"
    screener_manifest = _json(screener_manifest_path) if screener_manifest_path.exists() else {}

    checks: dict[str, bool] = {
        "active_identity": ident.get("pass", False),
        "sole_strategy_role": spec.get("role") == "SOLE_PRODUCTION_STRATEGY",
        "summary_exists": (LIVE_BACKTEST / "summary.json").exists(),
        "trade_log_exists": (LIVE_BACKTEST / "trade_log.csv").exists(),
        "canonical_summary_release": live_summary.get("release") == RELEASE,
        "native_summary_release": native_summary.get("release") == RELEASE,
        "screener_manifest_release": screener_manifest.get("release") == RELEASE,
        "audit_release": live_audit.get("release") == RELEASE,
        "audit_hard_integrity": live_audit.get("hard_integrity_pass") is True,
        "audit_stop_semantics_same_close": "same close" in str(live_audit.get("stop_execution_diagnostics", {}).get("semantics", "")).lower(),
        "rankings_exists": (LIVE_SCREENER / "monthly_rankings.csv").exists(),
        "forward_prefix_lock": lock.get("status") == "PASS",
    }
    checks.update(_baseline_checks(spec))
    checks.update(_surface_checks())

    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "release": RELEASE,
        "status": status,
        "checks": checks,
        "identity": ident,
        "prefix_lock": lock,
        "immutable_baseline": spec.get("immutable_baseline"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("=" * 78)
    print("FOUNDATION 2.2 FINAL PRODUCTION STATE VERIFICATION")
    print("=" * 78)
    print(f"Status: {status}")
    for k, v in checks.items():
        print(f"{k}: {v}")
    print(f"Report: {OUT}")
    print("=" * 78)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
