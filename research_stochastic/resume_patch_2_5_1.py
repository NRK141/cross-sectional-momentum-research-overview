from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data_access import load_config, load_spec, verify_frozen_production_identity
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
)
from research_protocol import verify_holdout_freeze
from run_stochastic_standalone import (
    _checkpoint_dir,
    _entry_forecast_types,
    _publish_output_bundle,
    _window_for_mode,
)

PATCH = "2.5.1"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"checkpoint is incomplete; missing {path.name}")
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("development", "holdout"), default="development")
    args = ap.parse_args()
    mode = str(args.mode)

    identity = verify_frozen_production_identity(ROOT)
    cfg = load_config(ROOT)
    spec = load_spec(ROOT)
    if str(spec.get("patch")) != PATCH:
        raise RuntimeError(f"live spec is {spec.get('patch')!r}; expected {PATCH}")
    if mode == "holdout":
        verify_holdout_freeze(ROOT, spec)
    s = spec["standalone"]
    window = _window_for_mode(spec, mode)
    cp = _checkpoint_dir(mode)
    meta_path = cp / "CHECKPOINT.json"
    if not meta_path.exists():
        raise RuntimeError(f"No Patch {PATCH} {mode} simulation checkpoint exists at {cp}.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("patch") != PATCH or meta.get("research_mode") != mode:
        raise RuntimeError(f"checkpoint identity mismatch: {meta}")
    run_id = str(meta.get("run_id"))
    if not run_id or run_id == "None":
        raise RuntimeError("checkpoint run_id is missing")

    print("=" * 112)
    print(f"PATCH {PATCH} - RESUME {mode.upper()} DIAGNOSTICS FROM SAVED SIMULATION")
    print(f"[CHECKPOINT] {cp.resolve()}")
    print("No feature-store load or Monte Carlo strategy rerun will be performed.")
    print("=" * 112)

    eq = _load_csv(cp / "equity_curve.csv")
    trades = _load_csv(cp / "trade_log.csv")
    events = _load_csv(cp / "event_log.csv")
    forecasts = _load_csv(cp / "model_forecasts.csv")
    forward = _load_csv(cp / "forecast_forward_outcomes.csv")

    entry_forecasts = (
        forecasts[forecasts["forecast_type"].isin(_entry_forecast_types())].copy()
        if len(forecasts)
        else pd.DataFrame()
    )
    diag_cfg = spec.get("diagnostics", {})
    metrics = list(diag_cfg.get("forecast_metrics", []))
    qdiag = realized_quantile_diagnostics(
        forward,
        metrics,
        realized_col="realized_forward_return",
        buckets=int(diag_cfg.get("quantile_buckets", 5)),
    )
    ic_detail, ic_summary = cross_sectional_ic(
        forward, metrics, date_col="date", realized_col="realized_forward_return",
        min_names=int(diag_cfg.get("ic_min_names", 20)),
    )
    ic_by_source, ic_summary_by_source = cross_sectional_ic_by_source(
        forward, metrics, source_col="signal_source", date_col="date",
        realized_col="realized_forward_return", min_names=int(diag_cfg.get("ic_min_names", 20)),
    )
    nonoverlap_forward = non_overlapping_forecast_frame(forward)
    ic_nonoverlap, ic_nonoverlap_summary = cross_sectional_ic_by_source(
        nonoverlap_forward, metrics, source_col="signal_source", date_col="date",
        realized_col="realized_forward_return", min_names=int(diag_cfg.get("ic_min_names", 20)),
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
    if len(trades) and "exit_reason" in trades.columns:
        discontinuation_trades = trades[trades["exit_reason"] == "SECURITY_DISCONTINUATION"].copy()
    else:
        discontinuation_trades = pd.DataFrame()
    if len(discontinuation_trades):
        stress_delta = float(pd.to_numeric(
            discontinuation_trades.get("zero_recovery_incremental_loss_usd", pd.Series(dtype=float)), errors="coerce"
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
    if len(entry_forecasts) and "signal_source" in entry_forecasts.columns:
        selected = entry_forecasts[entry_forecasts.get("selected_for_signal", False).astype(bool)]
        signal_source_counts = {str(k): int(v) for k, v in selected.groupby("signal_source").size().to_dict().items()}

    entry_order_audit = pd.DataFrame()
    entry_order_summary = {"submitted": 0, "filled": 0, "cancelled": 0, "unresolved": 0, "cancel_reasons": {}}
    if len(events) and "event" in events.columns and "order_id" in events.columns:
        audit_events = events[events["event"].isin(["SIGNAL", "ENTRY", "ENTRY_ORDER_CANCELLED"])].dropna(subset=["order_id"]).copy()
        rows = []
        for order_id, og in audit_events.groupby("order_id", sort=True):
            sig = og[og["event"] == "SIGNAL"]
            filled = og[og["event"] == "ENTRY"]
            cancelled = og[og["event"] == "ENTRY_ORDER_CANCELLED"]
            status = "FILLED" if len(filled) else ("CANCELLED" if len(cancelled) else "UNRESOLVED")
            base = sig.iloc[0] if len(sig) else og.iloc[0]
            rows.append({
                "order_id": str(order_id), "signal_date": base.get("date"), "ticker": base.get("ticker"),
                "permno": base.get("permno"), "signal_source": base.get("signal_source"),
                "target_weight": base.get("target_weight"), "status": status,
                "execution_date": (filled.iloc[0].get("date") if len(filled) else (cancelled.iloc[0].get("date") if len(cancelled) else pd.NaT)),
                "actual_weight": filled.iloc[0].get("actual_weight") if len(filled) else None,
                "cancel_reason": cancelled.iloc[0].get("reason") if len(cancelled) else None,
            })
        entry_order_audit = pd.DataFrame(rows)
        if len(entry_order_audit):
            entry_order_summary.update({
                "submitted": int(len(entry_order_audit)),
                "filled": int((entry_order_audit["status"] == "FILLED").sum()),
                "cancelled": int((entry_order_audit["status"] == "CANCELLED").sum()),
                "unresolved": int((entry_order_audit["status"] == "UNRESOLVED").sum()),
                "cancel_reasons": {str(k): int(v) for k, v in entry_order_audit.loc[entry_order_audit["status"] == "CANCELLED", "cancel_reason"].value_counts(dropna=False).to_dict().items()},
            })
            if entry_order_summary["unresolved"]:
                raise RuntimeError(f"Recovered entry-order audit has unresolved orders: {entry_order_summary}")

    summary = {
        "patch": PATCH,
        "research_mode": mode,
        "run_id": run_id,
        "system": "Standalone Heston-Merton structural-baseline strategy v6",
        "role": "RESEARCH_ONLY_INDEPENDENT_STRATEGY",
        "evaluation_window": {
            "history_start": str(window["history_start"]),
            "score_start": str(window["score_start"]),
            "score_end": str(window.get("score_end")) if window.get("score_end") else None,
            "min_loaded_date": meta.get("min_loaded_date"),
            "max_loaded_date": meta.get("max_loaded_date"),
        },
        "production_identity": identity,
        "warmup_diagnostics": {
            "history_start": str(window["history_start"]),
            "score_start": str(window["score_start"]),
            "global_sessions_before_score_start": int(meta.get("warmup_global_sessions", 0)),
            "required_max_calibration_observations": int(meta.get("required_warmup_sessions", spec["model"]["max_calibration_observations"])),
            "first_entry_date": (pd.Timestamp(trades["entry_date"].min()).isoformat() if len(trades) else None),
            "first_signal_date": (pd.Timestamp(events.loc[events["event"] == "SIGNAL", "date"].min()).isoformat() if len(events) and (events["event"] == "SIGNAL").any() else None),
        },
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
                pd.to_numeric(forward.get("realized_forward_return", pd.Series(dtype=float)), errors="coerce").notna().sum()
            ) if len(forward) else 0,
            "calibration_cache_entries": int(meta.get("calibration_cache_entries", 0)),
            "forecast_cache_entries": int(meta.get("forecast_cache_entries", 0)),
            "boundary_entries_rejected": int(meta.get("boundary_entries_rejected", 0)),
        },
        "standalone_spec": s,
        "model_spec": spec["model"],
        "diagnostic_ic_summary": ic_summary.to_dict("records") if len(ic_summary) else [],
        "diagnostic_ic_summary_by_source": ic_summary_by_source.to_dict("records") if len(ic_summary_by_source) else [],
        "diagnostic_ic_nonoverlap_summary": ic_nonoverlap_summary.to_dict("records") if len(ic_nonoverlap_summary) else [],
        "entry_order_audit": entry_order_summary,
        "selection_lift_summary": {
            "forecast_dates": int(len(selection_lift)),
            "mean_lift": float(pd.to_numeric(selection_lift.get("selection_lift_vs_other_passing", pd.Series(dtype=float)), errors="coerce").mean()) if len(selection_lift) else None,
            "median_lift": float(pd.to_numeric(selection_lift.get("selection_lift_vs_other_passing", pd.Series(dtype=float)), errors="coerce").median()) if len(selection_lift) else None,
            "positive_lift_fraction": float((pd.to_numeric(selection_lift.get("selection_lift_vs_other_passing", pd.Series(dtype=float)), errors="coerce") > 0).mean()) if len(selection_lift) else None,
        },
        "right_boundary": {"policy": spec["boundary"]["policy"], "entries_rejected": int(meta.get("boundary_entries_rejected", 0))},
        "research_guardrail": (
            "DEVELOPMENT ONLY. No rows after 2024-12-31 were loaded. 2025-present is sealed from further tuning but is not pristine because earlier patches inspected 2025-2026."
            if mode == "development"
            else "SEALED CONFIRMATION WINDOW. Do not tune from these results; earlier patches already inspected part of this period."
        ),
        "recovered_from_checkpoint": True,
    }
    out = ROOT / str(window["output_dir"])
    manifest = {
        "production_identity": identity,
        "spec_patch": PATCH,
        "research_mode": mode,
        "run_id": run_id,
        "history_start": str(window["history_start"]),
        "score_start_date": str(window["score_start"]),
        "score_end_date": str(window.get("score_end")) if window.get("score_end") else None,
        "min_loaded_date": meta.get("min_loaded_date"),
        "max_loaded_date": meta.get("max_loaded_date"),
        "rows_loaded": int(meta.get("rows_loaded", 0)),
        "output_dir": str(out),
        "recovered_from_checkpoint": True,
    }
    run_started = dict(meta.get("run_started") or {})
    run_started.update({"patch": PATCH, "research_mode": mode, "run_id": run_id, "status": "COMPLETED_FROM_CHECKPOINT"})

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
            "forecast_monthly_ic.csv": ic_detail,
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
        {"summary.json": summary, "manifest.json": manifest, "RUN_STARTED.json": run_started},
        patch=PATCH,
        run_id=run_id,
    )
    shutil.rmtree(cp, ignore_errors=True)
    for stale in (out / "RUN_FAILED.json", out / "ERROR_TRACEBACK.txt"):
        if stale.exists():
            stale.unlink()
    print(f"[PASS] Published recovered Patch {PATCH} {mode} bundle: {out.resolve()}")


if __name__ == "__main__":
    main()
