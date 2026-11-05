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

PATCH = "2.4.1"


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
        forward,
        metrics,
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
    signal_source_counts = {}
    if len(entry_forecasts) and "signal_source" in entry_forecasts.columns:
        selected = entry_forecasts[entry_forecasts.get("selected_for_signal", False).astype(bool)]
        signal_source_counts = {str(k): int(v) for k, v in selected.groupby("signal_source").size().to_dict().items()}

    summary = {
        "patch": PATCH,
        "research_mode": mode,
        "run_id": run_id,
        "system": "Standalone Heston-Merton persistent-drift strategy v5",
        "role": "RESEARCH_ONLY_INDEPENDENT_STRATEGY",
        "evaluation_window": {
            "history_start": str(window["history_start"]),
            "score_start": str(window["score_start"]),
            "score_end": str(window.get("score_end")) if window.get("score_end") else None,
            "min_loaded_date": meta.get("min_loaded_date"),
            "max_loaded_date": meta.get("max_loaded_date"),
        },
        "production_identity": identity,
        "portfolio_metrics": pm,
        "trade_metrics": tm,
        "risk_diagnostics": risk_diag,
        "trade_concentration": concentration,
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
        "selection_lift_summary": {
            "months": int(len(selection_lift)),
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
            "forecast_monthly_ic.csv": ic_detail,
            "forecast_ic_summary.csv": ic_summary,
            "forecast_selection_lift.csv": selection_lift,
            "forecast_path_risk_diagnostics.csv": path_risk_diag,
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
