from __future__ import annotations

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
from run_stochastic_standalone import _checkpoint_dir, _publish_output_bundle

PATCH = "2.3.5"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"checkpoint is incomplete; missing {path.name}")
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    identity = verify_frozen_production_identity(ROOT)
    cfg = load_config(ROOT)
    spec = load_spec(ROOT)
    if str(spec.get("patch")) != PATCH:
        raise RuntimeError(f"live spec is {spec.get('patch')!r}; expected {PATCH}")
    s = spec["standalone"]
    cp = _checkpoint_dir()
    meta_path = cp / "CHECKPOINT.json"
    if not meta_path.exists():
        raise RuntimeError(
            f"No Patch {PATCH} simulation checkpoint exists at {cp}. "
            "A full strategy run is required."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("patch") != PATCH:
        raise RuntimeError(f"checkpoint is patch {meta.get('patch')!r}; expected {PATCH}")
    run_id = str(meta.get("run_id"))
    if not run_id or run_id == "None":
        raise RuntimeError("checkpoint run_id is missing")

    print("=" * 112)
    print(f"PATCH {PATCH} - RESUME DIAGNOSTICS FROM SAVED SIMULATION CHECKPOINT")
    print(f"[CHECKPOINT] {cp.resolve()}")
    print(f"[RUN ID] {run_id}")
    print("No feature-store load or Monte Carlo strategy rerun will be performed.")
    print("=" * 112)

    eq = _load_csv(cp / "equity_curve.csv")
    trades = _load_csv(cp / "trade_log.csv")
    events = _load_csv(cp / "event_log.csv")
    forecasts = _load_csv(cp / "model_forecasts.csv")
    forward = _load_csv(cp / "forecast_forward_outcomes.csv")

    entry_forecasts = forecasts[forecasts["forecast_type"] == "MONTH_END_ENTRY"].copy() if len(forecasts) else pd.DataFrame()
    diag_cfg = spec.get("diagnostics", {})
    metrics = list(diag_cfg.get("forecast_metrics", []))
    qdiag = realized_quantile_diagnostics(
        forward, metrics, realized_col="realized_forward_return", buckets=int(diag_cfg.get("quantile_buckets", 5))
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

    pm = portfolio_metrics(eq["equity"], int(cfg["execution"]["annualization_days"])) if len(eq) else portfolio_metrics(pd.Series(dtype=float))
    tm = trade_metrics(trades)
    risk_diag = equity_risk_diagnostics(eq)
    concentration = trade_concentration_diagnostics(trades)

    summary = {
        "patch": PATCH,
        "run_id": run_id,
        "system": "Standalone Heston-Merton persistent-drift strategy v4",
        "role": "RESEARCH_ONLY_INDEPENDENT_STRATEGY",
        "production_identity": identity,
        "model_interpretation": "Historical physical-measure Heston-Merton proxy; no option-implied surface is used.",
        "portfolio_metrics": pm,
        "trade_metrics": tm,
        "risk_diagnostics": risk_diag,
        "trade_concentration": concentration,
        "forecast_summary": {
            "entry_forecasts": int(len(entry_forecasts)),
            "entry_forecast_passes": int(entry_forecasts["pass"].sum()) if len(entry_forecasts) else 0,
            "signals_selected": int(entry_forecasts.get("selected_for_signal", pd.Series(dtype=bool)).sum()) if len(entry_forecasts) else 0,
            "forward_outcomes_available": int(pd.to_numeric(forward.get("realized_forward_return", pd.Series(dtype=float)), errors="coerce").notna().sum()) if len(forward) else 0,
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
        "research_guardrail": "2026 is research-visible. Patch 2.3.5 results are development evidence, not an untouched holdout.",
        "recovered_from_checkpoint": True,
    }
    out = ROOT / "outputs" / "research" / "stochastic_standalone_patch_2_3_5"
    manifest = {
        "production_identity": identity,
        "spec_patch": PATCH,
        "run_id": run_id,
        "score_start_date": cfg["data"]["score_start_date"],
        "rows_loaded": int(meta.get("rows_loaded", 0)),
        "output_dir": str(out),
        "retired_shadow": True,
        "publication_policy": "STAGE_COMPLETE_BUNDLE_THEN_PROMOTE_FILES_RUN_COMPLETE_LAST",
        "recovered_from_checkpoint": True,
    }
    run_started = dict(meta.get("run_started") or {})
    run_started.update({"patch": PATCH, "run_id": run_id, "status": "COMPLETED_FROM_CHECKPOINT"})

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
    print(f"[PASS] Published recovered Patch {PATCH} bundle: {out.resolve()}")


if __name__ == "__main__":
    main()
