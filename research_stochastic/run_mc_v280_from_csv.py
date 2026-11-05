from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mc_v280 import HestonMertonParams, MonteCarloConfig, forecast_trade_distribution


ALIASES = {
    "date": ["date", "forecast_date", "signal_date"],
    "id": ["PERMNO", "permno", "ticker", "symbol", "security_id"],
    "spot": ["spot", "s0", "price", "close", "signal_close"],
    "v0": ["v0", "variance0", "initial_variance", "heston_v0"],
    "kappa": ["kappa", "heston_kappa"],
    "theta": ["theta", "heston_theta"],
    "xi": ["xi", "sigma_v", "vol_of_vol", "heston_xi"],
    "rho": ["rho", "heston_rho"],
    "mu": ["mu_shrunk", "mu", "drift", "annual_mu"],
    "jump_lambda": ["jump_lambda", "lambda_jump", "jump_intensity", "lambda_j"],
    "jump_mu": ["jump_mu", "mu_jump", "jump_mean", "mu_j"],
    "jump_sigma": ["jump_sigma", "sigma_jump", "jump_std", "sigma_j"],
}


def resolve(df: pd.DataFrame, key: str, required: bool = True) -> str | None:
    for c in ALIASES[key]:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not resolve required column {key!r}; tried {ALIASES[key]}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a Patch 2.8.0 forecast panel from calibrated parameter rows.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="optimizer_forecast_panel_v280.csv")
    ap.add_argument("--stop-pct", type=float, default=0.025)
    ap.add_argument("--horizon-days", type=int, default=20)
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--scrambles", type=int, default=8)
    ap.add_argument("--min-power", type=int, default=8)
    ap.add_argument("--max-power", type=int, default=13)
    ap.add_argument("--drift-convention", choices=["total_arithmetic", "continuous_arithmetic"], default="total_arithmetic")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    cols = {k: resolve(df, k, required=(k not in {"jump_lambda", "jump_mu", "jump_sigma"})) for k in ALIASES}
    cfg = MonteCarloConfig(
        horizon_days=args.horizon_days,
        substeps_per_day=args.substeps,
        n_scrambles=args.scrambles,
        min_power=args.min_power,
        max_power=args.max_power,
        stop_pct=args.stop_pct,
    )

    rows = []
    print("=" * 78)
    print("PATCH 2.8.0 - FORECAST PANEL REBUILD")
    print("=" * 78)
    print("Resolved columns:", json.dumps(cols, indent=2))
    print(f"Rows: {len(df):,}")

    for i, r in df.iterrows():
        date_value = str(r[cols["date"]])
        ident = str(r[cols["id"]])
        p = HestonMertonParams(
            spot=float(r[cols["spot"]]),
            v0=float(r[cols["v0"]]),
            kappa=float(r[cols["kappa"]]),
            theta=float(r[cols["theta"]]),
            xi=float(r[cols["xi"]]),
            rho=float(r[cols["rho"]]),
            mu=float(r[cols["mu"]]),
            jump_lambda=float(r[cols["jump_lambda"]]) if cols["jump_lambda"] else 0.0,
            jump_mu=float(r[cols["jump_mu"]]) if cols["jump_mu"] else 0.0,
            jump_sigma=float(r[cols["jump_sigma"]]) if cols["jump_sigma"] else 0.0,
            drift_convention=args.drift_convention,
        )
        result = forecast_trade_distribution(p, cfg, crn_context_key=date_value)
        record = r.to_dict()
        record.update(result.to_record(include_scramble_means=False))
        record["mc_scramble_means_json"] = json.dumps(result.scramble_means)
        rows.append(record)
        if len(rows) % 25 == 0 or len(rows) == len(df):
            print(f"[{len(rows):,}/{len(df):,}] latest={date_value} {ident} mean={result.mc_trade_return_mean:.5%} se={result.mc_mean_se:.5%} paths={result.mc_paths_used:,}")

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"[DONE] wrote {args.output} ({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
