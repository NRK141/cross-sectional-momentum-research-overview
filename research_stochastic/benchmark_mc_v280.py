from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mc_v280 import HestonMertonParams, MonteCarloConfig, forecast_trade_distribution


def main() -> int:
    p = HestonMertonParams(
        spot=100.0, v0=0.05, kappa=2.0, theta=0.04, xi=0.55, rho=-0.7,
        mu=0.08, jump_lambda=0.25, jump_mu=-0.04, jump_sigma=0.08,
    )
    print("=" * 78)
    print("PATCH 2.8.0 - RQMC PRECISION / COST BENCHMARK")
    print("=" * 78)
    for power in (8, 9, 10, 11, 12):
        cfg = MonteCarloConfig(
            horizon_days=20, substeps_per_day=2, n_scrambles=8,
            min_power=power, max_power=power, stop_pct=0.025,
        )
        t0 = time.perf_counter()
        r = forecast_trade_distribution(p, cfg, crn_context_key="BENCHMARK_DATE")
        sec = time.perf_counter() - t0
        print(
            f"2^{power:2d} x {cfg.n_scrambles:2d} | paths={r.mc_paths_used:7,d} | "
            f"mean={r.mc_trade_return_mean: .5%} | SE={r.mc_mean_se:.5%} | {sec:7.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
