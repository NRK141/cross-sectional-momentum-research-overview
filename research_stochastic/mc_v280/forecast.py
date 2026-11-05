from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .config import HestonMertonParams, MonteCarloConfig
from .payoff import lower_expected_shortfall, stopped_trade_payoff
from .qe_heston_merton import simulate_daily_close_paths_from_uniforms
from .rqmc import sobol_uniforms
from .spec import ForecastSpec


@dataclass(frozen=True)
class ForecastResult:
    mc_trade_return_mean: float
    mc_trade_return_median: float
    mc_trade_return_sd: float
    mc_mean_se: float
    mc_p_profit: float
    mc_p_stop: float
    mc_p_stop_se: float
    mc_p_loss_5: float
    mc_p_loss_8: float
    mc_p_loss_10: float
    mc_q05: float
    mc_q10: float
    mc_q25: float
    mc_es05: float
    mc_es10: float
    mc_paths_used: int
    mc_scrambles: int
    mc_power: int
    mc_converged: bool
    martingale_fallback_rate: float
    forecast_spec_hash: str
    simulation_version: str
    payoff_mode: str
    scramble_means: tuple[float, ...]

    def to_record(self, include_scramble_means: bool = False) -> dict:
        d = asdict(self)
        if not include_scramble_means:
            d.pop("scramble_means", None)
        return d

    def to_legacy_record(self, params: HestonMertonParams, cfg: MonteCarloConfig, *, include_scramble_means: bool = False) -> dict:
        """Compatibility aliases used by the pre-2.8 research engine.

        The old field names are retained so reporting/optimizer code can continue to
        operate while the underlying statistic is now the execution-aligned stopped
        trade distribution for entry forecasts.
        """
        d = self.to_record(include_scramble_means=include_scramble_means)
        # Preserve the pre-2.8 forecast-record contract: downstream research
        # code expects model/calibration fields and forecast statistics to coexist
        # in the same mapping.  The optimizer panel calls the risk allocator before
        # it separately appends Calibration.to_dict(), so omitting theta/mu here
        # causes a live KeyError even though the stochastic forecast itself is valid.
        d.update({
            "mu": float(params.mu),
            "v0": float(params.v0),
            "theta": float(params.theta),
            "kappa": float(params.kappa),
            "xi": float(params.xi),
            "rho": float(params.rho),
            "lambda_jump": float(params.jump_lambda),
            "jump_mu_log": float(params.jump_mu),
            "jump_sigma_log": float(params.jump_sigma),
            "expected_return": self.mc_trade_return_mean,
            "median_return": self.mc_trade_return_median,
            "probability_positive": self.mc_p_profit,
            "probability_loss_5": self.mc_p_loss_5,
            "probability_loss_8": self.mc_p_loss_8,
            "probability_loss_10": self.mc_p_loss_10,
            "expected_shortfall_5": self.mc_es05,
            "terminal_volatility": self.mc_trade_return_sd,
            "jump_probability_horizon": float(1.0 - np.exp(-params.jump_lambda * cfg.horizon_days / cfg.trading_days_per_year)),
        })
        return d


def _summarize(scramble_returns: list[np.ndarray], scramble_stops: list[np.ndarray], fallback_rates: list[float]) -> dict:
    means = np.array([float(np.mean(r)) for r in scramble_returns], dtype=float)
    pooled = np.concatenate(scramble_returns)
    pooled_stop = np.concatenate(scramble_stops)
    stop_means = np.array([float(np.mean(s)) for s in scramble_stops], dtype=float)
    mean = float(np.mean(means))
    se = float(np.std(means, ddof=1) / np.sqrt(len(means)))
    stop_se = float(np.std(stop_means, ddof=1) / np.sqrt(len(stop_means))) if len(stop_means) > 1 else 0.0
    return {
        "mean": mean,
        "se": se,
        "median": float(np.median(pooled)),
        "sd": float(np.std(pooled, ddof=1)),
        "p_profit": float(np.mean(pooled > 0.0)),
        "p_stop": float(np.mean(pooled_stop)),
        "p_stop_se": stop_se,
        "p_loss_5": float(np.mean(pooled <= -0.05)),
        "p_loss_8": float(np.mean(pooled <= -0.08)),
        "p_loss_10": float(np.mean(pooled <= -0.10)),
        "q05": float(np.quantile(pooled, 0.05)),
        "q10": float(np.quantile(pooled, 0.10)),
        "q25": float(np.quantile(pooled, 0.25)),
        "es05": lower_expected_shortfall(pooled, 0.05),
        "es10": lower_expected_shortfall(pooled, 0.10),
        "scramble_means": tuple(float(x) for x in means),
        "fallback_rate": float(np.mean(fallback_rates)),
    }


def _tolerance(cfg: MonteCarloConfig, mean: float) -> float:
    return max(cfg.target_abs_se, cfg.target_rel_se * max(abs(mean), cfg.relative_floor))


def _result(params: HestonMertonParams, cfg: MonteCarloConfig, power: int, s: dict, converged: bool, payoff_mode: str) -> ForecastResult:
    spec = ForecastSpec.from_config(cfg, payoff_mode=payoff_mode)
    n_per_scramble = 1 << power
    return ForecastResult(
        mc_trade_return_mean=s["mean"],
        mc_trade_return_median=s["median"],
        mc_trade_return_sd=s["sd"],
        mc_mean_se=s["se"],
        mc_p_profit=s["p_profit"],
        mc_p_stop=s["p_stop"],
        mc_p_stop_se=s["p_stop_se"],
        mc_p_loss_5=s["p_loss_5"],
        mc_p_loss_8=s["p_loss_8"],
        mc_p_loss_10=s["p_loss_10"],
        mc_q05=s["q05"],
        mc_q10=s["q10"],
        mc_q25=s["q25"],
        mc_es05=s["es05"],
        mc_es10=s["es10"],
        mc_paths_used=n_per_scramble * cfg.n_scrambles,
        mc_scrambles=cfg.n_scrambles,
        mc_power=power,
        mc_converged=converged,
        martingale_fallback_rate=s["fallback_rate"],
        forecast_spec_hash=spec.digest(),
        simulation_version=spec.patch_version,
        payoff_mode=payoff_mode,
        scramble_means=s["scramble_means"],
    )


def forecast_trade_distribution(
    params: HestonMertonParams,
    cfg: MonteCarloConfig,
    *,
    crn_context_key: str,
) -> ForecastResult:
    """Adaptive randomized-QMC execution-aligned entry forecast for one security/date."""
    return forecast_trade_distributions_for_stops(
        params,
        cfg,
        stop_pcts=[cfg.stop_pct],
        crn_context_key=crn_context_key,
    )[float(cfg.stop_pct)]


def forecast_trade_distributions_for_stops(
    params: HestonMertonParams,
    cfg: MonteCarloConfig,
    *,
    stop_pcts: list[float] | tuple[float, ...],
    crn_context_key: str,
) -> dict[float, ForecastResult]:
    """Evaluate several close-stop levels from the SAME simulated paths.

    This is the optimizer-panel workhorse.  It avoids rerunning Heston-Merton for
    each 2/2.5/3% stop while keeping every stop configuration aligned to the exact
    same RQMC scenarios.
    """
    params.validate()
    cfg.validate()
    stops = tuple(sorted({float(x) for x in stop_pcts}))
    if not stops or any(not (0.0 < x < 1.0) for x in stops):
        raise ValueError("stop_pcts must contain values in (0,1)")

    chosen: tuple[int, dict[float, dict]] | None = None
    all_converged = False
    for power in range(cfg.min_power, cfg.max_power + 1):
        by_stop_returns: dict[float, list[np.ndarray]] = {s: [] for s in stops}
        by_stop_flags: dict[float, list[np.ndarray]] = {s: [] for s in stops}
        fallback_rates: list[float] = []

        for sid in range(cfg.n_scrambles):
            u, _design = sobol_uniforms(
                power=power,
                dimension=4 * cfg.n_substeps,
                base_seed=cfg.base_seed,
                context_key=crn_context_key,
                scramble_id=sid,
                clip_eps=cfg.random_clip_eps,
            )
            paths, diag = simulate_daily_close_paths_from_uniforms(params, cfg, u)
            fallback_rates.append(diag.martingale_fallback_rate)
            for stop in stops:
                payoff = stopped_trade_payoff(paths, stop)
                by_stop_returns[stop].append(payoff.returns)
                by_stop_flags[stop].append(payoff.stopped)

        summaries = {
            stop: _summarize(by_stop_returns[stop], by_stop_flags[stop], fallback_rates)
            for stop in stops
        }
        chosen = (power, summaries)
        all_converged = all(s["se"] <= _tolerance(cfg, s["mean"]) for s in summaries.values())
        if all_converged:
            break

    assert chosen is not None
    power, summaries = chosen
    out: dict[float, ForecastResult] = {}
    for stop, s in summaries.items():
        stop_cfg = replace(cfg, stop_pct=stop)
        out[stop] = _result(params, stop_cfg, power, s, all_converged, "entry_stopped_trade")
    return out


def forecast_terminal_distribution(
    params: HestonMertonParams,
    cfg: MonteCarloConfig,
    *,
    crn_context_key: str,
) -> ForecastResult:
    """Adaptive RQMC mark-to-horizon distribution for an ALREADY HELD position.

    Rechecks are not new entries: t=0 is the current close and the statistic is
    final/current - 1 over the remaining horizon.  This deliberately avoids the
    next-close-entry transformation used by entry forecasts.
    """
    params.validate()
    cfg.validate()
    chosen = None
    converged = False
    for power in range(cfg.min_power, cfg.max_power + 1):
        returns: list[np.ndarray] = []
        stops: list[np.ndarray] = []
        fallback_rates: list[float] = []
        for sid in range(cfg.n_scrambles):
            u, _ = sobol_uniforms(
                power=power,
                dimension=4 * cfg.n_substeps,
                base_seed=cfg.base_seed,
                context_key=crn_context_key,
                scramble_id=sid,
                clip_eps=cfg.random_clip_eps,
            )
            paths, diag = simulate_daily_close_paths_from_uniforms(params, cfg, u)
            r = paths[:, -1] / params.spot - 1.0
            returns.append(r)
            stops.append(np.zeros(len(r), dtype=bool))
            fallback_rates.append(diag.martingale_fallback_rate)
        s = _summarize(returns, stops, fallback_rates)
        chosen = (power, s)
        converged = s["se"] <= _tolerance(cfg, s["mean"])
        if converged:
            break
    assert chosen is not None
    power, s = chosen
    return _result(params, cfg, power, s, converged, "held_position_terminal")


def forecast_parameter_ensemble(
    parameter_draws: list[HestonMertonParams],
    cfg: MonteCarloConfig,
    *,
    crn_context_key: str,
) -> dict:
    if len(parameter_draws) < 2:
        raise ValueError("parameter_draws must contain at least 2 draws")
    results = [forecast_trade_distribution(p, cfg, crn_context_key=crn_context_key) for p in parameter_draws]
    means = np.array([r.mc_trade_return_mean for r in results], dtype=float)
    numerical_var = float(np.mean([r.mc_mean_se**2 for r in results]))
    parameter_var = float(np.var(means, ddof=1))
    total_mean_se = float(np.sqrt(parameter_var / len(results) + numerical_var / len(results)))
    return {
        "ensemble_mean": float(np.mean(means)),
        "parameter_sd_of_conditional_mean": float(np.sqrt(parameter_var)),
        "avg_numerical_se": float(np.sqrt(numerical_var)),
        "ensemble_mean_se": total_mean_se,
        "n_parameter_draws": len(results),
    }


def refine_pair_until_resolved(
    params_a: HestonMertonParams,
    params_b: HestonMertonParams,
    cfg: MonteCarloConfig,
    *,
    crn_context_key: str,
    indifference_zone: float = 0.0,
    alpha: float = 0.05,
) -> dict:
    from .ranking import paired_difference

    last = None
    for power in range(cfg.min_power, cfg.max_power + 1):
        fixed = replace(cfg, min_power=power, max_power=power)
        a = forecast_trade_distribution(params_a, fixed, crn_context_key=crn_context_key)
        b = forecast_trade_distribution(params_b, fixed, crn_context_key=crn_context_key)
        diff = paired_difference(
            np.asarray(a.scramble_means),
            np.asarray(b.scramble_means),
            alpha=alpha,
            indifference_zone=indifference_zone,
        )
        last = {
            "a": a,
            "b": b,
            "power": power,
            "paired_difference": diff,
            "rank_resolved": diff.resolved,
        }
        if diff.resolved:
            break
    assert last is not None
    return last
