from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable
import hashlib
import json
import math

import numpy as np
import pandas as pd

from mc_v280 import (
    HestonMertonParams,
    MonteCarloConfig,
    empirical_bayes_normal,
    forecast_terminal_distribution as rqmc_terminal_forecast,
    forecast_trade_distribution as rqmc_trade_forecast,
    forecast_trade_distributions_for_stops as rqmc_trade_forecasts_for_stops,
)
from mc_v280.qe_heston_merton import simulate_daily_close_paths_from_uniforms

EPS = 1e-12
TRADING_DAYS = 252


@dataclass(frozen=True)
class Calibration:
    """Historical physical-measure Heston-Merton proxy calibration.

    Patch 2.9.3 keeps the original public fields but adds uncertainty and
    regularization diagnostics.  ``mu`` and the jump parameters may be replaced
    by a same-date cross-sectional empirical-Bayes/reliability regularization;
    the ``*_raw`` fields retain the original per-security calibration.
    """

    n_obs: int
    mu: float
    mu_short: float
    mu_long: float
    drift_agreement: float
    v0: float
    theta: float
    kappa: float
    xi: float
    rho: float
    lambda_jump: float
    jump_mu_log: float
    jump_sigma_log: float
    jump_count: int
    robust_daily_sigma: float
    mu_short_se: float = float("nan")
    mu_long_se: float = float("nan")
    mu_se: float = float("nan")
    mu_raw: float = float("nan")
    lambda_jump_raw: float = float("nan")
    jump_mu_log_raw: float = float("nan")
    jump_sigma_log_raw: float = float("nan")
    mu_shrink_weight: float = 1.0
    jump_shrink_weight: float = 1.0
    mu_prior: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TerminalStats:
    expected_return: float
    median_return: float
    probability_positive: float
    probability_loss_5: float
    probability_loss_8: float
    probability_loss_10: float
    expected_shortfall_5: float
    terminal_volatility: float
    jump_probability_horizon: float

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *[str(x) for x in parts]]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % (2**32 - 1)


def _finite_array(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(values), dtype=float)
    return x[np.isfinite(x)]


def _ewma_mean(x: np.ndarray, half_life: float) -> float:
    if len(x) == 0:
        return 0.0
    half_life = max(float(half_life), 1.0)
    alpha = 1.0 - math.exp(math.log(0.5) / half_life)
    m = float(x[0])
    for val in x[1:]:
        m = (1.0 - alpha) * m + alpha * float(val)
    return m


def _ewma_effective_n(n: int, half_life: float) -> float:
    if n <= 1:
        return 1.0
    alpha = 1.0 - math.exp(math.log(0.5) / max(float(half_life), 1.0))
    asymptotic = (2.0 - alpha) / max(alpha, 1e-12)
    return float(max(1.0, min(float(n), asymptotic)))


def calibrate_heston_merton(
    simple_returns: Iterable[float],
    *,
    min_observations: int = 252,
    max_observations: int = 504,
    variance_window: int = 20,
    jump_sigma_threshold: float = 3.5,
    drift_short_half_life_days: float = 63.0,
    drift_long_half_life_days: float = 252.0,
    drift_short_weight: float = 0.50,
    drift_shrinkage: float = 0.25,
    max_abs_annual_drift: float = 0.75,
) -> Calibration | None:
    """Calibrate a trailing physical-measure Heston-Merton proxy.

    Drift uncertainty is explicitly estimated because Patch 2.9.3 no longer
    treats a noisy sign estimate as a binary universe-selection rule.
    """
    r = _finite_array(simple_returns)
    r = r[r > -0.999999]
    if len(r) < int(min_observations):
        return None
    if len(r) > int(max_observations):
        r = r[-int(max_observations):]

    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    robust_sigma = 1.4826 * mad
    if not math.isfinite(robust_sigma) or robust_sigma < 1e-6:
        robust_sigma = float(np.std(r, ddof=1)) if len(r) > 1 else 0.01
    robust_sigma = max(robust_sigma, 1e-4)

    threshold = float(jump_sigma_threshold) * robust_sigma
    jump_mask = np.abs(r - med) > threshold
    jump_r = r[jump_mask]
    clip_lo, clip_hi = med - threshold, med + threshold
    r_diff = np.clip(r, clip_lo, clip_hi)

    short_raw = _ewma_mean(r_diff, float(drift_short_half_life_days)) * TRADING_DAYS
    long_raw = _ewma_mean(r_diff, float(drift_long_half_life_days)) * TRADING_DAYS
    cap = abs(float(max_abs_annual_drift))
    shrink = float(drift_shrinkage)
    mu_short = float(np.clip(short_raw * shrink, -cap, cap))
    mu_long = float(np.clip(long_raw * shrink, -cap, cap))
    w_short = float(np.clip(drift_short_weight, 0.0, 1.0))
    mu = float(np.clip(w_short * mu_short + (1.0 - w_short) * mu_long, -cap, cap))

    short_neff = _ewma_effective_n(len(r_diff), float(drift_short_half_life_days))
    long_neff = _ewma_effective_n(len(r_diff), float(drift_long_half_life_days))
    mu_short_se = robust_sigma / math.sqrt(short_neff) * TRADING_DAYS * abs(shrink)
    mu_long_se = robust_sigma / math.sqrt(long_neff) * TRADING_DAYS * abs(shrink)
    # Conservative because the short/long EWMA estimates are positively correlated.
    mu_se = w_short * mu_short_se + (1.0 - w_short) * mu_long_se

    denom = max(abs(mu_short), abs(mu_long), 1e-12)
    same_sign = mu_short * mu_long > 0.0
    agreement_mag = min(abs(mu_short), abs(mu_long)) / denom
    drift_agreement = float(agreement_mag if same_sign else -agreement_mag)

    s = pd.Series(r_diff, dtype=float)
    rv = s.rolling(int(variance_window), min_periods=int(variance_window)).var(ddof=1)
    rv_ann = (rv * TRADING_DAYS).dropna().to_numpy(dtype=float)
    if len(rv_ann) < 10:
        return None
    rv_ann = np.clip(rv_ann, 1e-8, 9.0)
    v0 = float(rv_ann[-1])
    theta_emp = float(np.mean(rv_ann))

    x = rv_ann[:-1]
    y = rv_ann[1:]
    if len(x) >= 3 and float(np.var(x)) > EPS:
        beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
        beta = float(np.clip(beta, 0.01, 0.999))
        alpha = float(np.mean(y) - beta * np.mean(x))
        theta_ar = alpha / max(1.0 - beta, 1e-6)
        if not math.isfinite(theta_ar) or theta_ar <= 0:
            theta_ar = theta_emp
        theta = float(np.clip(0.5 * theta_emp + 0.5 * theta_ar, 1e-8, 9.0))
        kappa = float(np.clip(-math.log(beta) * TRADING_DAYS, 0.05, 25.0))
        residual = y - (alpha + beta * x)
        mean_v = max(float(np.mean(x)), 1e-8)
        xi = float(np.std(residual, ddof=1) * math.sqrt(TRADING_DAYS) / math.sqrt(mean_v))
        xi = float(np.clip(xi, 0.01, 5.0))
    else:
        theta = theta_emp
        kappa = 2.0
        xi = 0.5

    rv_series = s.rolling(int(variance_window), min_periods=int(variance_window)).var(ddof=1) * TRADING_DAYS
    dv = rv_series.diff()
    aligned = pd.DataFrame({"r": s, "dv": dv}).dropna()
    if len(aligned) >= 20 and aligned["r"].std(ddof=1) > 0 and aligned["dv"].std(ddof=1) > 0:
        rho = float(aligned["r"].corr(aligned["dv"]))
    else:
        rho = -0.3
    rho = float(np.clip(rho, -0.95, 0.95))

    years = max(len(r) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    lambda_jump = float(np.clip(len(jump_r) / years, 0.0, 24.0))
    if len(jump_r):
        jump_log = np.log1p(np.clip(jump_r, -0.95, None))
        jump_mu = float(np.mean(jump_log))
        jump_sigma = float(np.std(jump_log, ddof=1)) if len(jump_log) > 1 else robust_sigma
    else:
        jump_mu = 0.0
        jump_sigma = robust_sigma
    jump_sigma = float(np.clip(abs(jump_sigma), 1e-4, 1.0))

    return Calibration(
        n_obs=int(len(r)),
        mu=mu,
        mu_short=mu_short,
        mu_long=mu_long,
        drift_agreement=drift_agreement,
        v0=max(v0, 1e-8),
        theta=max(theta, 1e-8),
        kappa=kappa,
        xi=xi,
        rho=rho,
        lambda_jump=lambda_jump,
        jump_mu_log=jump_mu,
        jump_sigma_log=jump_sigma,
        jump_count=int(len(jump_r)),
        robust_daily_sigma=robust_sigma,
        mu_short_se=float(mu_short_se),
        mu_long_se=float(mu_long_se),
        mu_se=float(mu_se),
        mu_raw=float(mu),
        lambda_jump_raw=float(lambda_jump),
        jump_mu_log_raw=float(jump_mu),
        jump_sigma_log_raw=float(jump_sigma),
    )


def regularize_calibration_cross_section(
    calibrations: dict[int, Calibration],
    *,
    jump_prior_strength: float = 8.0,
    max_mu_reliability_weight: float = 0.35,
) -> tuple[dict[int, Calibration], dict]:
    """Regularize same-date drift and sparse jump estimates across candidates.

    This removes the old positive-drift sign gate.  Drift is empirical-Bayes
    shrunk according to its estimated standard error; jump parameters are pooled
    toward same-date priors according to observed jump count.
    """
    if not calibrations:
        return {}, {"n": 0}
    keys = list(calibrations)
    cals = [calibrations[k] for k in keys]
    mu = np.asarray([c.mu for c in cals], dtype=float)
    se = np.asarray([c.mu_se for c in cals], dtype=float)
    try:
        _mu_shrunk, mu_w_raw, prior_mu, between_var = empirical_bayes_normal(mu, se)
        # Patch 2.9.3: the cross-sectional search itself creates winner's-curse
        # pressure.  A statistically "reliable" extreme drift estimate must not
        # be allowed to retain ~90%+ of its stock-specific value and then become
        # the alpha signal.  Heston-Merton is now a risk model, so cap the amount
        # of stock-specific drift that can survive into path generation.
        cap = float(np.clip(max_mu_reliability_weight, 0.0, 1.0))
        mu_w = np.minimum(np.asarray(mu_w_raw, dtype=float), cap)
        mu_shrunk = float(prior_mu) + mu_w * (mu - float(prior_mu))
    except ValueError:
        prior_mu = float(np.nanmedian(mu)) if len(mu) else 0.0
        cap = float(np.clip(max_mu_reliability_weight, 0.0, 1.0))
        mu_w = np.full_like(mu, cap)
        mu_shrunk = float(prior_mu) + mu_w * (mu - float(prior_mu))
        between_var = 0.0

    lambdas = np.asarray([max(c.lambda_jump, 0.0) for c in cals], dtype=float)
    jump_mu = np.asarray([c.jump_mu_log for c in cals], dtype=float)
    jump_sigma = np.asarray([max(c.jump_sigma_log, 1e-6) for c in cals], dtype=float)
    counts = np.asarray([max(c.jump_count, 0) for c in cals], dtype=float)
    valid_jump = counts > 0
    prior_lambda_log = float(np.nanmedian(np.log1p(lambdas)))
    prior_jump_mu = float(np.nanmedian(jump_mu[valid_jump])) if np.any(valid_jump) else 0.0
    prior_jump_sigma_log = float(np.nanmedian(np.log(jump_sigma[valid_jump]))) if np.any(valid_jump) else float(np.nanmedian(np.log(jump_sigma)))
    jw = np.clip(counts / (counts + float(jump_prior_strength)), 0.0, 1.0)
    lambda_reg = np.expm1(prior_lambda_log + jw * (np.log1p(lambdas) - prior_lambda_log))
    jump_mu_reg = prior_jump_mu + jw * (jump_mu - prior_jump_mu)
    jump_sigma_reg = np.exp(prior_jump_sigma_log + jw * (np.log(jump_sigma) - prior_jump_sigma_log))

    out: dict[int, Calibration] = {}
    for i, key in enumerate(keys):
        c = cals[i]
        out[key] = replace(
            c,
            mu=float(mu_shrunk[i]),
            lambda_jump=float(np.clip(lambda_reg[i], 0.0, 24.0)),
            jump_mu_log=float(jump_mu_reg[i]),
            jump_sigma_log=float(np.clip(jump_sigma_reg[i], 1e-4, 1.0)),
            mu_raw=float(c.mu_raw if math.isfinite(c.mu_raw) else c.mu),
            lambda_jump_raw=float(c.lambda_jump_raw if math.isfinite(c.lambda_jump_raw) else c.lambda_jump),
            jump_mu_log_raw=float(c.jump_mu_log_raw if math.isfinite(c.jump_mu_log_raw) else c.jump_mu_log),
            jump_sigma_log_raw=float(c.jump_sigma_log_raw if math.isfinite(c.jump_sigma_log_raw) else c.jump_sigma_log),
            mu_shrink_weight=float(mu_w[i]),
            jump_shrink_weight=float(jw[i]),
            mu_prior=float(prior_mu),
        )
    return out, {
        "n": len(out),
        "mu_prior": float(prior_mu),
        "mu_between_var": float(between_var),
        "mu_reliability_cap": float(np.clip(max_mu_reliability_weight, 0.0, 1.0)),
        "jump_lambda_prior": float(np.expm1(prior_lambda_log)),
        "jump_mu_prior": float(prior_jump_mu),
        "jump_sigma_prior": float(np.exp(prior_jump_sigma_log)),
    }


def calibration_to_params(s0: float, calibration: Calibration) -> HestonMertonParams:
    return HestonMertonParams(
        spot=float(s0),
        v0=float(calibration.v0),
        kappa=float(calibration.kappa),
        theta=float(calibration.theta),
        xi=float(calibration.xi),
        rho=float(calibration.rho),
        mu=float(calibration.mu),
        jump_lambda=float(calibration.lambda_jump),
        jump_mu=float(calibration.jump_mu_log),
        jump_sigma=float(calibration.jump_sigma_log),
        drift_convention="total_arithmetic",
    )


def monte_carlo_config(
    spec: dict,
    *,
    horizon_days: int,
    requested_paths: int,
    stop_pct: float,
    purpose: str,
) -> MonteCarloConfig:
    """Translate legacy path budgets into adaptive RQMC precision tiers."""
    model_mc = spec.get("model", {}).get("monte_carlo", {})
    n_scrambles = int(model_mc.get("n_scrambles", 8))
    per_scramble = max(2, int(math.ceil(max(int(requested_paths), n_scrambles) / n_scrambles)))
    min_power = max(1, int(math.ceil(math.log2(per_scramble))))
    is_screen = "optimizer_panel" in str(purpose).lower()
    max_extra = int(model_mc.get("screen_max_extra_power" if is_screen else "exact_max_extra_power", 3 if is_screen else 4))
    hard_max = int(model_mc.get("max_power", 13))
    max_power = min(hard_max, min_power + max_extra)
    target_abs = float(model_mc.get("screen_target_abs_se" if is_screen else "exact_target_abs_se", 0.0010 if is_screen else 0.0005))
    target_rel = float(model_mc.get("screen_target_rel_se" if is_screen else "exact_target_rel_se", 0.10 if is_screen else 0.05))
    return MonteCarloConfig(
        horizon_days=int(horizon_days),
        trading_days_per_year=int(model_mc.get("trading_days_per_year", TRADING_DAYS)),
        substeps_per_day=int(model_mc.get("substeps_per_day", 2)),
        psi_c=float(model_mc.get("psi_c", 1.5)),
        n_scrambles=n_scrambles,
        min_power=min_power,
        max_power=max_power,
        base_seed=int(spec.get("random_seed", 871231)),
        target_abs_se=target_abs,
        target_rel_se=target_rel,
        relative_floor=float(model_mc.get("relative_floor", 0.0025)),
        stop_pct=float(stop_pct),
        martingale_correction=bool(model_mc.get("martingale_correction", True)),
        max_martingale_fallback_rate=float(model_mc.get("max_martingale_fallback_rate", 0.001)),
    )


def adaptive_entry_forecast(
    s0: float,
    calibration: Calibration,
    spec: dict,
    *,
    horizon_days: int,
    requested_paths: int,
    stop_pct: float,
    purpose: str,
    context_key: str,
) -> dict:
    params = calibration_to_params(s0, calibration)
    cfg = monte_carlo_config(spec, horizon_days=horizon_days, requested_paths=requested_paths, stop_pct=stop_pct, purpose=purpose)
    result = rqmc_trade_forecast(params, cfg, crn_context_key=context_key)
    return result.to_legacy_record(params, cfg)


def adaptive_entry_forecasts_for_stops(
    s0: float,
    calibration: Calibration,
    spec: dict,
    *,
    horizon_days: int,
    requested_paths: int,
    stop_pcts: list[float],
    purpose: str,
    context_key: str,
) -> dict[float, dict]:
    params = calibration_to_params(s0, calibration)
    base_stop = float(stop_pcts[0])
    cfg = monte_carlo_config(spec, horizon_days=horizon_days, requested_paths=requested_paths, stop_pct=base_stop, purpose=purpose)
    results = rqmc_trade_forecasts_for_stops(params, cfg, stop_pcts=stop_pcts, crn_context_key=context_key)
    out: dict[float, dict] = {}
    for stop, result in results.items():
        stop_cfg = replace(cfg, stop_pct=float(stop))
        out[float(stop)] = result.to_legacy_record(params, stop_cfg)
    return out


def adaptive_recheck_forecast(
    s0: float,
    calibration: Calibration,
    spec: dict,
    *,
    horizon_days: int,
    requested_paths: int,
    purpose: str,
    context_key: str,
) -> dict:
    params = calibration_to_params(s0, calibration)
    stop = float(spec.get("standalone", {}).get("emergency_stop_pct", 0.025))
    cfg = monte_carlo_config(spec, horizon_days=horizon_days, requested_paths=requested_paths, stop_pct=stop, purpose=purpose)
    result = rqmc_terminal_forecast(params, cfg, crn_context_key=context_key)
    return result.to_legacy_record(params, cfg)


def simulate_price_paths(
    s0: float,
    calibration: Calibration,
    *,
    horizon_days: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """QE-M Heston-Merton path simulator preserving the legacy public API.

    Production entry forecasts use scrambled Sobol RQMC through the adaptive
    helpers above.  This function uses deterministic pseudo-random uniforms only
    for compatibility/tests and returns t=1..H closes like the old API.
    """
    if s0 <= 0 or not math.isfinite(s0):
        raise ValueError("s0 must be finite and positive")
    horizon_days = int(horizon_days)
    n_paths = int(n_paths)
    if horizon_days <= 0 or n_paths <= 0:
        raise ValueError("horizon_days and n_paths must be positive")
    cfg = MonteCarloConfig(
        horizon_days=horizon_days,
        substeps_per_day=2,
        n_scrambles=2,
        min_power=1,
        max_power=1,
        stop_pct=0.025,
    )
    params = calibration_to_params(s0, calibration)
    rng = np.random.default_rng(int(seed))
    u = rng.random((n_paths, 4 * cfg.n_substeps))
    np.clip(u, cfg.random_clip_eps, 1.0 - cfg.random_clip_eps, out=u)
    daily, _ = simulate_daily_close_paths_from_uniforms(params, cfg, u)
    return daily[:, 1:]


def terminal_stats(paths: np.ndarray, s0: float, calibration: Calibration) -> TerminalStats:
    terminal = np.asarray(paths[:, -1], dtype=float)
    ret = terminal / float(s0) - 1.0
    ret = ret[np.isfinite(ret)]
    if len(ret) == 0:
        raise RuntimeError("No finite simulated terminal returns")
    q5 = float(np.quantile(ret, 0.05))
    worst = ret[ret <= q5]
    es5 = float(np.mean(worst)) if len(worst) else q5
    return TerminalStats(
        expected_return=float(np.mean(ret)),
        median_return=float(np.median(ret)),
        probability_positive=float(np.mean(ret > 0.0)),
        probability_loss_5=float(np.mean(ret <= -0.05)),
        probability_loss_8=float(np.mean(ret <= -0.08)),
        probability_loss_10=float(np.mean(ret <= -0.10)),
        expected_shortfall_5=es5,
        terminal_volatility=float(np.std(ret, ddof=1)) if len(ret) > 1 else 0.0,
        jump_probability_horizon=float(1.0 - math.exp(-calibration.lambda_jump * paths.shape[1] / TRADING_DAYS)),
    )


def simulate_terminal_distribution(
    s0: float,
    calibration: Calibration,
    *,
    horizon_days: int,
    n_paths: int,
    seed: int,
) -> TerminalStats:
    paths = simulate_price_paths(s0, calibration, horizon_days=horizon_days, n_paths=n_paths, seed=seed)
    return terminal_stats(paths, s0, calibration)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
