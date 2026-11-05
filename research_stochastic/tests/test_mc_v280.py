from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mc_v280 import HestonMertonParams, MonteCarloConfig, paired_difference, stopped_trade_payoff
from mc_v280.qe_heston_merton import simulate_daily_close_paths_from_uniforms
from mc_v280.rqmc import sobol_uniforms
from mc_v280.distributions import normal_ppf, poisson_ppf, student_t_critical_two_sided


def test_payoff_uses_next_close_and_no_entry_day_stop():
    paths = np.array([[100.0, 90.0, 89.0, 87.0, 95.0]])
    out = stopped_trade_payoff(paths, 0.025)
    assert out.exit_day[0] == 3
    assert np.isclose(out.returns[0], 87.0 / 90.0 - 1.0)


def test_gap_loss_is_not_clamped_to_stop():
    paths = np.array([[100.0, 100.0, 90.0]])
    out = stopped_trade_payoff(paths, 0.025)
    assert np.isclose(out.returns[0], -0.10)


def test_sobol_same_context_is_reproducible():
    a, _ = sobol_uniforms(power=4, dimension=8, base_seed=1, context_key="D", scramble_id=0)
    b, _ = sobol_uniforms(power=4, dimension=8, base_seed=1, context_key="D", scramble_id=0)
    assert np.array_equal(a, b)


def test_paired_difference_detects_clear_gap():
    a = np.array([0.02, 0.021, 0.019, 0.0205, 0.0195, 0.0202, 0.0198, 0.0201])
    b = np.array([0.01, 0.011, 0.009, 0.0105, 0.0095, 0.0102, 0.0098, 0.0101])
    d = paired_difference(a, b, indifference_zone=0.001)
    assert d.resolved
    assert d.ci_low > 0.001


def test_qe_paths_are_positive_and_finite():
    cfg = MonteCarloConfig(horizon_days=5, substeps_per_day=2, n_scrambles=2, min_power=5, max_power=5)
    p = HestonMertonParams(
        spot=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7,
        mu=0.07, jump_lambda=0.2, jump_mu=-0.04, jump_sigma=0.08,
    )
    u, _ = sobol_uniforms(power=5, dimension=4*cfg.n_substeps, base_seed=cfg.base_seed, context_key="T", scramble_id=0)
    paths, diag = simulate_daily_close_paths_from_uniforms(p, cfg, u)
    assert np.all(np.isfinite(paths))
    assert np.all(paths > 0)
    assert diag.martingale_fallback_rate <= cfg.max_martingale_fallback_rate


def test_dependency_free_distribution_transforms():
    q = np.array([0.001, 0.025, 0.5, 0.975, 0.999])
    z = normal_ppf(q)
    assert np.allclose(z, [-3.090232306, -1.959963985, 0.0, 1.959963985, 3.090232306], atol=2e-7)
    u = np.array([0.1, 0.8, 0.95, 0.999])
    assert np.array_equal(poisson_ppf(u, 0.5), np.array([0, 1, 2, 4]))
    assert abs(student_t_critical_two_sided(0.05, 7) - 2.364624252) < 0.001


def test_sobol_scramble_is_reproducible_and_balanced():
    a, _ = sobol_uniforms(power=5, dimension=8, base_seed=7, context_key="BAL", scramble_id=0)
    b, _ = sobol_uniforms(power=5, dimension=8, base_seed=7, context_key="BAL", scramble_id=0)
    c, _ = sobol_uniforms(power=5, dimension=8, base_seed=7, context_key="BAL", scramble_id=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # A scrambled base-2 Sobol net retains one point per 1/32 stratum in each coordinate.
    for j in range(a.shape[1]):
        bins = np.floor(a[:, j] * 32).astype(int)
        assert len(np.unique(bins)) == 32


def test_legacy_forecast_record_preserves_model_fields_needed_by_live_risk_allocator():
    from mc_v280 import HestonMertonParams, MonteCarloConfig, forecast_trade_distribution

    params = HestonMertonParams(
        spot=100.0,
        v0=0.04,
        kappa=2.0,
        theta=0.05,
        xi=0.45,
        rho=-0.55,
        mu=0.08,
        jump_lambda=1.25,
        jump_mu=-0.025,
        jump_sigma=0.08,
        drift_convention="total_arithmetic",
    )
    cfg = MonteCarloConfig(
        horizon_days=5,
        substeps_per_day=1,
        n_scrambles=4,
        min_power=3,
        max_power=3,
        stop_pct=0.025,
    )
    result = forecast_trade_distribution(params, cfg, crn_context_key="schema-regression")
    record = result.to_legacy_record(params, cfg)

    expected = {
        "mu": params.mu,
        "v0": params.v0,
        "theta": params.theta,
        "kappa": params.kappa,
        "xi": params.xi,
        "rho": params.rho,
        "lambda_jump": params.jump_lambda,
        "jump_mu_log": params.jump_mu,
        "jump_sigma_log": params.jump_sigma,
    }
    for key, value in expected.items():
        assert key in record
        assert np.isclose(float(record[key]), float(value))
    assert "terminal_volatility" in record
