from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .distributions import normal_ppf, poisson_ppf

from .config import HestonMertonParams, MonteCarloConfig


@dataclass(frozen=True)
class SimulationDiagnostics:
    martingale_fallbacks: int
    qe_low_branch_draws: int
    qe_high_branch_draws: int
    total_variance_steps: int

    @property
    def martingale_fallback_rate(self) -> float:
        if self.total_variance_steps == 0:
            return 0.0
        return self.martingale_fallbacks / self.total_variance_steps


def _deterministic_variance_step(v: np.ndarray, p: HestonMertonParams, dt: float) -> np.ndarray:
    e = np.exp(-p.kappa * dt)
    return p.theta + (v - p.theta) * e


def simulate_daily_close_paths_from_uniforms(
    params: HestonMertonParams,
    cfg: MonteCarloConfig,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, SimulationDiagnostics]:
    """QE-M Heston + exact aggregate Merton log jumps.

    uniforms shape = [n_paths, 4 * cfg.n_substeps]
    Per substep dimensions:
      u_var, u_spot, u_jump_count, u_jump_size

    The same uniforms may be reused across securities on a forecast date to create
    common random numbers for lower-noise cross-sectional comparisons.
    """
    params.validate()
    cfg.validate()
    u = np.asarray(uniforms, dtype=float)
    expected_dim = 4 * cfg.n_substeps
    if u.ndim != 2 or u.shape[1] != expected_dim:
        raise ValueError(f"uniforms must have shape [n_paths, {expected_dim}]")

    n_paths = u.shape[0]
    dt = cfg.dt
    sqrt_eps = 1e-16

    log_s = np.full(n_paths, np.log(params.spot), dtype=float)
    v = np.full(n_paths, params.v0, dtype=float)
    daily = np.empty((n_paths, cfg.horizon_days + 1), dtype=float)
    daily[:, 0] = params.spot

    # Predictor-corrector constants. For xi->0 we use a deterministic-variance branch.
    use_deterministic_variance = params.xi <= 1e-10
    if not use_deterministic_variance:
        rho = params.rho
        xi = params.xi
        gamma1 = gamma2 = 0.5
        k1 = gamma1 * dt * (params.kappa * rho / xi - 0.5) - rho / xi
        k2 = gamma2 * dt * (params.kappa * rho / xi - 0.5) + rho / xi
        k3 = gamma1 * dt * (1.0 - rho * rho)
        k4 = gamma2 * dt * (1.0 - rho * rho)
        k0_uncorrected = -rho * params.kappa * params.theta * dt / xi
        A = k2 + 0.5 * k4

    lam_dt = params.jump_lambda * dt
    diffusion_mu = params.diffusion_arithmetic_drift

    martingale_fallbacks = 0
    low_draws = 0
    high_draws = 0
    total_steps = 0

    dim = 0
    for step in range(cfg.n_substeps):
        u_var = u[:, dim]
        u_spot = u[:, dim + 1]
        u_count = u[:, dim + 2]
        u_jump = u[:, dim + 3]
        dim += 4

        z_spot = normal_ppf(u_spot)
        v_prev = v

        if use_deterministic_variance:
            v_next = _deterministic_variance_step(v_prev, params, dt)
            # Central approximation to integrated variance over the small substep.
            int_var = np.maximum(0.5 * (v_prev + v_next) * dt, 0.0)
            log_s = log_s + diffusion_mu * dt - 0.5 * int_var + np.sqrt(int_var) * z_spot
        else:
            e = np.exp(-params.kappa * dt)
            m = params.theta + (v_prev - params.theta) * e
            s2 = (
                v_prev * params.xi**2 * e * (1.0 - e) / params.kappa
                + params.theta * params.xi**2 * (1.0 - e) ** 2 / (2.0 * params.kappa)
            )
            m = np.maximum(m, sqrt_eps)
            s2 = np.maximum(s2, 0.0)
            psi = s2 / (m * m)

            low = psi <= cfg.psi_c
            high = ~low
            v_next = np.empty_like(v_prev)
            k0 = np.full_like(v_prev, k0_uncorrected)

            if np.any(low):
                psi_l = np.maximum(psi[low], 1e-14)
                invpsi2 = 2.0 / psi_l
                b2 = invpsi2 - 1.0 + np.sqrt(np.maximum(invpsi2 * (invpsi2 - 1.0), 0.0))
                a = m[low] / (1.0 + b2)
                z_var = normal_ppf(u_var[low])
                v_next[low] = a * (np.sqrt(b2) + z_var) ** 2

                if cfg.martingale_correction:
                    denom = 1.0 - 2.0 * A * a
                    regular = denom > 1e-12
                    corrected = np.empty_like(a)
                    corrected[:] = k0_uncorrected
                    if np.any(regular):
                        ar = a[regular]
                        b2r = b2[regular]
                        corrected[regular] = (
                            -A * b2r * ar / (1.0 - 2.0 * A * ar)
                            + 0.5 * np.log(1.0 - 2.0 * A * ar)
                            - (k1 + 0.5 * k3) * v_prev[low][regular]
                        )
                    martingale_fallbacks += int(np.sum(~regular))
                    k0[low] = corrected
                low_draws += int(np.sum(low))

            if np.any(high):
                psi_h = psi[high]
                mh = m[high]
                p0 = (psi_h - 1.0) / (psi_h + 1.0)
                beta = (1.0 - p0) / mh
                uh = u_var[high]
                nonzero = uh > p0
                vh = np.zeros_like(mh)
                if np.any(nonzero):
                    vh[nonzero] = np.log((1.0 - p0[nonzero]) / (1.0 - uh[nonzero])) / beta[nonzero]
                v_next[high] = vh

                if cfg.martingale_correction:
                    regular = beta > (A + 1e-12)
                    corrected = np.empty_like(beta)
                    corrected[:] = k0_uncorrected
                    if np.any(regular):
                        pr = p0[regular]
                        br = beta[regular]
                        mgf = pr + br * (1.0 - pr) / (br - A)
                        valid_mgf = np.isfinite(mgf) & (mgf > 0.0)
                        temp = corrected[regular]
                        if np.any(valid_mgf):
                            temp[valid_mgf] = (
                                -np.log(mgf[valid_mgf])
                                - (k1 + 0.5 * k3) * v_prev[high][regular][valid_mgf]
                            )
                        corrected[regular] = temp
                        local_bad = np.zeros(np.sum(regular), dtype=bool)
                        local_bad[~valid_mgf] = True
                        martingale_fallbacks += int(np.sum(local_bad))
                    martingale_fallbacks += int(np.sum(~regular))
                    k0[high] = corrected
                high_draws += int(np.sum(high))

            inst_var = np.maximum(k3 * v_prev + k4 * v_next, 0.0)
            log_s = log_s + diffusion_mu * dt + k0 + k1 * v_prev + k2 * v_next + np.sqrt(inst_var) * z_spot

        # Exact compound-Poisson aggregate when log jump sizes are normal.
        if lam_dt > 0.0:
            n_jump = poisson_ppf(u_count, lam_dt).astype(np.int64)
            has_jump = n_jump > 0
            if np.any(has_jump):
                z_jump = normal_ppf(u_jump[has_jump])
                nj = n_jump[has_jump].astype(float)
                jump_log = nj * params.jump_mu + np.sqrt(nj) * params.jump_sigma * z_jump
                log_s[has_jump] += jump_log

        v = np.maximum(v_next, 0.0)
        total_steps += n_paths

        if (step + 1) % cfg.substeps_per_day == 0:
            day = (step + 1) // cfg.substeps_per_day
            daily[:, day] = np.exp(log_s)

    diag = SimulationDiagnostics(
        martingale_fallbacks=martingale_fallbacks,
        qe_low_branch_draws=low_draws,
        qe_high_branch_draws=high_draws,
        total_variance_steps=total_steps,
    )
    if cfg.martingale_correction and diag.martingale_fallback_rate > cfg.max_martingale_fallback_rate:
        raise RuntimeError(
            "QE-M regularity failed too often: "
            f"fallback_rate={diag.martingale_fallback_rate:.6%} > "
            f"allowed={cfg.max_martingale_fallback_rate:.6%}. "
            "Increase substeps_per_day or inspect the calibrated parameters."
        )
    return daily, diag
