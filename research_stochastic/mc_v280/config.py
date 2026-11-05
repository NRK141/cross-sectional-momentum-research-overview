from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


DriftConvention = Literal["total_arithmetic", "continuous_arithmetic"]


@dataclass(frozen=True)
class HestonMertonParams:
    """Physical-measure Heston-Merton parameters.

    mu:
      - total_arithmetic: target annualized expected simple return of S.
      - continuous_arithmetic: annualized drift of the diffusion component before jumps.
    jump_mu/jump_sigma describe the NORMAL distribution of log jump sizes.
    """

    spot: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    mu: float
    jump_lambda: float = 0.0
    jump_mu: float = 0.0
    jump_sigma: float = 0.0
    drift_convention: DriftConvention = "total_arithmetic"

    def validate(self) -> None:
        if not (self.spot > 0.0):
            raise ValueError("spot must be > 0")
        if self.v0 < 0.0:
            raise ValueError("v0 must be >= 0")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be > 0")
        if self.theta < 0.0:
            raise ValueError("theta must be >= 0")
        if self.xi < 0.0:
            raise ValueError("xi must be >= 0")
        if not (-0.999999 < self.rho < 0.999999):
            raise ValueError("rho must lie strictly inside (-1, 1)")
        if self.jump_lambda < 0.0:
            raise ValueError("jump_lambda must be >= 0")
        if self.jump_sigma < 0.0:
            raise ValueError("jump_sigma must be >= 0")
        if self.drift_convention not in ("total_arithmetic", "continuous_arithmetic"):
            raise ValueError(f"unsupported drift_convention={self.drift_convention!r}")

    @property
    def jump_kappa(self) -> float:
        """E[J-1] for multiplicative jump J = exp(Y), Y normal."""
        import math

        return math.exp(self.jump_mu + 0.5 * self.jump_sigma**2) - 1.0

    @property
    def diffusion_arithmetic_drift(self) -> float:
        if self.drift_convention == "total_arithmetic":
            return self.mu - self.jump_lambda * self.jump_kappa
        return self.mu

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MonteCarloConfig:
    horizon_days: int = 20
    trading_days_per_year: int = 252
    substeps_per_day: int = 2
    psi_c: float = 1.5

    # Randomized quasi-Monte Carlo.
    n_scrambles: int = 8
    min_power: int = 8      # 2^8 points / scramble
    max_power: int = 13     # 2^13 points / scramble
    base_seed: int = 871231

    # Adaptive mean precision; either threshold can stop the escalation.
    target_abs_se: float = 2.5e-4
    target_rel_se: float = 0.05
    relative_floor: float = 0.0025

    # Strategy payoff semantics.
    stop_pct: float = 0.025

    # Diagnostics / safety.
    martingale_correction: bool = True
    max_martingale_fallback_rate: float = 1e-3
    random_clip_eps: float = 1e-12

    def validate(self) -> None:
        if self.horizon_days < 2:
            raise ValueError("horizon_days must be >= 2 for next-close entry + later stop")
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be > 0")
        if self.substeps_per_day <= 0:
            raise ValueError("substeps_per_day must be > 0")
        if not (1.0 <= self.psi_c <= 2.0):
            raise ValueError("psi_c should be in [1, 2]")
        if self.n_scrambles < 2:
            raise ValueError("n_scrambles must be >= 2 so MC SE is estimable")
        if not (1 <= self.min_power <= self.max_power):
            raise ValueError("require 1 <= min_power <= max_power")
        if not (0.0 < self.stop_pct < 1.0):
            raise ValueError("stop_pct must be in (0, 1)")
        if self.target_abs_se <= 0.0:
            raise ValueError("target_abs_se must be > 0")
        if self.target_rel_se <= 0.0:
            raise ValueError("target_rel_se must be > 0")
        if not (0.0 < self.random_clip_eps < 1e-3):
            raise ValueError("random_clip_eps must be small and positive")

    @property
    def dt(self) -> float:
        return 1.0 / (self.trading_days_per_year * self.substeps_per_day)

    @property
    def n_substeps(self) -> int:
        return self.horizon_days * self.substeps_per_day
