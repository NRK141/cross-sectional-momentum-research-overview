from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .config import MonteCarloConfig


PATCH_VERSION = "2.9.1"
SIMULATION_SCHEME = "andersen_qe_m_heston_merton_rqmc"
ENTRY_TIMING_VERSION = "signal_t0_entry_next_close_t1"
STOP_SEMANTICS_VERSION = "close_only_first_test_t2_actual_close_fill"
RNG_VERSION = "numpy_lms_shift_scrambled_sobol_joe_kuo_512_crn_v2"
DRIFT_VERSION = "cross_sectional_empirical_bayes_drift_and_jump_regularization_v1"


@dataclass(frozen=True)
class ForecastSpec:
    patch_version: str
    simulation_scheme: str
    entry_timing_version: str
    stop_semantics_version: str
    rng_version: str
    drift_version: str
    payoff_mode: str
    config: dict

    @classmethod
    def from_config(cls, cfg: MonteCarloConfig, *, payoff_mode: str = "entry_stopped_trade") -> "ForecastSpec":
        return cls(
            patch_version=PATCH_VERSION,
            simulation_scheme=SIMULATION_SCHEME,
            entry_timing_version=ENTRY_TIMING_VERSION,
            stop_semantics_version=STOP_SEMANTICS_VERSION,
            rng_version=RNG_VERSION,
            drift_version=DRIFT_VERSION,
            payoff_mode=str(payoff_mode),
            config=asdict(cfg),
        )

    def digest(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
