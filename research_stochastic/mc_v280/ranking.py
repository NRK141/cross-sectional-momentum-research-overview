from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .distributions import student_t_critical_two_sided


@dataclass(frozen=True)
class PairedDifference:
    mean_diff: float
    se_diff: float
    ci_low: float
    ci_high: float
    resolved: bool


def paired_difference(
    a_scramble_means: np.ndarray,
    b_scramble_means: np.ndarray,
    *,
    alpha: float = 0.05,
    indifference_zone: float = 0.0,
) -> PairedDifference:
    a = np.asarray(a_scramble_means, dtype=float)
    b = np.asarray(b_scramble_means, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired scramble mean arrays must be same 1-D shape")
    if a.size < 2:
        raise ValueError("need at least two paired scrambles")
    d = a - b
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    crit = student_t_critical_two_sided(alpha, d.size - 1)
    lo = mean - crit * se
    hi = mean + crit * se
    # Resolved only if the entire CI clears +/- the economically irrelevant zone.
    resolved = (lo > indifference_zone) or (hi < -indifference_zone)
    return PairedDifference(mean, se, float(lo), float(hi), bool(resolved))


def cutoff_diagnostic(
    scramble_mean_matrix: np.ndarray,
    names: list[str],
    *,
    top_k: int,
    alpha: float = 0.05,
    indifference_zone: float = 0.0,
) -> dict:
    """Assess whether the last selected name is distinguishable from first excluded.

    matrix shape [n_scrambles, n_candidates]; CRN pairing requires rows to refer to
    the same scramble/scenario block for every candidate.
    """
    x = np.asarray(scramble_mean_matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError("matrix must be [n_scrambles, len(names)]")
    if not (1 <= top_k < len(names)):
        raise ValueError("top_k must leave at least one excluded candidate")

    means = np.mean(x, axis=0)
    order = np.argsort(-means)
    selected_idx = order[top_k - 1]
    excluded_idx = order[top_k]
    pdiff = paired_difference(
        x[:, selected_idx], x[:, excluded_idx],
        alpha=alpha,
        indifference_zone=indifference_zone,
    )
    return {
        "selected_boundary_name": names[selected_idx],
        "excluded_boundary_name": names[excluded_idx],
        "selected_boundary_mean": float(means[selected_idx]),
        "excluded_boundary_mean": float(means[excluded_idx]),
        "rank_delta_to_cutoff": pdiff.mean_diff,
        "rank_delta_se": pdiff.se_diff,
        "rank_ci_low": pdiff.ci_low,
        "rank_ci_high": pdiff.ci_high,
        "rank_resolved": pdiff.resolved,
    }
