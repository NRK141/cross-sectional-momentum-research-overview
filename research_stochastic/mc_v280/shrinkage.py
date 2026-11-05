from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_bayes_normal(
    estimate: np.ndarray,
    estimate_se: np.ndarray,
    *,
    prior_mean: float | None = None,
    between_var: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Normal-normal empirical-Bayes shrinkage for noisy cross-sectional estimates.

    Returns: shrunk, weights, fitted_prior_mean, fitted_between_var.
    """
    x = np.asarray(estimate, dtype=float)
    se = np.asarray(estimate_se, dtype=float)
    valid = np.isfinite(x) & np.isfinite(se) & (se >= 0.0)
    if np.sum(valid) < 2:
        raise ValueError("need at least two valid estimates")

    xv = x[valid]
    sev = se[valid]
    if prior_mean is None:
        # Precision weighting is capped so one tiny SE cannot dominate the prior.
        floor = max(float(np.nanmedian(sev)) * 0.25, 1e-8)
        precision = 1.0 / np.maximum(sev, floor) ** 2
        prior_mean = float(np.sum(precision * xv) / np.sum(precision))
    if between_var is None:
        raw_var = float(np.var(xv, ddof=1))
        noise_var = float(np.mean(sev**2))
        between_var = max(raw_var - noise_var, 0.0)

    denom = between_var + se**2
    weights = np.divide(between_var, denom, out=np.zeros_like(denom), where=denom > 0.0)
    shrunk = prior_mean + weights * (x - prior_mean)
    shrunk[~valid] = np.nan
    weights[~valid] = np.nan
    return shrunk, weights, float(prior_mean), float(between_var)


def shrink_drift_panel(
    df: pd.DataFrame,
    *,
    drift_col: str = "mu_raw",
    drift_se_col: str = "mu_se",
    group_col: str | None = None,
    output_col: str = "mu_shrunk",
    weight_col: str = "mu_shrink_weight",
) -> pd.DataFrame:
    """Shrink stock drift estimates toward a cross-sectional or group prior."""
    out = df.copy()
    out[output_col] = np.nan
    out[weight_col] = np.nan
    out["mu_prior_mean"] = np.nan
    out["mu_prior_between_var"] = np.nan

    if group_col is None:
        groups = [(None, out.index)]
    else:
        groups = list(out.groupby(group_col, dropna=False).groups.items())

    for _, idx in groups:
        block = out.loc[idx]
        try:
            shrunk, w, pm, bv = empirical_bayes_normal(
                block[drift_col].to_numpy(float),
                block[drift_se_col].to_numpy(float),
            )
        except ValueError:
            continue
        out.loc[idx, output_col] = shrunk
        out.loc[idx, weight_col] = w
        out.loc[idx, "mu_prior_mean"] = pm
        out.loc[idx, "mu_prior_between_var"] = bv
    return out


def reliability_shrink(raw: np.ndarray, prior: np.ndarray | float, effective_n: np.ndarray, prior_strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Generic shrinkage for sparse jump/Heston estimates.

    weight = n / (n + prior_strength). For positive parameters, callers should
    preferably pass values already transformed to log scale.
    """
    x = np.asarray(raw, dtype=float)
    n = np.asarray(effective_n, dtype=float)
    p = np.asarray(prior, dtype=float)
    if prior_strength <= 0.0:
        raise ValueError("prior_strength must be > 0")
    w = np.clip(n / (n + prior_strength), 0.0, 1.0)
    return p + w * (x - p), w
