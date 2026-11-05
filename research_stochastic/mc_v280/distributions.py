from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def normal_ppf(u: np.ndarray | float) -> np.ndarray | float:
    """Inverse standard-normal CDF using Acklam's rational approximation.

    Accuracy is ~1e-8 or better over the clipped RQMC range used by this engine.
    No SciPy dependency is required.
    """
    x = np.asarray(u, dtype=float)
    if np.any((x <= 0.0) | (x >= 1.0)):
        raise ValueError("normal_ppf requires probabilities strictly inside (0,1)")

    a = np.array([
        -3.969683028665376e01, 2.209460984245205e02,
        -2.759285104469687e02, 1.383577518672690e02,
        -3.066479806614716e01, 2.506628277459239e00,
    ])
    b = np.array([
        -5.447609879822406e01, 1.615858368580409e02,
        -1.556989798598866e02, 6.680131188771972e01,
        -1.328068155288572e01,
    ])
    c = np.array([
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00,  2.938163982698783e00,
    ])
    d = np.array([
         7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00,
    ])
    plow = 0.02425
    phigh = 1.0 - plow
    out = np.empty_like(x)

    lo = x < plow
    hi = x > phigh
    mid = ~(lo | hi)

    if np.any(lo):
        q = np.sqrt(-2.0 * np.log(x[lo]))
        out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if np.any(hi):
        q = np.sqrt(-2.0 * np.log1p(-x[hi]))
        out[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if np.any(mid):
        q = x[mid] - 0.5
        r = q*q
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)

    return float(out) if x.ndim == 0 else out


def poisson_ppf(u: np.ndarray | float, lam: float) -> np.ndarray | int:
    """Exact inverse CDF for a Poisson variable using recurrence of its PMF.

    Heston-Merton calls this with ``lam = jump_lambda * dt``; at daily/subdaily
    resolution lam is small, making the recurrence both exact and inexpensive.
    """
    if lam < 0.0 or not math.isfinite(lam):
        raise ValueError("Poisson lambda must be finite and non-negative")
    x = np.asarray(u, dtype=float)
    if np.any((x < 0.0) | (x >= 1.0)):
        raise ValueError("poisson_ppf requires probabilities in [0,1)")
    out = np.zeros(x.shape, dtype=np.int64)
    if lam == 0.0:
        return int(out) if x.ndim == 0 else out
    if lam > 700.0:
        raise ValueError("per-step Poisson lambda is implausibly large for this Heston-Merton engine")

    pk = math.exp(-lam)
    cdf = pk
    unresolved = x > cdf
    k = 0
    max_k = int(max(64, math.ceil(lam + 14.0*math.sqrt(lam + 1.0) + 64.0)))
    while np.any(unresolved):
        k += 1
        pk *= lam / k
        cdf += pk
        hit = unresolved & (x <= cdf)
        out[hit] = k
        unresolved &= ~hit
        if k >= max_k:
            # Reaching this branch would indicate floating-point saturation, not a
            # normal Heston-Merton calibration. Assign conservatively and stop.
            out[unresolved] = k
            break
    return int(out) if x.ndim == 0 else out


def student_t_critical_two_sided(alpha: float, df: int) -> float:
    """Two-sided Student-t critical value without SciPy.

    df=1 and df=2 use exact inversions. For df>=3 a third-order Cornish-Fisher
    expansion around the normal quantile is used. At the production setting
    (8 scrambles => df=7, alpha=.05) the error is below 0.001 versus SciPy.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if df < 1:
        raise ValueError("df must be >= 1")
    p = 1.0 - alpha/2.0
    if df == 1:
        return math.tan(math.pi*(p - 0.5))
    if df == 2:
        y = 2.0*p - 1.0
        return math.sqrt(2.0)*y / math.sqrt(max(1e-30, 1.0-y*y))
    z = NormalDist().inv_cdf(p)
    v = float(df)
    z2 = z*z
    return float(
        z
        + (z**3 + z)/(4.0*v)
        + (5.0*z**5 + 16.0*z**3 + 3.0*z)/(96.0*v*v)
        + (3.0*z**7 + 19.0*z**5 + 17.0*z**3 - 15.0*z)/(384.0*v**3)
    )
