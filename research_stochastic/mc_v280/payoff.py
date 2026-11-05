from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PayoffResult:
    returns: np.ndarray
    stopped: np.ndarray
    exit_day: np.ndarray
    entry_price: np.ndarray
    exit_price: np.ndarray


def stopped_trade_payoff(close_paths: np.ndarray, stop_pct: float) -> PayoffResult:
    """Apply the strategy's corrected trade semantics to simulated DAILY closes.

    Path indexing:
      t=0 : signal-day close (NO ENTRY)
      t=1 : next-close entry
      t=2 : first legal stop check
      ...
      t=H : final legal close / terminal exit

    The stop is close-based. A close below the nominal stop realizes the actual close,
    so losses may exceed stop_pct after a gap.
    """
    p = np.asarray(close_paths, dtype=float)
    if p.ndim != 2:
        raise ValueError("close_paths must be [n_paths, horizon_days+1]")
    if p.shape[1] < 3:
        raise ValueError("need t=0 signal, t=1 entry, and at least t=2 exit/stop")
    if not (0.0 < stop_pct < 1.0):
        raise ValueError("stop_pct must be in (0, 1)")
    if np.any(~np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("close_paths contains non-finite or non-positive prices")

    n, cols = p.shape
    final_day = cols - 1
    entry = p[:, 1]
    stop_level = entry * (1.0 - stop_pct)

    stopped = np.zeros(n, dtype=bool)
    exit_day = np.full(n, final_day, dtype=np.int16)
    exit_price = p[:, final_day].copy()

    for day in range(2, cols):
        hit = (~stopped) & (p[:, day] <= stop_level)
        if np.any(hit):
            stopped[hit] = True
            exit_day[hit] = day
            exit_price[hit] = p[hit, day]

    returns = exit_price / entry - 1.0
    return PayoffResult(
        returns=returns,
        stopped=stopped,
        exit_day=exit_day,
        entry_price=entry,
        exit_price=exit_price,
    )


def lower_expected_shortfall(x: np.ndarray, q: float) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return float("nan")
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1)")
    threshold = np.quantile(arr, q)
    tail = arr[arr <= threshold]
    return float(np.mean(tail)) if tail.size else float(threshold)
