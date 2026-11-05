from __future__ import annotations
import numpy as np
import pandas as pd


def percentile_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Cross-sectional percentile composite. Higher is always better."""
    score = pd.Series(0.0, index=df.index, dtype=float)
    for c, w in weights.items():
        if c not in df.columns:
            raise KeyError(f"Required ranking column missing: {c}")
        values = pd.to_numeric(df[c], errors="coerce")
        score = score.add(values.rank(pct=True, method="average") * float(w), fill_value=0.0)
    return score


def rank_month_end(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply frozen leadership filters and rank one common market month-end cross-section."""
    s = cfg["screener"]
    x = frame.copy()

    required = {
        "permno", "ticker", "eligible", "close", "adv20", "sma100", "sma200",
        "momentum_3_1", "momentum_6_1", "momentum_12_1", "trend_strength",
    }
    missing = sorted(required.difference(x.columns))
    if missing:
        raise KeyError(f"Month-end frame missing required columns: {missing}")

    # Defensive type normalization. DuckDB normally returns numeric dtypes, but this makes
    # the screener robust to mixed parquet schemas / older partial stores.
    numeric = [
        "permno", "close", "adv20", "sma100", "sma200", "momentum_3_1",
        "momentum_6_1", "momentum_12_1", "trend_strength",
    ]
    for c in numeric:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x["eligible"] = x["eligible"].fillna(False).astype(bool)
    x = x[x["eligible"]]

    finite_cols = ["permno", "close", "adv20", "sma100", "sma200", *s["score_weights"].keys()]
    finite_cols = list(dict.fromkeys(finite_cols))
    finite_mask = np.ones(len(x), dtype=bool)
    for c in finite_cols:
        finite_mask &= np.isfinite(pd.to_numeric(x[c], errors="coerce").to_numpy(dtype=float, na_value=np.nan))
    x = x.loc[finite_mask].copy()

    x = x[x["momentum_12_1"] >= float(s["min_12_1_momentum"])]
    x = x[x["momentum_6_1"] >= float(s["min_6_1_momentum"])]
    if s.get("require_close_above_sma100", True):
        x = x[x["close"] > x["sma100"]]
    if s.get("require_close_above_sma200", True):
        x = x[x["close"] > x["sma200"]]

    if x.empty:
        return x.assign(leadership_score=pd.Series(dtype=float), rank=pd.Series(dtype=int))

    x["leadership_score"] = percentile_score(x, s["score_weights"])
    # Stable deterministic tie breakers make repeated runs byte-for-byte reproducible.
    x = x.sort_values(
        ["leadership_score", "adv20", "permno"],
        ascending=[False, False, True],
        kind="mergesort",
    ).copy()
    x["rank"] = np.arange(1, len(x) + 1, dtype=int)
    return x
