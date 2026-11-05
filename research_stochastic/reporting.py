from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


def portfolio_metrics(equity: pd.Series, annualization: int = 252) -> dict:
    x = pd.Series(equity).dropna().astype(float)
    if len(x) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "annualized_volatility": 0.0, "total_return": 0.0}
    r = x.pct_change().fillna(0.0)
    years = max((len(x) - 1) / annualization, 1 / annualization)
    cagr = float((x.iloc[-1] / x.iloc[0]) ** (1 / years) - 1) if x.iloc[0] > 0 else 0.0
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    vol = sd * math.sqrt(annualization)
    sharpe = float(r.mean() / sd * math.sqrt(annualization)) if sd > 0 else 0.0
    dd = x / x.cummax() - 1.0
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "annualized_volatility": float(vol),
        "total_return": float(x.iloc[-1] / x.iloc[0] - 1.0),
    }


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty or "net_return" not in trades.columns:
        return {
            "closed_trades": 0,
            "win_rate": 0.0,
            "profit_factor": None,
            "average_net_trade_return": 0.0,
            "median_net_trade_return": 0.0,
        }
    x = pd.to_numeric(trades["net_return"], errors="coerce").dropna().astype(float)
    wins = x[x > 0]
    losses = x[x <= 0]
    gp = float(wins.sum())
    gl = float(-losses.sum())
    pf = float(gp / gl) if gl > 0 else None
    return {
        "closed_trades": int(len(x)),
        "win_rate": float((x > 0).mean()) if len(x) else 0.0,
        "profit_factor": pf,
        "average_net_trade_return": float(x.mean()) if len(x) else 0.0,
        "median_net_trade_return": float(x.median()) if len(x) else 0.0,
    }


def safe_json_value(v):
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if not math.isfinite(v) else v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=safe_json_value), encoding="utf-8")


def compare_metrics(candidate: dict, baseline: dict | None, label: str) -> dict:
    out = {"candidate_label": label, "candidate": candidate, "baseline": None, "delta": None}
    if not baseline:
        return out
    base_metrics = baseline.get("strategy_metrics", baseline.get("portfolio_metrics", {})) or {}
    out["baseline"] = base_metrics
    delta = {}
    for k in ("cagr", "sharpe", "max_drawdown", "annualized_volatility", "total_return"):
        if k in candidate and k in base_metrics and candidate[k] is not None and base_metrics[k] is not None:
            try:
                delta[k] = float(candidate[k]) - float(base_metrics[k])
            except Exception:
                pass
    out["delta"] = delta
    return out


def realized_quantile_diagnostics(
    frame: pd.DataFrame,
    metrics: list[str],
    *,
    realized_col: str = "net_return",
    buckets: int = 5,
) -> pd.DataFrame:
    """Summarize realized outcomes by forecast-metric quantile without fitting anything.

    This is intentionally descriptive. It is emitted after a research run so future
    model revisions can be based on whether a forecast field actually discriminates
    realized outcomes.
    """
    if frame.empty or realized_col not in frame.columns:
        return pd.DataFrame(columns=["metric", "bucket", "n", "forecast_min", "forecast_max", "forecast_mean", "mean_realized_return", "median_realized_return", "win_rate"])
    rows = []
    for metric in metrics:
        if metric not in frame.columns:
            continue
        x = frame[[metric, realized_col]].copy()
        x[metric] = pd.to_numeric(x[metric], errors="coerce")
        x[realized_col] = pd.to_numeric(x[realized_col], errors="coerce")
        x = x.dropna()
        if len(x) < 2:
            continue
        q = min(int(buckets), int(x[metric].nunique()), len(x))
        if q < 2:
            continue
        try:
            labels = pd.qcut(x[metric], q=q, labels=False, duplicates="drop")
        except ValueError:
            continue
        x = x.assign(_bucket=labels)
        for b, g in x.groupby("_bucket", sort=True):
            r = g[realized_col].astype(float)
            rows.append({
                "metric": metric,
                "bucket": int(b) + 1,
                "n": int(len(g)),
                "forecast_min": float(g[metric].min()),
                "forecast_max": float(g[metric].max()),
                "forecast_mean": float(g[metric].mean()),
                "mean_realized_return": float(r.mean()),
                "median_realized_return": float(r.median()),
                "win_rate": float((r > 0).mean()),
            })
    return pd.DataFrame(rows)


def _spearman_corr_no_scipy(a: pd.Series, b: pd.Series) -> float:
    """Dependency-free Spearman rank correlation.

    Pandas delegates its built-in Spearman correlation path to SciPy.
    The production/research venv does not require SciPy, so compute average
    ranks with pandas and Pearson correlation with NumPy directly.
    """
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    ra = aa[mask].rank(method="average").to_numpy(dtype=float)
    rb = bb[mask].rank(method="average").to_numpy(dtype=float)
    if np.nanstd(ra) <= 0.0 or np.nanstd(rb) <= 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def cross_sectional_ic(
    frame: pd.DataFrame,
    metrics: list[str],
    *,
    date_col: str = "date",
    realized_col: str = "realized_forward_return",
    min_names: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast-date Spearman information coefficients for diagnostic-only outcomes."""
    if frame.empty or date_col not in frame.columns or realized_col not in frame.columns:
        return pd.DataFrame(), pd.DataFrame()
    work = frame.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    rows = []
    for dt, g in work.groupby(date_col, sort=True):
        for metric in metrics:
            if metric not in g.columns:
                continue
            x = g[[metric, realized_col]].copy()
            x[metric] = pd.to_numeric(x[metric], errors="coerce")
            x[realized_col] = pd.to_numeric(x[realized_col], errors="coerce")
            x = x.dropna()
            if len(x) < int(min_names) or x[metric].nunique() < 2 or x[realized_col].nunique() < 2:
                continue
            ic = _spearman_corr_no_scipy(x[metric], x[realized_col])
            if math.isfinite(ic):
                rows.append({"date": pd.Timestamp(dt), "metric": metric, "n": int(len(x)), "spearman_ic": ic})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = (
        detail.groupby("metric", as_index=False)
        .agg(
            forecast_dates=("spearman_ic", "size"),
            mean_ic=("spearman_ic", "mean"),
            median_ic=("spearman_ic", "median"),
            ic_std=("spearman_ic", "std"),
            positive_ic_fraction=("spearman_ic", lambda s: float((s > 0).mean())),
        )
        .sort_values(["mean_ic", "median_ic"], ascending=False)
        .reset_index(drop=True)
    )
    return detail, summary



def cross_sectional_ic_by_source(
    frame: pd.DataFrame,
    metrics: list[str],
    *,
    source_col: str = "signal_source",
    date_col: str = "date",
    realized_col: str = "realized_forward_return",
    min_names: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Source-split forecast-date ICs (e.g. MONTH_END vs WEEKLY_REFILL)."""
    if frame.empty or source_col not in frame.columns:
        return pd.DataFrame(), pd.DataFrame()
    details = []
    summaries = []
    for source, g in frame.groupby(source_col, sort=True):
        d, sm = cross_sectional_ic(
            g, metrics, date_col=date_col, realized_col=realized_col, min_names=min_names
        )
        if len(d):
            d.insert(0, "signal_source", str(source))
            details.append(d)
        if len(sm):
            sm.insert(0, "signal_source", str(source))
            summaries.append(sm)
    return (
        pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
    )


def non_overlapping_forecast_frame(
    frame: pd.DataFrame,
    *,
    source_col: str = "signal_source",
    date_col: str = "date",
    entry_col: str = "realized_entry_date",
    terminal_col: str = "realized_terminal_date",
) -> pd.DataFrame:
    """Keep source-specific forecast dates whose realized horizons do not overlap.

    This is diagnostic-only and uses realized date labels after the simulation. It
    does not affect trading decisions. The goal is to avoid treating adjacent
    weekly 20-session forecasts as independent IC observations.
    """
    required = {source_col, date_col, entry_col, terminal_col}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=frame.columns)
    work = frame.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[entry_col] = pd.to_datetime(work[entry_col], errors="coerce")
    work[terminal_col] = pd.to_datetime(work[terminal_col], errors="coerce")
    keep_pairs: set[tuple[str, pd.Timestamp]] = set()
    for source, sg in work.groupby(source_col, sort=True):
        date_windows = (
            sg.dropna(subset=[entry_col, terminal_col])
            .groupby(date_col, as_index=False)
            .agg(_entry=(entry_col, "min"), _terminal=(terminal_col, "max"))
            .sort_values(date_col)
        )
        last_terminal = None
        for _, rec in date_windows.iterrows():
            entry = pd.Timestamp(rec["_entry"])
            terminal = pd.Timestamp(rec["_terminal"])
            if last_terminal is None or entry > last_terminal:
                keep_pairs.add((str(source), pd.Timestamp(rec[date_col])))
                last_terminal = terminal
    mask = [
        (str(src), pd.Timestamp(dt)) in keep_pairs
        for src, dt in zip(work[source_col], work[date_col])
    ]
    return work.loc[mask].copy()

def trade_concentration_diagnostics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"closed_trades": 0}
    out = {"closed_trades": int(len(trades))}
    if "net_pnl_usd" in trades.columns:
        pnl = pd.to_numeric(trades["net_pnl_usd"], errors="coerce").dropna().astype(float)
        positive = pnl[pnl > 0].sort_values(ascending=False)
        gp = float(positive.sum()) if len(positive) else 0.0
        out["total_net_pnl_usd"] = float(pnl.sum()) if len(pnl) else 0.0
        out["gross_positive_pnl_usd"] = gp
        if len(positive) and gp > 0:
            out["largest_winner_share_of_gross_positive_pnl"] = float(positive.iloc[0] / gp)
            out["top5_winner_share_of_gross_positive_pnl"] = float(positive.iloc[:5].sum() / gp)
            out["top10_winner_share_of_gross_positive_pnl"] = float(positive.iloc[:10].sum() / gp)
    if "net_return" in trades.columns:
        r = pd.to_numeric(trades["net_return"], errors="coerce").dropna().astype(float)
        if len(r):
            out["largest_net_trade_return"] = float(r.max())
            out["worst_net_trade_return"] = float(r.min())
    if "exit_reason" in trades.columns:
        exit_stats = []
        for reason, g in trades.groupby("exit_reason", sort=True):
            r = pd.to_numeric(g["net_return"], errors="coerce").dropna().astype(float)
            exit_stats.append({
                "exit_reason": str(reason),
                "trades": int(len(g)),
                "average_net_return": float(r.mean()) if len(r) else None,
                "win_rate": float((r > 0).mean()) if len(r) else None,
            })
        out["exit_reason_stats"] = exit_stats
    return out


def equity_risk_diagnostics(equity: pd.DataFrame) -> dict:
    if equity.empty or "equity" not in equity.columns:
        return {}
    x = equity.copy()
    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"])
    e = pd.to_numeric(x["equity"], errors="coerce").astype(float)
    peak = e.cummax()
    dd = e / peak - 1.0
    worst_idx = int(dd.idxmin()) if len(dd) else 0
    out = {
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "average_gross_exposure": float(pd.to_numeric(x.get("gross_exposure", 0.0), errors="coerce").fillna(0.0).mean()) if len(x) else 0.0,
        "max_gross_exposure": float(pd.to_numeric(x.get("gross_exposure", 0.0), errors="coerce").fillna(0.0).max()) if len(x) else 0.0,
    }
    if "date" in x.columns and len(x):
        out["max_drawdown_date"] = pd.Timestamp(x.loc[worst_idx, "date"]).isoformat()
        underwater = dd < 0
        max_run = run = 0
        for flag in underwater.to_numpy(dtype=bool):
            run = run + 1 if flag else 0
            max_run = max(max_run, run)
        out["max_underwater_sessions"] = int(max_run)
        calendar = []
        by_year = x.groupby(x["date"].dt.year, sort=True)
        for year, g in by_year:
            start_eq = float(g.iloc[0]["equity"])
            end_eq = float(g.iloc[-1]["equity"])
            if start_eq > 0:
                calendar.append({"year": int(year), "return": float(end_eq / start_eq - 1.0)})
        out["calendar_returns"] = calendar
    return out


def selection_lift_diagnostics(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    selected_col: str = "selected_for_signal",
    pass_col: str = "pass",
    realized_col: str = "realized_forward_return",
) -> pd.DataFrame:
    """Monthly selected-vs-passing realized return lift for diagnostic use only."""
    required = {date_col, selected_col, pass_col, realized_col}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    work = frame.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[realized_col] = pd.to_numeric(work[realized_col], errors="coerce")
    rows = []
    for dt, g in work.groupby(date_col, sort=True):
        g = g[g[pass_col].astype(bool) & g[realized_col].notna()].copy()
        selected = g[g[selected_col].astype(bool)]
        other = g[~g[selected_col].astype(bool)]
        if selected.empty:
            continue
        sel_mean = float(selected[realized_col].mean())
        other_mean = float(other[realized_col].mean()) if len(other) else None
        rows.append({
            "date": pd.Timestamp(dt),
            "selected_n": int(len(selected)),
            "other_passing_n": int(len(other)),
            "selected_mean_realized_return": sel_mean,
            "selected_median_realized_return": float(selected[realized_col].median()),
            "selected_win_rate": float((selected[realized_col] > 0).mean()),
            "other_passing_mean_realized_return": other_mean,
            "selection_lift_vs_other_passing": (sel_mean - other_mean) if other_mean is not None else None,
        })
    return pd.DataFrame(rows)
