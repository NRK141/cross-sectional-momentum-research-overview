from __future__ import annotations
import math
from typing import Iterable
import numpy as np
import pandas as pd

def portfolio_metrics(equity: pd.Series, annualization: int = 252) -> dict:
    equity = equity.dropna().astype(float)
    if len(equity) < 2:
        return {"cagr":0.0,"sharpe":0.0,"max_drawdown":0.0,"annualized_volatility":0.0}
    r = equity.pct_change().fillna(0.0)
    years = max((len(equity)-1)/annualization, 1/annualization)
    cagr = float((equity.iloc[-1]/equity.iloc[0])**(1/years)-1) if equity.iloc[0] > 0 else 0.0
    vol = float(r.std(ddof=1)*np.sqrt(annualization)) if len(r)>1 else 0.0
    sharpe = float(r.mean()/r.std(ddof=1)*np.sqrt(annualization)) if r.std(ddof=1)>0 else 0.0
    dd = equity/equity.cummax()-1.0
    return {"cagr":cagr,"sharpe":sharpe,"max_drawdown":float(dd.min()),"annualized_volatility":vol}

def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"closed_trades":0,"win_rate":0.0,"profit_factor":float('nan'),"payoff_ratio":float('nan'),"average_net_trade_return":0.0,"average_winner":float('nan'),"average_loser":float('nan'),"expectancy_per_trade":0.0}
    x = trades["net_return"].astype(float)
    wins=x[x>0]; losses=x[x<=0]
    gross_profit=float(wins.sum())
    gross_loss=float(-losses.sum())
    avg_win=float(wins.mean()) if len(wins) else float('nan')
    avg_loss=float(losses.mean()) if len(losses) else float('nan')
    payoff=float(avg_win/abs(avg_loss)) if len(wins) and len(losses) and avg_loss != 0 else float('nan')
    pf=float(gross_profit/gross_loss) if gross_loss>0 else float('inf')
    return {"closed_trades":int(len(x)),"win_rate":float((x>0).mean()),"profit_factor":pf,"payoff_ratio":payoff,"average_net_trade_return":float(x.mean()),"median_net_trade_return":float(x.median()),"average_winner":avg_win,"average_loser":avg_loss,"expectancy_per_trade":float(x.mean())}

TRADING_DAYS = 252
CALENDAR_DAYS = 365.25


def prepare_equity(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "date" not in x or "equity" not in x:
        raise ValueError("equity curve must contain date and equity columns")
    x["date"] = pd.to_datetime(x["date"], errors="raise")
    x["equity"] = pd.to_numeric(x["equity"], errors="raise")
    x = x.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if len(x) < 2:
        raise ValueError("equity curve requires at least two dated observations")
    if (x["equity"] <= 0).any() or not np.isfinite(x["equity"]).all():
        raise ValueError("equity values must be finite and positive")
    x["daily_return"] = x["equity"].pct_change().fillna(0.0)
    return x


def calendar_cagr_from_equity(df: pd.DataFrame) -> float:
    x = prepare_equity(df)
    elapsed_days = (x["date"].iloc[-1] - x["date"].iloc[0]).days
    years = max(elapsed_days / CALENDAR_DAYS, 1.0 / CALENDAR_DAYS)
    return float((x["equity"].iloc[-1] / x["equity"].iloc[0]) ** (1.0 / years) - 1.0)


def _daily_metrics(returns: pd.Series, wealth: pd.Series, elapsed_years: float) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    w = pd.to_numeric(wealth, errors="coerce").dropna()
    if len(w) < 2:
        return {}
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    vol = sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    sharpe = float(r.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    dd = w / w.cummax() - 1.0
    years = max(elapsed_years, 1.0 / CALENDAR_DAYS)
    cagr = float((w.iloc[-1] / w.iloc[0]) ** (1.0 / years) - 1.0)
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "annualized_volatility": float(vol),
        "max_drawdown": float(dd.min()),
        "total_return": float(w.iloc[-1] / w.iloc[0] - 1.0),
    }


def metrics_from_equity(df: pd.DataFrame) -> dict:
    x = prepare_equity(df)
    years = max((x["date"].iloc[-1] - x["date"].iloc[0]).days / CALENDAR_DAYS, 1.0 / CALENDAR_DAYS)
    return _daily_metrics(x["daily_return"].iloc[1:], x["equity"], years)


def period_metrics(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict:
    """Calculate period metrics while preserving the prior close as the wealth anchor.

    This avoids dropping the first trading day's return when a period starts in the
    middle of the full equity series.
    """
    x = prepare_equity(df)
    start_ts = pd.Timestamp(start) if start else x["date"].iloc[0]
    end_ts = pd.Timestamp(end) if end else x["date"].iloc[-1]
    in_period = x[(x["date"] >= start_ts) & (x["date"] <= end_ts)].copy()
    if in_period.empty:
        return {}

    prior = x[x["date"] < in_period["date"].iloc[0]].tail(1)
    if len(prior):
        wealth = pd.concat([prior[["date", "equity"]], in_period[["date", "equity"]]], ignore_index=True)
        returns = in_period["daily_return"]
        anchor_date = prior["date"].iloc[0]
    else:
        wealth = in_period[["date", "equity"]].copy()
        returns = in_period["daily_return"].iloc[1:]
        anchor_date = wealth["date"].iloc[0]

    years = max((wealth["date"].iloc[-1] - anchor_date).days / CALENDAR_DAYS, 1.0 / CALENDAR_DAYS)
    return _daily_metrics(returns, wealth["equity"], years)


def yearly_returns(df: pd.DataFrame) -> pd.DataFrame:
    x = prepare_equity(df)
    x["year"] = x["date"].dt.year
    rows = []
    for year, g in x.groupby("year", sort=True):
        compounded = float((1.0 + g["daily_return"]).prod() - 1.0)
        rows.append({"year": int(year), "strategy_return": compounded})
    return pd.DataFrame(rows)


def profit_factor(values: Iterable[float]) -> float:
    s = pd.Series(list(values), dtype=float)
    gains = float(s[s > 0].sum())
    losses = float(-s[s < 0].sum())
    return float(gains / losses) if losses > 0 else float("inf")
