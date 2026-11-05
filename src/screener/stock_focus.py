from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.backtest.engine import candidate_ok
from src.common.config import load_config, root_path
from src.common.logging_utils import log
from src.production.identity import RELEASE, ROOT, validate_active_identity

SPEC_PATH = ROOT / "STOCK_SCREENER_SPEC.json"
OUT_DIR = ROOT / "outputs" / "stock_screener"
MANIFEST_PATH = ROOT / "data" / "manifests" / "stock_screener_success.json"


def q(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x


def _load_spec() -> dict[str, Any]:
    if not SPEC_PATH.exists():
        raise FileNotFoundError(f"Missing stock screener specification: {SPEC_PATH}")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("strategy_release") != RELEASE:
        raise RuntimeError(
            f"Stock screener spec targets {spec.get('strategy_release')!r}, expected {RELEASE!r}"
        )
    if int(spec.get("top_n", 0)) != 15:
        raise RuntimeError("F2.2 stock focus screener must remain Top 15")
    if spec.get("monthly_leadership_gate") is not False:
        raise RuntimeError("Stock focus screener must not pre-gate on the existing monthly Top-15 ranking")
    cap = spec.get("capital_model", {})
    initial = float(cap.get("initial_capital_usd", 0.0))
    position = float(cap.get("position_dollars_per_trade", 0.0))
    if initial != 5000.0 or position != 2000.0:
        raise RuntimeError("Patch 2.1.4 stock screener requires $5,000 standalone capital and $2,000 fixed trade notional")
    weights = spec.get("score", {}).get("weights", {})
    if abs(sum(float(v) for v in weights.values()) - 1.0) > 1e-12:
        raise RuntimeError("Stock screener score weights must sum to 1.0")
    conc = spec.get("single_trade_concentration", {})
    if conc.get("enabled") is not True:
        raise RuntimeError("Patch 2.1.4 requires the single-trade concentration adjustment to be enabled")
    threshold = float(conc.get("threshold_strictly_greater_than", float("nan")))
    if not math.isfinite(threshold) or threshold != 0.75:
        raise RuntimeError("Patch 2.1.4 requires a strict single-trade P&L concentration cutoff of 75%")
    if conc.get("recursive") is not False:
        raise RuntimeError("Single-trade concentration adjustment must be single-pass/non-recursive")
    return spec


@dataclass
class ReturnAccumulator:
    annualization: int
    observations: int = 0
    n_returns: int = 0
    mean_return: float = 0.0
    m2_return: float = 0.0
    first_equity: float | None = None
    last_equity: float | None = None
    previous_equity: float | None = None
    peak_equity: float | None = None
    max_drawdown: float = 0.0
    first_date: pd.Timestamp | None = None
    last_date: pd.Timestamp | None = None

    def add(self, date: pd.Timestamp, equity: float) -> None:
        date = pd.Timestamp(date)
        equity = float(equity)
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError(f"Invalid standalone equity {equity} on {date.date()}")

        if self.observations == 0:
            daily_return = 0.0
            self.first_equity = equity
            self.first_date = date
            self.peak_equity = equity
        else:
            assert self.previous_equity is not None
            daily_return = equity / self.previous_equity - 1.0

        self.n_returns += 1
        delta = daily_return - self.mean_return
        self.mean_return += delta / self.n_returns
        self.m2_return += delta * (daily_return - self.mean_return)

        self.peak_equity = max(float(self.peak_equity), equity)
        self.max_drawdown = min(self.max_drawdown, equity / float(self.peak_equity) - 1.0)
        self.previous_equity = equity
        self.last_equity = equity
        self.last_date = date
        self.observations += 1

    def metrics(self) -> dict[str, float]:
        if self.observations < 2 or self.first_equity is None or self.last_equity is None:
            return {
                "cagr": 0.0,
                "sharpe": 0.0,
                "annualized_volatility": 0.0,
                "max_drawdown": float(self.max_drawdown),
                "total_return": 0.0,
            }
        variance = self.m2_return / (self.n_returns - 1) if self.n_returns > 1 else 0.0
        sd = math.sqrt(max(variance, 0.0))
        vol = sd * math.sqrt(self.annualization) if sd > 0 else 0.0
        sharpe = self.mean_return / sd * math.sqrt(self.annualization) if sd > 0 else 0.0
        years = max((self.observations - 1) / self.annualization, 1.0 / self.annualization)
        cagr = (self.last_equity / self.first_equity) ** (1.0 / years) - 1.0
        return {
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "annualized_volatility": float(vol),
            "max_drawdown": float(self.max_drawdown),
            "total_return": float(self.last_equity / self.first_equity - 1.0),
        }


@dataclass
class StockPosition:
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    entry_cost: float
    peak_price: float
    held_days: int = 0


@dataclass
class StandaloneStockState:
    permno: int
    latest_ticker: str
    latest_sector: str
    current_close: float
    cfg: dict[str, Any]
    capital_model: dict[str, Any] | None = None
    veto_entry_dates: set[pd.Timestamp] = field(default_factory=set)
    cash: float = field(init=False)
    fixed_position_dollars: float | None = field(init=False, default=None)
    initial_capital: float = field(init=False)
    position: StockPosition | None = None
    pending_signal: bool = False
    last_exit: pd.Timestamp | None = None
    returns: ReturnAccumulator = field(init=False)
    ranking_returns: ReturnAccumulator | None = field(init=False, default=None)
    ranking_excluded_entry_date: pd.Timestamp | None = None
    ranking_excluded_realized_pnl: float = field(init=False, default=0.0)
    ranking_excluded_active_basis: float | None = field(init=False, default=None)
    trades: list[dict[str, Any]] = field(default_factory=list)
    boundary_entry_rejections: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capital_model is not None:
            self.initial_capital = float(self.capital_model["initial_capital_usd"])
            self.fixed_position_dollars = float(self.capital_model["position_dollars_per_trade"])
            if self.initial_capital <= 0 or self.fixed_position_dollars <= 0:
                raise ValueError("Stock screener capital values must be positive")
            if self.fixed_position_dollars > self.initial_capital:
                raise ValueError("Stock screener position dollars cannot exceed standalone initial capital")
        else:
            self.initial_capital = float(self.cfg["portfolio"]["initial_capital"])
            self.fixed_position_dollars = None
        self.cash = self.initial_capital
        annualization = int(self.cfg["execution"]["annualization_days"])
        self.returns = ReturnAccumulator(annualization)
        if self.ranking_excluded_entry_date is not None:
            self.ranking_excluded_entry_date = pd.Timestamp(self.ranking_excluded_entry_date)
            self.ranking_returns = ReturnAccumulator(annualization)
        self.veto_entry_dates = {pd.Timestamp(x) for x in self.veto_entry_dates}

    @property
    def weight(self) -> float:
        p = self.cfg["portfolio"]
        return min(float(p["base_position_weight"]), float(p["max_position_weight"]))

    @property
    def cost(self) -> float:
        return float(self.cfg["execution"]["transaction_cost_bps_per_side"]) / 10000.0

    def process_row(self, row: Any) -> None:
        date = pd.Timestamp(getattr(row, "date"))
        price = float(getattr(row, "close"))
        if not math.isfinite(price) or price <= 0:
            return

        # Execute a prior-close signal at this security's next observed close.
        if self.pending_signal and self.position is None:
            self.pending_signal = False
            if date in self.veto_entry_dates:
                self.boundary_entry_rejections.append(date.date().isoformat())
            else:
                if self.fixed_position_dollars is not None:
                    # Patch 2.1.4 stock-screener sizing: each symbol is an independent
                    # $5,000 account and each trade targets exactly $2,000 notional.
                    # Fractional-equivalent shares avoid price-level/whole-share utilization bias.
                    target = self.fixed_position_dollars
                else:
                    equity_before = float(self.cash)
                    target = equity_before * self.weight
                notional = min(target, self.cash / (1.0 + self.cost))
                if notional > 0:
                    shares = notional / price
                    fee = notional * self.cost
                    self.cash -= notional + fee
                    self.position = StockPosition(
                        entry_date=date,
                        entry_price=price,
                        shares=shares,
                        entry_cost=fee,
                        peak_price=price,
                        held_days=0,
                    )
                    if self.ranking_excluded_entry_date is not None and date == self.ranking_excluded_entry_date:
                        self.ranking_excluded_active_basis = float(notional + fee)

        # F2.2 exits are evaluated and filled at the same current close.
        if self.position is not None:
            pos = self.position
            ex = self.cfg["exit"]
            pos.held_days += 1
            pos.peak_price = max(pos.peak_price, price)
            reason: str | None = None
            if price <= pos.entry_price * (1.0 - float(ex["initial_stop_pct"])):
                reason = "STOP"
            elif (
                pos.peak_price >= pos.entry_price * (1.0 + float(ex["trailing_activates_at_gain"]))
                and price <= pos.peak_price * (1.0 - float(ex["trailing_stop_pct_from_peak"]))
            ):
                reason = "TRAIL"
            elif (
                pos.held_days >= int(ex["minimum_holding_days_before_trend_exit"])
                and price < float(getattr(row, "sma20"))
            ):
                reason = "TREND_BREAK"
            elif pos.held_days >= int(ex["maximum_holding_days"]):
                reason = "MAX_HOLD"

            if reason is not None:
                proceeds = pos.shares * price
                exit_fee = proceeds * self.cost
                self.cash += proceeds - exit_fee
                gross = price / pos.entry_price - 1.0
                basis = pos.shares * pos.entry_price + pos.entry_cost
                net = (proceeds - exit_fee - basis) / basis
                net_pnl_usd = float(proceeds - exit_fee - basis)
                excluded_from_ranking = bool(
                    self.ranking_excluded_entry_date is not None
                    and pd.Timestamp(pos.entry_date) == self.ranking_excluded_entry_date
                )
                self.trades.append(
                    {
                        "permno": self.permno,
                        "ticker": self.latest_ticker,
                        "sector": self.latest_sector,
                        "entry_date": pos.entry_date,
                        "exit_date": date,
                        "entry_price": pos.entry_price,
                        "entry_notional": float(pos.shares * pos.entry_price),
                        "exit_price": price,
                        "held_days": int(pos.held_days),
                        "gross_return": float(gross),
                        "net_return": float(net),
                        "net_pnl_usd": net_pnl_usd,
                        "exit_reason": reason,
                        "excluded_from_ranking": excluded_from_ranking,
                    }
                )
                if excluded_from_ranking:
                    self.ranking_excluded_realized_pnl += net_pnl_usd
                    self.ranking_excluded_active_basis = None
                self.position = None
                self.last_exit = date

        # Generate a signal from the current close. This screener intentionally does not
        # use the existing monthly leadership Top-15 gate: this per-stock test is what
        # creates the new focus Top-15 list.
        if self.position is None:
            cooldown = int(self.cfg["entry"]["candidate_cooldown_days"])
            cooling = self.last_exit is not None and (date - self.last_exit).days < cooldown
            if not cooling:
                try:
                    if candidate_ok(pd.Series(row._asdict()), self.cfg):
                        self.pending_signal = True
                except Exception:
                    pass

        open_value = self.position.shares * price if self.position is not None else 0.0
        raw_equity = self.cash + open_value
        self.returns.add(date, raw_equity)
        if self.ranking_returns is not None:
            ranking_equity = raw_equity - self.ranking_excluded_realized_pnl
            if (
                self.position is not None
                and self.ranking_excluded_entry_date is not None
                and pd.Timestamp(self.position.entry_date) == self.ranking_excluded_entry_date
                and self.ranking_excluded_active_basis is not None
            ):
                excluded_unrealized_pnl = self.position.shares * price - self.ranking_excluded_active_basis
                ranking_equity -= excluded_unrealized_pnl
            self.ranking_returns.add(date, ranking_equity)

    def concentration_info(self, threshold: float) -> dict[str, Any]:
        total_pnl = float(sum(float(t.get("net_pnl_usd", 0.0)) for t in self.trades))
        winners = [t for t in self.trades if float(t.get("net_pnl_usd", 0.0)) > 0.0]
        largest = max(winners, key=lambda t: float(t["net_pnl_usd"])) if winners else None
        largest_pnl = float(largest["net_pnl_usd"]) if largest is not None else float("nan")
        share = largest_pnl / total_pnl if largest is not None and total_pnl > 0.0 else float("nan")
        triggered = bool(math.isfinite(share) and share > float(threshold))
        return {
            "total_net_trade_pnl_usd": total_pnl,
            "largest_trade_net_pnl_usd": largest_pnl,
            "largest_trade_net_return": float(largest["net_return"]) if largest is not None else float("nan"),
            "largest_trade_pnl_share": float(share),
            "largest_trade_entry_date": pd.Timestamp(largest["entry_date"]) if largest is not None else None,
            "largest_trade_exit_date": pd.Timestamp(largest["exit_date"]) if largest is not None else None,
            "triggered": triggered,
        }

    def result(self) -> dict[str, Any]:
        if self.position is not None:
            raise RuntimeError(f"Cannot score {self.permno}: unresolved terminal position remains")

        raw_pm = self.returns.metrics()
        ranking_pm = self.ranking_returns.metrics() if self.ranking_returns is not None else raw_pm
        raw_net = pd.Series([float(t["net_return"]) for t in self.trades], dtype=float)
        included_trades = [t for t in self.trades if not bool(t.get("excluded_from_ranking", False))]
        net = pd.Series([float(t["net_return"]) for t in included_trades], dtype=float)

        def trade_stats(values: pd.Series) -> tuple[float, float, float, float]:
            wins = values[values > 0]
            losses = values[values <= 0]
            gross_profit = float(wins.sum()) if len(wins) else 0.0
            gross_loss = float(-losses.sum()) if len(losses) else 0.0
            pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else float("nan"))
            avg = float(values.mean()) if len(values) else float("nan")
            median = float(values.median()) if len(values) else float("nan")
            win_rate = float((values > 0).mean()) if len(values) else float("nan")
            return float(pf), avg, median, win_rate

        pf, avg, median, win_rate = trade_stats(net)
        raw_pf, raw_avg, raw_median, raw_win_rate = trade_stats(raw_net)

        mdd = float(ranking_pm["max_drawdown"])
        cagr = float(ranking_pm["cagr"])
        calmar = cagr / abs(mdd) if mdd < 0 else (float("inf") if cagr > 0 else float("nan"))
        raw_mdd = float(raw_pm["max_drawdown"])
        raw_cagr = float(raw_pm["cagr"])
        raw_calmar = raw_cagr / abs(raw_mdd) if raw_mdd < 0 else (float("inf") if raw_cagr > 0 else float("nan"))

        concentration = self.concentration_info(float("inf"))
        excluded = next((t for t in self.trades if bool(t.get("excluded_from_ranking", False))), None)
        first_date = self.returns.first_date.date().isoformat() if self.returns.first_date is not None else None
        last_date = self.returns.last_date.date().isoformat() if self.returns.last_date is not None else None
        return {
            "permno": int(self.permno),
            "ticker": self.latest_ticker,
            "sector": self.latest_sector,
            "current_close": float(self.current_close),
            "screener_initial_capital_usd": float(self.initial_capital),
            "screener_position_dollars_per_trade": float(self.fixed_position_dollars) if self.fixed_position_dollars is not None else float("nan"),
            "screener_position_fraction_of_initial": (float(self.fixed_position_dollars) / float(self.initial_capital)) if self.fixed_position_dollars is not None else float(self.weight),
            "history_start": first_date,
            "history_end": last_date,
            "history_sessions": int(self.returns.observations),
            **ranking_pm,
            "calmar": float(calmar),
            "closed_trades": int(len(net)),
            "profit_factor": float(pf),
            "average_net_trade_return": avg,
            "median_net_trade_return": median,
            "win_rate": win_rate,
            "raw_cagr": raw_cagr,
            "raw_sharpe": float(raw_pm["sharpe"]),
            "raw_annualized_volatility": float(raw_pm["annualized_volatility"]),
            "raw_max_drawdown": raw_mdd,
            "raw_total_return": float(raw_pm["total_return"]),
            "raw_calmar": float(raw_calmar),
            "raw_closed_trades": int(len(raw_net)),
            "raw_profit_factor": float(raw_pf),
            "raw_average_net_trade_return": raw_avg,
            "raw_median_net_trade_return": raw_median,
            "raw_win_rate": raw_win_rate,
            "total_net_trade_pnl_usd": float(concentration["total_net_trade_pnl_usd"]),
            "largest_trade_net_pnl_usd": float(concentration["largest_trade_net_pnl_usd"]),
            "largest_trade_net_return": float(concentration["largest_trade_net_return"]),
            "largest_trade_pnl_share": float(concentration["largest_trade_pnl_share"]),
            "concentration_adjusted": bool(excluded is not None),
            "excluded_trade_entry_date": pd.Timestamp(excluded["entry_date"]).date().isoformat() if excluded is not None else None,
            "excluded_trade_exit_date": pd.Timestamp(excluded["exit_date"]).date().isoformat() if excluded is not None else None,
            "excluded_trade_net_return": float(excluded["net_return"]) if excluded is not None else float("nan"),
            "excluded_trade_net_pnl_usd": float(excluded["net_pnl_usd"]) if excluded is not None else float("nan"),
            "boundary_entry_rejections": int(len(self.boundary_entry_rejections)),
        }


def simulate_stock_frame(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    permno: int,
    latest_ticker: str,
    latest_sector: str = "Other",
    current_close: float | None = None,
    veto_entry_dates: Iterable[pd.Timestamp | str] = (),
    capital_model: dict[str, Any] | None = None,
    ranking_excluded_entry_date: pd.Timestamp | str | None = None,
) -> StandaloneStockState:
    x = frame.copy()
    x["date"] = pd.to_datetime(x["date"], errors="raise")
    x = x.sort_values("date", kind="mergesort")
    if x.empty:
        raise ValueError("Cannot simulate an empty stock frame")
    if current_close is None:
        current_close = float(x["close"].iloc[-1])
    state = StandaloneStockState(
        int(permno),
        str(latest_ticker),
        str(latest_sector),
        float(current_close),
        cfg,
        capital_model=capital_model,
        veto_entry_dates={pd.Timestamp(v) for v in veto_entry_dates},
        ranking_excluded_entry_date=(pd.Timestamp(ranking_excluded_entry_date) if ranking_excluded_entry_date is not None else None),
    )
    for row in x.itertuples(index=False):
        state.process_row(row)
    return state


def _configure_duckdb(con: Any, cfg: dict[str, Any]) -> None:
    comp = cfg.get("compute", {})
    con.execute(f"SET threads={int(comp.get('threads', 4))}")
    if comp.get("memory_limit"):
        con.execute(f"SET memory_limit={q(comp['memory_limit'])}")
    temp = root_path(cfg, cfg["data"]["temp_dir"])
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory={q(str(temp))}")
    con.execute(
        "SET preserve_insertion_order="
        + ("true" if comp.get("preserve_insertion_order", False) else "false")
    )


def _preflight_feature_store(con: Any, glob: str, evaluation_start: pd.Timestamp) -> pd.Timestamp:
    required = {
        "date", "permno", "ticker", "sector", "eligible", "close", "ret1", "ret5",
        "rsi2", "distance_20d_high", "sma20", "sma50",
    }
    cur = con.execute(
        f"SELECT * FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true) LIMIT 0"
    )
    cols = {d[0] for d in cur.description}
    missing = sorted(required.difference(cols))
    if missing:
        raise RuntimeError(f"Eligible feature store is missing stock-screener columns: {missing}")

    dup = int(con.execute(f"""
        SELECT COUNT(*) FROM (
          SELECT date, permno, COUNT(*) c
          FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true)
          WHERE date >= DATE {q(evaluation_start.date())}
          GROUP BY date, permno
          HAVING COUNT(*) > 1
        )
    """).fetchone()[0])
    if dup:
        raise RuntimeError(f"Eligible feature store contains {dup:,} duplicate date/PERMNO keys")

    latest = con.execute(
        f"SELECT MAX(date) FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true)"
    ).fetchone()[0]
    if latest is None:
        raise RuntimeError("Eligible feature store contains no rows")
    latest_ts = pd.Timestamp(latest)
    if latest_ts < evaluation_start:
        raise RuntimeError(f"Latest feature date {latest_ts.date()} predates evaluation start")
    return latest_ts


def _current_universe(con: Any, glob: str, latest: pd.Timestamp) -> pd.DataFrame:
    current = con.execute(f"""
        SELECT CAST(permno AS BIGINT) AS permno,
               CAST(ticker AS VARCHAR) AS ticker,
               CAST(COALESCE(sector, 'Other') AS VARCHAR) AS sector,
               CAST(close AS DOUBLE) AS current_close
        FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true)
        WHERE date = DATE {q(latest.date())} AND eligible
        ORDER BY permno
    """).fetchdf()
    if current.empty:
        raise RuntimeError(f"No eligible stocks exist on latest feature date {latest.date()}")
    if current["permno"].duplicated().any():
        raise RuntimeError("Latest-date eligible universe contains duplicate PERMNO rows")
    current["permno"] = current["permno"].astype(int)
    current["ticker"] = current["ticker"].astype(str)
    current["sector"] = current["sector"].fillna("Other").astype(str)
    return current


def _stream_first_pass(
    con: Any,
    glob: str,
    current: pd.DataFrame,
    cfg: dict[str, Any],
    start: pd.Timestamp,
    latest: pd.Timestamp,
    capital_model: dict[str, Any],
) -> dict[int, StandaloneStockState]:
    states = {
        int(r.permno): StandaloneStockState(
            int(r.permno), str(r.ticker), str(r.sector), float(r.current_close), cfg, capital_model=capital_model
        )
        for r in current.itertuples(index=False)
    }
    current_ids = current[["permno"]].copy()
    con.register("stock_focus_current_universe", current_ids)

    cols = "date,permno,ticker,sector,eligible,close,ret1,ret5,rsi2,distance_20d_high,sma20,sma50"
    for year in range(start.year, latest.year + 1):
        lo = max(start, pd.Timestamp(f"{year}-01-01"))
        hi = min(latest, pd.Timestamp(f"{year}-12-31"))
        log(f"Stock screener: evaluating {year} ({lo.date()} through {hi.date()})...")
        frame = con.execute(f"""
            SELECT {cols}
            FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true) f
            INNER JOIN stock_focus_current_universe u USING (permno)
            WHERE date BETWEEN DATE {q(lo.date())} AND DATE {q(hi.date())}
            ORDER BY date, permno
        """).fetchdf()
        if frame.empty:
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        processed = 0
        for row in frame.itertuples(index=False):
            state = states.get(int(row.permno))
            if state is not None:
                state.process_row(row)
                processed += 1
        log(f"Stock screener: {year} complete - {processed:,} security-session rows.")
        del frame
    con.unregister("stock_focus_current_universe")
    return states


def _rerun_unresolved(
    con: Any,
    glob: str,
    states: dict[int, StandaloneStockState],
    current: pd.DataFrame,
    cfg: dict[str, Any],
    start: pd.Timestamp,
    latest: pd.Timestamp,
    capital_model: dict[str, Any],
) -> None:
    unresolved = [s for s in states.values() if s.position is not None]
    if not unresolved:
        return
    log(
        f"Stock screener right boundary: {len(unresolved)} unresolved standalone position(s); "
        "rejecting only those terminal entries and rerunning affected stocks."
    )
    meta = current.set_index("permno")
    cols = "date,permno,ticker,sector,eligible,close,ret1,ret5,rsi2,distance_20d_high,sma20,sma50"

    for original in unresolved:
        permno = int(original.permno)
        frame = con.execute(f"""
            SELECT {cols}
            FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true)
            WHERE permno = {permno}
              AND date BETWEEN DATE {q(start.date())} AND DATE {q(latest.date())}
            ORDER BY date
        """).fetchdf()
        if frame.empty:
            raise RuntimeError(f"Cannot rerun unresolved PERMNO {permno}: history query is empty")
        veto: set[pd.Timestamp] = set()
        clean: StandaloneStockState | None = None
        for _ in range(12):
            trial = simulate_stock_frame(
                frame,
                cfg,
                permno=permno,
                latest_ticker=str(meta.loc[permno, "ticker"]),
                latest_sector=str(meta.loc[permno, "sector"]),
                current_close=float(meta.loc[permno, "current_close"]),
                veto_entry_dates=veto,
                capital_model=capital_model,
            )
            if trial.position is None:
                clean = trial
                break
            entry_date = pd.Timestamp(trial.position.entry_date)
            if entry_date in veto:
                raise RuntimeError(f"Boundary rerun for {permno} made no progress")
            veto.add(entry_date)
        if clean is None:
            raise RuntimeError(f"Exceeded right-boundary normalization passes for PERMNO {permno}")
        clean.boundary_entry_rejections = [d.date().isoformat() for d in sorted(veto)]
        states[permno] = clean
        log(
            f"Stock screener right boundary: {meta.loc[permno, 'ticker']} / {permno} "
            f"rejected {len(veto)} unresolved terminal entr{'y' if len(veto)==1 else 'ies'}."
        )



def _apply_single_trade_concentration_adjustment(
    con: Any,
    glob: str,
    states: dict[int, StandaloneStockState],
    current: pd.DataFrame,
    cfg: dict[str, Any],
    start: pd.Timestamp,
    latest: pd.Timestamp,
    capital_model: dict[str, Any],
    spec: dict[str, Any],
) -> int:
    """Neutralize one over-concentrated winner from ranking metrics, once per stock.

    The raw trade remains in the audit log and the strategy chronology is preserved.
    Only the ranking equity accumulator neutralizes the selected trade's mark-to-market
    and realized P&L. This is deliberately non-recursive.
    """
    conc = spec["single_trade_concentration"]
    threshold = float(conc["threshold_strictly_greater_than"])
    triggered: dict[int, dict[str, Any]] = {}
    for permno, state in states.items():
        info = state.concentration_info(threshold)
        if bool(info["triggered"]):
            triggered[int(permno)] = info

    if not triggered:
        log(
            f"Stock screener concentration adjustment: 0 stocks exceeded the strict "
            f"{threshold:.0%} single-trade P&L cutoff."
        )
        return 0

    log(
        f"Stock screener concentration adjustment: {len(triggered):,} stock(s) exceeded the strict "
        f"{threshold:.0%} single-trade P&L cutoff; recomputing ranking metrics with exactly one trade neutralized per stock."
    )

    meta = current.set_index("permno")
    adjusted: dict[int, StandaloneStockState] = {}
    for permno, info in triggered.items():
        original = states[permno]
        adjusted[permno] = StandaloneStockState(
            permno=permno,
            latest_ticker=str(meta.loc[permno, "ticker"]),
            latest_sector=str(meta.loc[permno, "sector"]),
            current_close=float(meta.loc[permno, "current_close"]),
            cfg=cfg,
            capital_model=capital_model,
            veto_entry_dates={pd.Timestamp(x) for x in original.boundary_entry_rejections},
            ranking_excluded_entry_date=pd.Timestamp(info["largest_trade_entry_date"]),
        )

    ids = pd.DataFrame({"permno": sorted(adjusted)})
    con.register("stock_focus_concentration_ids", ids)
    cols = "date,permno,ticker,sector,eligible,close,ret1,ret5,rsi2,distance_20d_high,sma20,sma50"
    try:
        for year in range(start.year, latest.year + 1):
            lo = max(start, pd.Timestamp(f"{year}-01-01"))
            hi = min(latest, pd.Timestamp(f"{year}-12-31"))
            frame = con.execute(f"""
                SELECT {cols}
                FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true) f
                INNER JOIN stock_focus_concentration_ids u USING (permno)
                WHERE date BETWEEN DATE {q(lo.date())} AND DATE {q(hi.date())}
                ORDER BY date, permno
            """).fetchdf()
            if frame.empty:
                continue
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            for row in frame.itertuples(index=False):
                state = adjusted.get(int(row.permno))
                if state is not None:
                    state.process_row(row)
            del frame
    finally:
        con.unregister("stock_focus_concentration_ids")

    for permno, state in adjusted.items():
        original = states[permno]
        if state.position is not None:
            raise RuntimeError(
                f"Concentration-adjusted rerun left unresolved terminal position for PERMNO {permno}"
            )
        original_sig = [
            (
                pd.Timestamp(t["entry_date"]),
                pd.Timestamp(t["exit_date"]),
                str(t["exit_reason"]),
                round(float(t["net_return"]), 12),
            )
            for t in original.trades
        ]
        adjusted_sig = [
            (
                pd.Timestamp(t["entry_date"]),
                pd.Timestamp(t["exit_date"]),
                str(t["exit_reason"]),
                round(float(t["net_return"]), 12),
            )
            for t in state.trades
        ]
        if original_sig != adjusted_sig:
            raise RuntimeError(
                f"Concentration adjustment changed raw F2.2 trade chronology for "
                f"{state.latest_ticker} / {permno}; refusing to rank."
            )
        excluded = [t for t in state.trades if bool(t.get("excluded_from_ranking", False))]
        if len(excluded) != 1:
            raise RuntimeError(
                f"Concentration adjustment for {state.latest_ticker} / {permno} expected exactly "
                f"one excluded ranking trade, got {len(excluded)}."
            )
        info = triggered[permno]
        if pd.Timestamp(excluded[0]["entry_date"]) != pd.Timestamp(info["largest_trade_entry_date"]):
            raise RuntimeError(
                f"Concentration adjustment excluded the wrong trade for {state.latest_ticker} / {permno}"
            )
        state.boundary_entry_rejections = list(original.boundary_entry_rejections)
        states[permno] = state

    return len(adjusted)

def _filter_and_rank(results: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the Patch 2.1.4 original-screener-style filters and ranking.

    The only intentional methodological adaptations versus the user's reference screener are:
    1) F2.2's current eligible universe replaces a public index universe; and
    2) the low-frequency F2.2 strategy uses a 20-trade minimum instead of >100 trades.
    """
    f = spec["minimum_evidence"]
    x = results.copy()

    def failures(r: pd.Series) -> str:
        bad: list[str] = []
        if int(r["closed_trades"]) < int(f["closed_trades_minimum"]):
            bad.append("closed_trades")
        if not _safe_float(r["cagr"]) > float(f["cagr_strictly_greater_than"]):
            bad.append("cagr")
        if not _safe_float(r["sharpe"]) > float(f["sharpe_strictly_greater_than"]):
            bad.append("sharpe")
        if not _safe_float(r["calmar"]) > float(f["calmar_strictly_greater_than"]):
            bad.append("calmar")
        return ";".join(bad)

    x["filter_failures"] = x.apply(failures, axis=1)
    x["passes_filters"] = x["filter_failures"].eq("")

    # Patch 2.1.4: print a sequential hard-filter waterfall before ranking so a
    # fail-closed run explains exactly where the candidate population disappeared.
    trades_ok = pd.to_numeric(x["closed_trades"], errors="coerce").fillna(-1) >= int(f["closed_trades_minimum"])
    cagr_ok = pd.to_numeric(x["cagr"], errors="coerce") > float(f["cagr_strictly_greater_than"])
    calmar_ok = pd.to_numeric(x["calmar"], errors="coerce") > float(f["calmar_strictly_greater_than"])
    sharpe_ok = pd.to_numeric(x["sharpe"], errors="coerce") > float(f["sharpe_strictly_greater_than"])
    n0 = int(len(x))
    n1 = int(trades_ok.sum())
    n2 = int((trades_ok & cagr_ok).sum())
    n3 = int((trades_ok & cagr_ok & calmar_ok).sum())
    n4 = int((trades_ok & cagr_ok & calmar_ok & sharpe_ok).sum())
    log(
        "Stock screener filter waterfall: "
        f"evaluated={n0:,} -> trades>={int(f['closed_trades_minimum'])}: {n1:,} "
        f"-> +CAGR>{float(f['cagr_strictly_greater_than']):g}: {n2:,} "
        f"-> +Calmar>{float(f['calmar_strictly_greater_than']):g}: {n3:,} "
        f"-> +Sharpe>{float(f['sharpe_strictly_greater_than']):g}: {n4:,}."
    )

    eligible = x.loc[x["passes_filters"]].copy()
    if len(eligible) < int(spec["top_n"]):
        raise RuntimeError(
            f"Only {len(eligible)} stocks passed stock-screener hard filters; "
            f"need at least {spec['top_n']} for a Top-{spec['top_n']} list. "
            "Do not silently fill the list with failed stocks. "
            "No stock-screener output files were replaced; the previous successful outputs remain intact."
        )

    def normalize(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        lo = float(values.min())
        hi = float(values.max())
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise RuntimeError("Cannot min-max normalize non-finite stock-screener metric")
        if hi <= lo:
            # A constant metric contains no cross-sectional ranking information.
            return pd.Series(0.0, index=values.index, dtype=float)
        return (values - lo) / (hi - lo)

    weights = spec["score"]["weights"]
    for col in ("sharpe", "calmar", "cagr", "closed_trades"):
        eligible[f"{col}_norm"] = normalize(eligible[col])

    eligible["composite_score"] = (
        eligible["sharpe_norm"] * float(weights["sharpe"])
        + eligible["calmar_norm"] * float(weights["calmar"])
        + eligible["cagr_norm"] * float(weights["cagr"])
        + eligible["closed_trades_norm"] * float(weights["closed_trades"])
    )
    eligible = eligible.sort_values(
        ["composite_score", "sharpe", "calmar", "cagr", "closed_trades", "permno"],
        ascending=[False, False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    eligible["rank"] = np.arange(1, len(eligible) + 1, dtype=int)
    return x, eligible

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _format_rank_view(frame: pd.DataFrame, *, include_score: bool) -> pd.DataFrame:
    cols = ["ticker"]
    if include_score:
        cols.append("composite_score")
    cols += ["sharpe", "calmar", "cagr", "closed_trades"]
    view = frame[cols].copy()
    rename = {
        "ticker": "Ticker", "composite_score": "Composite Score", "sharpe": "Sharpe",
        "calmar": "Calmar", "cagr": "CAGR", "closed_trades": "Num Trades",
    }
    view = view.rename(columns=rename)
    if "Composite Score" in view:
        view["Composite Score"] = pd.to_numeric(view["Composite Score"], errors="coerce").map(lambda v: f"{v:.4f}")
    for c in ["Sharpe", "Calmar"]:
        view[c] = pd.to_numeric(view[c], errors="coerce").map(lambda v: f"{v:.3f}" if math.isfinite(v) else "inf")
    view["CAGR"] = pd.to_numeric(view["CAGR"], errors="coerce").map(lambda v: f"{v:.2%}" if math.isfinite(v) else "n/a")
    return view


def _print_top15(top: pd.DataFrame, ranked: pd.DataFrame, latest: pd.Timestamp, start: pd.Timestamp, spec: dict[str, Any]) -> None:
    cap = spec["capital_model"]
    print("=" * 112)
    print("FOUNDATION 2.2 - STOCK FOCUS SCREENER - ORIGINAL METHODOLOGY / PATCH 2.1.4")
    print(f"Historical evaluation: {start.date()} through {latest.date()} | Current universe: latest-date eligible stocks")
    print(
        f"Standalone capital: ${cap['initial_capital_usd']:,.0f} | Fixed trade notional: "
        f"${cap['position_dollars_per_trade']:,.0f} ({cap['fraction_of_initial_capital']:.0%} of initial capital)"
    )
    conc = spec["single_trade_concentration"]
    print("Purpose: stock selection only - NO entry/exit signals")
    print(
        f"Single-trade concentration rule: exclude one largest winner from ranking metrics if it contributes "
        f"> {float(conc['threshold_strictly_greater_than']):.0%} of total net stock P&L (single-pass)."
    )
    print("=" * 112)
    print("=== TOP 15 STOCKS (COMPOSITE RANKING) ===")
    print(_format_rank_view(top, include_score=True).to_string(index=False))
    print("\n=== TOP 20 STOCKS (SHARPE) ===")
    print(_format_rank_view(ranked.sort_values("sharpe", ascending=False).head(20), include_score=False).to_string(index=False))
    print("\n=== TOP 20 STOCKS (CAGR) ===")
    print(_format_rank_view(ranked.sort_values("cagr", ascending=False).head(20), include_score=False).to_string(index=False))
    print("=" * 112)
    print(f"Top-15 CSV: {OUT_DIR / 'top_15_stocks.csv'}")
    print(f"Ticker list: {OUT_DIR / 'top_15_tickers.txt'}")
    print("=" * 112)

def main() -> None:
    identity = validate_active_identity()
    if not identity.get("pass", False):
        raise RuntimeError(
            "F2.2 production identity check failed. The stock screener refuses to run against a modified config/engine. "
            + json.dumps(identity.get("checks", {}), sort_keys=True)
        )
    cfg = load_config()
    spec = _load_spec()
    start = max(pd.Timestamp(spec["evaluation_start"]), pd.Timestamp(cfg["data"]["score_start_date"]))
    feat = root_path(cfg, cfg["data"]["feature_dir"]) / "eligible"
    if not feat.exists() or not any(feat.rglob("*.parquet")):
        raise FileNotFoundError(f"Eligible feature store not found under {feat}. Run 01_build_data.cmd first.")

    import duckdb

    con = duckdb.connect()
    _configure_duckdb(con, cfg)
    glob = str(feat).replace("\\", "/") + "/**/*.parquet"
    log("F2.2 stock screener preflight: validating production identity and feature store...")
    latest = _preflight_feature_store(con, glob, start)
    current = _current_universe(con, glob, latest)
    log(
        f"F2.2 stock screener preflight PASS: {len(current):,} currently eligible stocks on {latest.date()}; "
        f"historical evaluation begins {start.date()}."
    )

    capital_model = spec["capital_model"]
    states = _stream_first_pass(con, glob, current, cfg, start, latest, capital_model)
    _rerun_unresolved(con, glob, states, current, cfg, start, latest, capital_model)
    concentration_adjusted_count = _apply_single_trade_concentration_adjustment(
        con, glob, states, current, cfg, start, latest, capital_model, spec
    )

    result_rows: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    for state in states.values():
        result_rows.append(state.result())
        all_trades.extend(state.trades)
    results = pd.DataFrame(result_rows)
    all_results, ranked = _filter_and_rank(results, spec)
    top = ranked.head(int(spec["top_n"])).copy()

    # Replace the stock-screener output atomically at directory granularity. Production
    # monthly rankings and backtest outputs are intentionally untouched.
    stage = OUT_DIR.with_name(OUT_DIR.name + ".tmp")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(all_results.sort_values(["passes_filters", "ticker"], ascending=[False, True]), stage / "all_stock_results.csv")
    _atomic_write_csv(ranked, stage / "eligible_ranked_stocks.csv")
    _atomic_write_csv(top, stage / "top_15_stocks.csv")
    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values(["ticker", "entry_date", "exit_date"], kind="mergesort")
    _atomic_write_csv(trades, stage / "standalone_trade_log.csv")
    (stage / "top_15_tickers.txt").write_text(
        "\n".join(top["ticker"].astype(str).tolist()) + "\n", encoding="utf-8"
    )

    if OUT_DIR.exists():
        backup = OUT_DIR.with_name(OUT_DIR.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        OUT_DIR.replace(backup)
        try:
            stage.replace(OUT_DIR)
        except Exception:
            if backup.exists() and not OUT_DIR.exists():
                backup.replace(OUT_DIR)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        stage.replace(OUT_DIR)

    manifest = {
        "schema_version": 1,
        "feature": spec["feature"],
        "strategy_release": RELEASE,
        "completed_at": datetime.now().isoformat(),
        "evaluation_start": start.date().isoformat(),
        "latest_feature_date": latest.date().isoformat(),
        "current_universe_count": int(len(current)),
        "minimum_evidence_pass_count": int(len(ranked)),
        "top_n": int(spec["top_n"]),
        "monthly_leadership_gate": False,
        "selection_only": True,
        "capital_model": spec["capital_model"],
        "minimum_evidence": spec["minimum_evidence"],
        "score": spec["score"],
        "single_trade_concentration": spec["single_trade_concentration"],
        "concentration_adjusted_stock_count": int(concentration_adjusted_count),
        "signals_generated": False,
        "boundary_entry_rejections": int(pd.to_numeric(all_results["boundary_entry_rejections"], errors="coerce").fillna(0).sum()),
        "top_15_tickers": top["ticker"].astype(str).tolist(),
        "outputs": spec["outputs"],
        "production_config_sha256": identity.get("config_sha256"),
        "production_engine_sha256": identity.get("engine_sha256"),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_manifest.replace(MANIFEST_PATH)

    _print_top15(top, ranked, latest, start, spec)
    log(
        f"F2.2 stock screener complete: {len(current):,} current stocks evaluated, "
        f"{len(ranked):,} passed evidence filters, Top {len(top)} written."
    )


if __name__ == "__main__":
    main()
