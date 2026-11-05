from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import yaml

from stochastic_core import sha256_file

EXPECTED_CONFIG_SHA256 = "00c5f121a08bcb90ab85073f2e2a060ffafd57d92c37e2a047a93e488b77eca2"
EXPECTED_ENGINE_SHA256 = "28d4ea2900c0344947cb924e1554b62c8dd95b0b957cebf8fbed7dd84f169a51"


def _economic_price_index(raw: np.ndarray, ret: np.ndarray, dates: np.ndarray, permno: int):
    """Reinvested wealth-price index; never confuse nominal split quotes with loss.

    Source RET must already include split/distribution economics (CRSP convention).
    When RET is unavailable, raw quote changes are used only for ordinary <=20%
    changes; larger discrepancies are explicitly flagged and left unpriceable.
    """
    raw=np.asarray(raw,dtype=float); ret=np.asarray(ret,dtype=float)
    econ=np.full(len(raw),np.nan,dtype=float)
    events=[]; unresolved=[]
    if not len(raw):return econ,events,unresolved
    for i in range(len(raw)):
        price=float(raw[i]); r=float(ret[i])
        if not np.isfinite(price) or price <= 0:
            continue
        if i == 0 or not np.isfinite(econ[i-1]) or econ[i-1] <= 0:
            # A new continuous wealth segment can start here, but never
            # fabricate a return across a missing/unknown corporate action.
            econ[i] = price
            continue
        prevraw=float(raw[i-1])
        rawratio=price/prevraw if np.isfinite(prevraw) and prevraw>0 else np.nan
        if np.isfinite(r) and r > -1.0 and r < 100.0:
            econ[i]=econ[i-1]*(1.0+r)
            if np.isfinite(rawratio) and rawratio>0 and abs(rawratio-(1.0+r))>0.25:
                events.append({'permno':int(permno),'date':str(pd.Timestamp(dates[i]).date()),
                               'raw_price_ratio':float(rawratio),'economic_return':float(r),
                               'return_to_raw_ratio':float((1.0+r)/rawratio),
                               'classification':'RAW_RETURN_DISCONTINUITY_SPLIT_OR_DISTRIBUTION'})
        elif not np.isfinite(r) and np.isfinite(rawratio) and abs(rawratio-1.0)<=0.20:
            # Neutral fallback for normal small changes only, flagged separately.
            econ[i]=econ[i-1]*rawratio
        else:
            unresolved.append({'permno':int(permno),'date':str(pd.Timestamp(dates[i]).date()),
                               'raw_price_ratio':float(rawratio) if np.isfinite(rawratio) else None,
                               'return':float(r) if np.isfinite(r) else None})
    return econ,events,unresolved


@dataclass
class HistoryStore:
    dates: dict[int, np.ndarray]
    returns: dict[int, np.ndarray]
    closes: dict[int, np.ndarray]
    corporate_action_events: list[dict] | None = None
    unresolved_return_events: list[dict] | None = None

    @classmethod
    def from_daily(cls, daily: pd.DataFrame) -> "HistoryStore":
        return cls.from_frames([daily])

    @classmethod
    def from_frames(cls, frames) -> "HistoryStore":
        """Build a compact per-security history store from bounded DataFrame chunks.

        Patch 2.6.1 uses this instead of materializing the 20M+ row feature table.
        Frames may overlap; later frames win on duplicate PERMNO/date keys, which
        preserves the canonical feature-store return when calibration-only CRSP
        history is supplied first.
        """
        dparts: dict[int, list[np.ndarray]] = {}
        rparts: dict[int, list[np.ndarray]] = {}
        cparts: dict[int, list[np.ndarray]] = {}
        for frame in frames:
            if frame is None or len(frame) == 0:
                continue
            need = {"date", "permno", "ret1", "close"}
            missing = need.difference(frame.columns)
            if missing:
                raise ValueError(f"History frame missing columns: {sorted(missing)}")
            x = frame[["date", "permno", "ret1", "close"]].copy()
            x["date"] = pd.to_datetime(x["date"])
            x["permno"] = pd.to_numeric(x["permno"], errors="coerce")
            x = x.dropna(subset=["date", "permno"]).sort_values(["permno", "date"])
            for permno, g in x.groupby("permno", sort=False):
                key = int(permno)
                dparts.setdefault(key, []).append(g["date"].to_numpy(dtype="datetime64[ns]"))
                rparts.setdefault(key, []).append(pd.to_numeric(g["ret1"], errors="coerce").to_numpy(dtype=float))
                cparts.setdefault(key, []).append(pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=float))
            del x

        dates: dict[int, np.ndarray] = {}
        returns: dict[int, np.ndarray] = {}
        closes: dict[int, np.ndarray] = {}
        action_events: list[dict] = []
        unresolved: list[dict] = []
        for key in dparts:
            d = np.concatenate(dparts[key]) if len(dparts[key]) > 1 else dparts[key][0]
            r = np.concatenate(rparts[key]) if len(rparts[key]) > 1 else rparts[key][0]
            c = np.concatenate(cparts[key]) if len(cparts[key]) > 1 else cparts[key][0]
            order = np.argsort(d, kind="stable")
            d, r, c = d[order], r[order], c[order]
            if len(d) > 1:
                # Stable sort + keep-last means later frames win duplicate dates.
                keep = np.ones(len(d), dtype=bool)
                keep[:-1] = d[:-1] != d[1:]
                d, r, c = d[keep], r[keep], c[keep]
            # CRSP RET is the economic total return across splits and distributions.
            # Build a reinvested-wealth price index, anchored at the first raw close.
            # Synthetic units replace nominal shares throughout screen/exact replay.
            # Do not change the return series used by Heston calibration.
            econ, actions, unknown = _economic_price_index(c, r, d, key)
            dates[key], returns[key], closes[key] = d, r, econ
            action_events.extend(actions)
            unresolved.extend(unknown)
        return cls(dates=dates, returns=returns, closes=closes,
                   corporate_action_events=action_events, unresolved_return_events=unresolved)

    def economic_price(self, permno: int, date: pd.Timestamp) -> float | None:
        p = int(permno)
        if p not in self.dates:
            return None
        d = np.datetime64(pd.Timestamp(date).to_datetime64())
        i = int(np.searchsorted(self.dates[p], d, side="left"))
        if i >= len(self.dates[p]) or self.dates[p][i] != d:
            return None
        v = float(self.closes[p][i])
        return v if np.isfinite(v) and v > 0 else None

    def _signal_prices(self, permno: int, date: pd.Timestamp) -> tuple[np.ndarray, int] | None:
        p = int(permno)
        if p not in self.dates:
            return None
        d = np.datetime64(pd.Timestamp(date).to_datetime64())
        i = int(np.searchsorted(self.dates[p], d, side="left"))
        if i >= len(self.dates[p]) or self.dates[p][i] != d:
            return None
        return self.closes[p], i

    def repaired_signal_features(self, permno: int, date: pd.Timestamp) -> dict | None:
        """Split-neutral trend/location/ret5 and RSI2 from known economic closes.

        At time t, only historical closes through t are consulted. The eligibility
        flag remains an upstream feature-store input, explicitly audited elsewhere.
        """
        pair = self._signal_prices(permno, date)
        if pair is None:
            return None
        values, i = pair
        if i < 200 or not np.isfinite(values[i-200:i+1]).all() or (values[i-200:i+1] <= 0).any():
            return None
        v = values[i-200:i+1]
        changes = np.diff(v) / v[:-1]
        delta = np.diff(v[-61:])
        gains = np.maximum(delta, 0.0)
        losses = np.maximum(-delta, 0.0)
        ag = float(gains[0]); al = float(losses[0])
        for gain, loss in zip(gains[1:], losses[1:]):
            ag = 0.5 * float(gain) + 0.5 * ag
            al = 0.5 * float(loss) + 0.5 * al
        rsi = 100.0 * ag / (ag+al) if ag+al > 0 else 50.0
        return {
            'close': float(v[-1]),
            'sma20': float(v[-20:].mean()),
            'sma50': float(v[-50:].mean()),
            'sma100': float(v[-100:].mean()),
            'sma200': float(v[-200:].mean()),
            'ret5': float(v[-1]/v[-6]-1.0),
            'rsi2': float(rsi),
            'distance_20d_high': float(v[-1]/v[-20:].max()-1.0),
        }

    def economic_breadth(self, dates) -> pd.DataFrame:
        """All-universe, point-in-time breadth from the same economic price basis."""
        target = np.asarray(pd.to_datetime(list(dates)), dtype='datetime64[ns]')
        n = len(target)
        valid = np.zeros(n, dtype=np.int64)
        above100 = np.zeros(n, dtype=np.int64)
        above200 = np.zeros(n, dtype=np.int64)
        pos5 = np.zeros(n, dtype=np.int64)
        for perm, ds in self.dates.items():
            if len(ds) < 200:
                continue
            idx = np.searchsorted(ds, target, side='left')
            ok = idx < len(ds)
            if not ok.any(): continue
            ti = np.where(ok)[0]
            ii = idx[ti]
            same = ds[ii] == target[ti]
            ti, ii = ti[same], ii[same]
            ok2 = ii >= 200
            ti, ii = ti[ok2], ii[ok2]
            if not len(ti): continue
            p = self.closes[perm]
            good = np.isfinite(p) & (p > 0)
            cs = np.r_[0.0, np.cumsum(np.where(good, p, 0.0))]
            cn = np.r_[0, np.cumsum(good.astype(np.int64))]
            ok3 = (cn[ii+1] - cn[ii-199] == 200)
            ti, ii = ti[ok3], ii[ok3]
            if not len(ti): continue
            sma100 = (cs[ii+1]-cs[ii-99])/100.0
            sma200 = (cs[ii+1]-cs[ii-199])/200.0
            valid[ti] += 1
            above100[ti] += (p[ii] > sma100).astype(np.int64)
            above200[ti] += (p[ii] > sma200).astype(np.int64)
            pos5[ti] += (p[ii] > p[ii-5]).astype(np.int64)
        denom = np.maximum(valid, 1)
        return pd.DataFrame({'date':pd.to_datetime(target),
                             'breadth_above_sma200': np.where(valid>0,above200/denom,np.nan),
                             'breadth_above_sma100':np.where(valid>0,above100/denom,np.nan),
                             'breadth_ret5_positive':np.where(valid>0,pos5/denom,np.nan),
                             'universe_rows':valid})

    def trailing_returns(self, permno: int, date: pd.Timestamp, max_obs: int) -> np.ndarray:
        p = int(permno)
        if p not in self.dates:
            return np.empty(0, dtype=float)
        d = np.datetime64(pd.Timestamp(date).to_datetime64())
        idx = int(np.searchsorted(self.dates[p], d, side="right"))
        lo = max(0, idx - int(max_obs))
        x = self.returns[p][lo:idx]
        return x[np.isfinite(x)]

    def prior_closes(self, permno: int, date: pd.Timestamp, count: int) -> np.ndarray:
        p = int(permno)
        if p not in self.dates:
            return np.empty(0, dtype=float)
        d = np.datetime64(pd.Timestamp(date).to_datetime64())
        idx = int(np.searchsorted(self.dates[p], d, side="right"))
        lo = max(0, idx - int(count))
        x = self.closes[p][lo:idx]
        return x[np.isfinite(x)]

    def forward_closes(self, permno: int, date: pd.Timestamp, count: int, *, include_signal_date: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return future security-session dates/closes after ``date``.

        Used only for post-run diagnostics. Trading decisions never call this
        method, which keeps the strategy lookahead-free.
        """
        p = int(permno)
        if p not in self.dates:
            return np.empty(0, dtype="datetime64[ns]"), np.empty(0, dtype=float)
        d = np.datetime64(pd.Timestamp(date).to_datetime64())
        side = "left" if include_signal_date else "right"
        idx = int(np.searchsorted(self.dates[p], d, side=side))
        hi = min(len(self.dates[p]), idx + int(count))
        ds = self.dates[p][idx:hi]
        px = self.closes[p][idx:hi]
        mask = np.isfinite(px)
        return ds[mask], px[mask]

    def realized_forward_path(self, permno: int, signal_date: pd.Timestamp, held_sessions: int) -> dict | None:
        """Diagnostic-only realized path matching next-close entry semantics.

        Entry is the first security-session close after ``signal_date``. The
        terminal close is the ``held_sessions``-th held close (entry day = 1).
        """
        ds, px = self.forward_closes(int(permno), signal_date, int(held_sessions))
        if len(px) < int(held_sessions) or len(px) == 0:
            return None
        entry = float(px[0])
        terminal = float(px[int(held_sessions) - 1])
        if not (math.isfinite(entry) and math.isfinite(terminal) and entry > 0):
            return None
        rel = px[: int(held_sessions)] / entry - 1.0
        return {
            "realized_entry_date": pd.Timestamp(ds[0]),
            "realized_terminal_date": pd.Timestamp(ds[int(held_sessions) - 1]),
            "realized_forward_return": float(terminal / entry - 1.0),
            "realized_mfe": float(np.nanmax(rel)),
            "realized_mae": float(np.nanmin(rel)),
        }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(root: Path | None = None) -> dict:
    root = root or repo_root()
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg["_root"] = str(root)
    return cfg


def verify_frozen_production_identity(root: Path | None = None) -> dict:
    root = root or repo_root()
    cfg_path = root / "config.yaml"
    engine_path = root / "src" / "backtest" / "engine.py"
    if not cfg_path.exists() or not engine_path.exists():
        raise FileNotFoundError("Frozen production config/engine not found; apply this patch inside the F2.2 repository.")
    cfg_hash = sha256_file(cfg_path)
    eng_hash = sha256_file(engine_path)
    if cfg_hash != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            f"Production config identity mismatch: {cfg_hash}. Expected frozen F2.2 {EXPECTED_CONFIG_SHA256}."
        )
    if eng_hash != EXPECTED_ENGINE_SHA256:
        raise RuntimeError(
            f"Production engine identity mismatch: {eng_hash}. Expected frozen F2.2 {EXPECTED_ENGINE_SHA256}."
        )
    return {"config_sha256": cfg_hash, "engine_sha256": eng_hash, "pass": True}


def feature_glob(cfg: dict) -> str:
    root = Path(cfg["_root"])
    feat = root / str(cfg["data"]["feature_dir"]) / "eligible"
    return str(feat).replace("\\", "/") + "/**/*.parquet"


def _q(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def load_feature_calendar(
    cfg: dict,
    history_start: str | pd.Timestamp,
    history_end: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    """Load only the global feature-store trading calendar."""
    import duckdb
    start = pd.Timestamp(history_start).strftime("%Y-%m-%d")
    end = pd.Timestamp(history_end).strftime("%Y-%m-%d")
    glob = feature_glob(cfg)
    con = duckdb.connect()
    try:
        x = con.execute(
            f"""SELECT DISTINCT date FROM read_parquet({_q(glob)}, hive_partitioning=true)
                WHERE date >= DATE {_q(start)} AND date <= DATE {_q(end)} ORDER BY date"""
        ).fetchdf()
    finally:
        con.close()
    if x.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(x["date"]))


def iter_feature_quote_history_chunks(
    cfg: dict,
    history_start: str | pd.Timestamp,
    history_end: str | pd.Timestamp,
):
    """Yield bounded quarterly quote/return chunks for HistoryStore construction.

    Only four columns are loaded. Quarterly bounds keep every DuckDB->Pandas
    conversion small while preserving all daily closes and canonical feature-store
    ``ret1`` observations needed by simulation.
    """
    import duckdb
    start = pd.Timestamp(history_start)
    end = pd.Timestamp(history_end)
    glob = feature_glob(cfg)
    cursor = pd.Timestamp(year=start.year, month=((start.month - 1) // 3) * 3 + 1, day=1)
    while cursor <= end:
        next_q = cursor + pd.offsets.QuarterBegin(startingMonth=1)
        q_end = next_q - pd.Timedelta(days=1)
        a = max(start, cursor)
        b = min(end, q_end)
        con = duckdb.connect()
        try:
            x = con.execute(
                f"""SELECT date, permno, close, ret1
                    FROM read_parquet({_q(glob)}, hive_partitioning=true)
                    WHERE date >= DATE {_q(a.strftime('%Y-%m-%d'))}
                      AND date <= DATE {_q(b.strftime('%Y-%m-%d'))}
                      AND permno IS NOT NULL
                    ORDER BY permno, date"""
            ).fetchdf()
        finally:
            con.close()
        if len(x):
            x["date"] = pd.to_datetime(x["date"])
            x["permno"] = pd.to_numeric(x["permno"], errors="coerce").astype("Int64")
            x = x.dropna(subset=["permno", "date"]).copy()
            x["permno"] = x["permno"].astype(int)
        yield x
        del x
        cursor = next_q


def load_feature_rows_for_dates(cfg: dict, dates) -> pd.DataFrame:
    """Load only signal-date feature rows required by Patch 2.9.3 alpha research.

    All columns are contemporaneously known at the signal close; entries remain
    next-close.  The optimizer still restricts tradable rows to ``eligible=TRUE``.
    """
    import duckdb
    dts = sorted({pd.Timestamp(d).normalize() for d in dates})
    cols = [
        "date", "permno", "ticker", "sector", "eligible", "close",
        "ret5", "rsi2", "distance_20d_high", "sma20", "sma50", "sma100", "sma200",
    ]
    if not dts:
        return pd.DataFrame(columns=cols)
    glob = feature_glob(cfg)
    literals = ",".join(f"DATE {_q(d.strftime('%Y-%m-%d'))}" for d in dts)
    con = duckdb.connect()
    try:
        x = con.execute(
            f"""SELECT date, permno, ticker, sector, eligible, close,
                       ret5, rsi2, distance_20d_high, sma20, sma50, sma100, sma200
                FROM read_parquet({_q(glob)}, hive_partitioning=true)
                WHERE date IN ({literals}) AND eligible = TRUE
                ORDER BY date, permno"""
        ).fetchdf()
    finally:
        con.close()
    if len(x):
        x["date"] = pd.to_datetime(x["date"])
        x["permno"] = pd.to_numeric(x["permno"], errors="coerce")
        x = x.dropna(subset=["date", "permno"]).copy()
        x["permno"] = x["permno"].astype(int)
    return x


def load_feature_market_breadth_for_dates(cfg: dict, dates) -> pd.DataFrame:
    """Causal one-row-per-date market-breadth diagnostics from the feature store.

    Breadth is calculated across all rows with the relevant feature available,
    not only the stochastic strategy's eligible subset.  This avoids loading the
    full universe into Python while providing a regime variable that is known at
    the signal close.  No forward-return or post-signal data enters the query.
    """
    import duckdb
    dts = sorted({pd.Timestamp(d).normalize() for d in dates})
    cols = ["date", "breadth_above_sma200", "breadth_above_sma100", "breadth_ret5_positive", "universe_rows"]
    if not dts:
        return pd.DataFrame(columns=cols)
    glob = feature_glob(cfg)
    literals = ",".join(f"DATE {_q(d.strftime('%Y-%m-%d'))}" for d in dts)
    con = duckdb.connect()
    try:
        x = con.execute(
            f"""SELECT date,
                       AVG(CASE WHEN close IS NULL OR sma200 IS NULL THEN NULL
                                WHEN close > sma200 THEN 1.0 ELSE 0.0 END) AS breadth_above_sma200,
                       AVG(CASE WHEN close IS NULL OR sma100 IS NULL THEN NULL
                                WHEN close > sma100 THEN 1.0 ELSE 0.0 END) AS breadth_above_sma100,
                       AVG(CASE WHEN ret5 IS NULL THEN NULL
                                WHEN ret5 > 0 THEN 1.0 ELSE 0.0 END) AS breadth_ret5_positive,
                       COUNT(*) AS universe_rows
                FROM read_parquet({_q(glob)}, hive_partitioning=true)
                WHERE date IN ({literals})
                GROUP BY date
                ORDER BY date"""
        ).fetchdf()
    finally:
        con.close()
    if len(x):
        x["date"] = pd.to_datetime(x["date"])
    return x

def load_feature_history(
    cfg: dict,
    history_start: str | pd.Timestamp | None = None,
    history_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load only columns needed by the stochastic research systems.

    Patch 2.6.1 adds an optional hard end bound so development research can
    physically avoid loading any 2025+ rows. This is stronger than merely
    filtering after load: post-development outcomes never enter process memory.
    """
    import duckdb

    if history_start is None:
        history_start = cfg["data"].get("research_start_date", "2015-01-01")
    history_start = pd.Timestamp(history_start).strftime("%Y-%m-%d")
    end_clause = ""
    if history_end is not None:
        history_end = pd.Timestamp(history_end).strftime("%Y-%m-%d")
        end_clause = f" AND date <= DATE {_q(history_end)}"
    glob = feature_glob(cfg)
    con = duckdb.connect()
    sql = f"""
        SELECT date, permno, ticker, sector, eligible, close, ret1, ret5, rsi2,
               distance_20d_high, sma20, sma50, sma100, sma200
        FROM read_parquet({_q(glob)}, hive_partitioning=true)
        WHERE date >= DATE {_q(history_start)}{end_clause}
        ORDER BY date, permno
    """
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    if history_end is not None and len(df):
        hard_end = pd.Timestamp(history_end)
        if pd.Timestamp(df["date"].max()) > hard_end:
            raise RuntimeError(
                f"Feature loader violated hard history_end: max={df['date'].max()} end={hard_end}"
            )
    return df




def _resolve_data_path(cfg: dict, value: str) -> Path:
    """Resolve a config data path exactly like the production data pipeline."""
    p = Path(str(value))
    if p.is_absolute():
        return p
    return Path(cfg["_root"]) / p


def normalized_glob(cfg: dict) -> str:
    """Return the normalized CRSP parquet glob used for calibration-only backfill."""
    value = cfg.get("data", {}).get("normalized_dir")
    if not value:
        raise RuntimeError("config.yaml is missing data.normalized_dir; cannot backfill pre-feature calibration history.")
    norm = _resolve_data_path(cfg, value)
    if not norm.exists():
        raise FileNotFoundError(
            f"Normalized CRSP store not found: {norm}. The feature store begins too late for full 2015 calibration, "
            "so Patch 2.6.1 requires the existing normalized CRSP parquet store."
        )
    return str(norm).replace("\\", "/") + "/**/*.parquet"


def load_normalized_calibration_backfill(
    cfg: dict,
    *,
    before_date: str | pd.Timestamp,
    global_sessions_needed: int,
) -> pd.DataFrame:
    """Load only the missing pre-feature CRSP history needed for calibration.

    This is calibration-only data. It never contains feature columns and is never
    passed to the signal/portfolio engine. The query takes the most recent N global
    CRSP sessions strictly before ``before_date`` so the Heston-Merton history store
    can reach its configured maximum lookback at the start of the scored window.
    """
    needed = int(global_sessions_needed)
    if needed <= 0:
        return pd.DataFrame(columns=["date", "permno", "ret1", "close"])

    import duckdb

    cutoff = pd.Timestamp(before_date).strftime("%Y-%m-%d")
    glob = normalized_glob(cfg)
    con = duckdb.connect()
    try:
        dates = con.execute(
            f"""
            SELECT date
            FROM (
                SELECT DISTINCT date
                FROM read_parquet({_q(glob)}, hive_partitioning=true, union_by_name=true)
                WHERE date < DATE {_q(cutoff)}
            ) d
            ORDER BY date DESC
            LIMIT {needed}
            """
        ).fetchdf()
        if len(dates) < needed:
            raise RuntimeError(
                f"Normalized CRSP store has only {len(dates)} global sessions before {cutoff}; "
                f"need {needed} additional sessions for full calibration warm-up."
            )
        start = pd.Timestamp(dates["date"].min()).strftime("%Y-%m-%d")
        df = con.execute(
            f"""
            SELECT date, permno, close, ret AS ret1
            FROM read_parquet({_q(glob)}, hive_partitioning=true, union_by_name=true)
            WHERE date >= DATE {_q(start)} AND date < DATE {_q(cutoff)}
              AND permno IS NOT NULL AND close IS NOT NULL AND close > 0
            ORDER BY date, permno
            """
        ).fetchdf()
    finally:
        con.close()

    if df.empty:
        raise RuntimeError("Normalized CRSP calibration backfill query returned no rows.")
    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ret1"] = pd.to_numeric(df["ret1"], errors="coerce")
    df = df.dropna(subset=["permno", "date", "close"]).copy()
    df["permno"] = df["permno"].astype(int)
    return df[["date", "permno", "ret1", "close"]]


def merge_calibration_history(feature_history: pd.DataFrame, normalized_backfill: pd.DataFrame) -> pd.DataFrame:
    """Combine calibration-only normalized rows with feature-store returns.

    Feature-store rows win on duplicate PERMNO/date keys because their ``ret1``
    definition is the canonical strategy feature.
    """
    cols = ["date", "permno", "ret1", "close"]
    f = feature_history[cols].copy()
    b = normalized_backfill[cols].copy() if len(normalized_backfill) else pd.DataFrame(columns=cols)
    f["_source_priority"] = 1
    b["_source_priority"] = 0
    out = pd.concat([b, f], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = (
        out.sort_values(["permno", "date", "_source_priority"])
        .drop_duplicates(["permno", "date"], keep="last")
        .drop(columns=["_source_priority"])
        .sort_values(["date", "permno"])
        .reset_index(drop=True)
    )
    return out


def load_f22_rankings(root: Path | None = None) -> pd.DataFrame:
    root = root or repo_root()
    candidates = [
        root / "outputs" / "screener" / "monthly_rankings.csv",
        root / "outputs" / "screener" / "monthly_top25.csv",
        root / "outputs" / "screener" / "monthly_top15.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "F2.2 monthly rankings not found. Run the production screener before stochastic shadow research."
        )
    r = pd.read_csv(path)
    if "effective_month" not in r.columns:
        if "ranking_date" not in r.columns:
            raise RuntimeError(f"Unsupported rankings schema in {path}")
        r["ranking_date"] = pd.to_datetime(r["ranking_date"])
        r["effective_month"] = r["ranking_date"].dt.to_period("M").astype(str)
    if "leadership_score" not in r.columns:
        # Current production rankings normally include this; preserve deterministic
        # order if a compact artifact omits it.
        if "rank" in r.columns:
            r["leadership_score"] = -pd.to_numeric(r["rank"], errors="coerce").fillna(999999.0)
        else:
            r["leadership_score"] = 0.0
    return r


def load_spec(root: Path | None = None) -> dict:
    root = root or repo_root()
    path = root / "research_stochastic" / "STOCHASTIC_RESEARCH_SPEC.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_production_summary(root: Path | None = None) -> dict | None:
    root = root or repo_root()
    candidates = [
        root / "outputs" / "baselines" / "foundation2_2_production_v0_2_2" / "summary.json",
        root / "outputs" / "backtest" / "summary.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def load_benchmark(cfg: dict, symbol: str = "SPY", start: str | pd.Timestamp | None = None) -> pd.DataFrame | None:
    root = Path(cfg["_root"])
    p = root / str(cfg["data"]["benchmark_dir"]) / f"{symbol.lower()}_daily.csv"
    if not p.exists():
        return None
    b = pd.read_csv(p)
    if "date" not in b.columns or "return" not in b.columns:
        return None
    b["date"] = pd.to_datetime(b["date"])
    if start is not None:
        b = b[b["date"] >= pd.Timestamp(start)].copy()
    return b.sort_values("date")
