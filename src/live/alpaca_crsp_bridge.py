from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .download_alpaca_2026_ytd import (
    StdlibSession,
    _headers,
    chunks,
    download_batch_with_quarantine,
    load_credentials,
    write_df_parquet,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "alpaca" / "2026" / "alpaca_us_equities_2026_ytd.parquet"
RAW_MANIFEST = ROOT / "data" / "manifests" / "alpaca_2026_ytd.json"
SYMBOLS = ROOT / "data" / "alpaca" / "2026" / "alpaca_symbols_2026.csv"
OUT_DIR = ROOT / "data" / "alpaca" / "2026" / "bridge"
BRIDGE_CSV = OUT_DIR / "alpaca_crsp_bridge_2026.csv"
UNMATCHED_CSV = OUT_DIR / "alpaca_crsp_unmatched_2026.csv"
AMBIG_CSV = OUT_DIR / "alpaca_crsp_ambiguous_2026.csv"
ANCHOR_PARQUET = OUT_DIR / "alpaca_anchor_dec2025.parquet"
ANCHOR_MATCH_CSV = OUT_DIR / "alpaca_crsp_anchor_matches.csv"
MANUAL = ROOT / "data" / "alpaca" / "2026" / "alpaca_crsp_manual_overrides.csv"
MANIFEST = ROOT / "data" / "manifests" / "alpaca_crsp_bridge_2026.json"
PERMNO_COLLISION_CSV = OUT_DIR / "alpaca_crsp_permno_collision_resolution_2026.csv"
LOG = ROOT / "logs" / "alpaca_crsp_bridge.log"
DATE_CANDIDATES = ("date", "dlycaldt", "caldt", "trade_date", "trading_date")
PERMNO_CANDIDATES = ("permno", "kypermno")
TICKER_CANDIDATES = ("ticker", "symbol", "tic")
PRICE_CANDIDATES = ("prc", "dlyprc", "close", "price")


class BridgeError(RuntimeError):
    pass


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text: str) -> int:
        for s in self.streams:
            s.write(text); s.flush()
        return len(text)
    def flush(self) -> None:
        for s in self.streams: s.flush()


def _duckdb():
    try:
        import duckdb
        return duckdb
    except ImportError as exc:
        raise BridgeError("DuckDB is required. Run 00_setup_env.cmd first.") from exc


def _q(x: str) -> str:
    return '"' + x.replace('"', '""') + '"'


def _s(x: str | Path) -> str:
    return "'" + str(x).replace("'", "''").replace("\\", "/") + "'"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _find_col(cols: list[str], candidates: tuple[str, ...], label: str) -> str:
    lookup = {c.lower(): c for c in cols}
    for x in candidates:
        if x in lookup:
            return lookup[x]
    raise BridgeError(f"Could not resolve {label}; expected one of {candidates}, found columns={cols}")


def _normalized_files() -> list[Path]:
    files = sorted((ROOT / "data" / "normalized").glob("year=2025/*.parquet"))
    if not files:
        raise BridgeError("No 2025 normalized Parquet found under data/normalized/year=2025")
    return files


def _schema(files: list[Path]) -> tuple[list[str], dict[str, str]]:
    con = _duckdb().connect()
    try:
        rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet({_s(files[0])}, hive_partitioning=false)").fetchall()
        return [str(r[0]) for r in rows], {str(r[0]): str(r[1]) for r in rows}
    finally:
        con.close()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def _ensure_override_template() -> None:
    if MANUAL.exists(): return
    _write_csv(MANUAL, [], ["symbol", "permno", "approved", "reason"])


def _load_overrides(valid_permnos: set[int]) -> dict[str, int]:
    _ensure_override_template()
    df = pd.read_csv(MANUAL, dtype=str).fillna("")
    out: dict[str, int] = {}
    for _, r in df.iterrows():
        if str(r.get("approved", "")).strip().upper() not in {"Y", "YES", "TRUE", "1"}:
            continue
        symbol = str(r.get("symbol", "")).strip().upper()
        try: permno = int(float(str(r.get("permno", "")).strip()))
        except Exception as exc: raise BridgeError(f"Invalid manual override PERMNO for {symbol}") from exc
        if not symbol or permno not in valid_permnos:
            raise BridgeError(f"Manual override {symbol}->{permno} references an unknown historical PERMNO")
        if symbol in out and out[symbol] != permno:
            raise BridgeError(f"Conflicting manual overrides for {symbol}")
        out[symbol] = permno
    return out




def load_2025_permno_boundary() -> pd.DataFrame:
    """Return the latest observed 2025 CRSP ticker for each PERMNO.

    Normalized date/PERMNO uniqueness should imply one ticker on the boundary date.
    We still carry a ticker-count guard so an anomalous history fails closed.
    """
    files = _normalized_files()
    cols, _ = _schema(files)
    date_col = _find_col(cols, DATE_CANDIDATES, "date column")
    permno_col = _find_col(cols, PERMNO_CANDIDATES, "PERMNO column")
    ticker_col = _find_col(cols, TICKER_CANDIDATES, "ticker column")
    glob = ROOT / "data" / "normalized" / "year=2025" / "*.parquet"
    con = _duckdb().connect()
    try:
        return con.execute(f"""
            WITH h AS (
              SELECT
                TRY_CAST({_q(permno_col)} AS BIGINT) AS permno,
                UPPER(TRIM(CAST({_q(ticker_col)} AS VARCHAR))) AS ticker,
                TRY_CAST({_q(date_col)} AS DATE) AS d
              FROM read_parquet({_s(glob)}, union_by_name=true, hive_partitioning=false)
              WHERE TRY_CAST({_q(permno_col)} AS BIGINT) IS NOT NULL
                AND {_q(ticker_col)} IS NOT NULL
                AND TRIM(CAST({_q(ticker_col)} AS VARCHAR)) <> ''
            ), mx AS (
              SELECT permno, MAX(d) AS boundary_date FROM h GROUP BY 1
            )
            SELECT
              h.permno,
              MIN(h.ticker) AS boundary_ticker,
              mx.boundary_date,
              COUNT(DISTINCT h.ticker) AS boundary_ticker_count
            FROM h
            JOIN mx ON mx.permno=h.permno AND mx.boundary_date=h.d
            GROUP BY h.permno,mx.boundary_date
            ORDER BY h.permno
        """).df()
    finally:
        con.close()


def resolve_inverse_permno_collisions(bridge: pd.DataFrame, boundary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Enforce one accepted forward ticker per historical PERMNO.

    Patch 1.20 mapped tickers independently.  If a PERMNO changed ticker during
    2025, both the historical alias and the year-end ticker could therefore map
    to the same PERMNO.  In 2026 those symbols can be different Alpaca assets.

    Resolution is identity based, never price based:
      * a single explicit MANUAL_APPROVED row wins a collision;
      * otherwise the unique latest 2025 CRSP ticker wins;
      * if neither rule yields one winner, every row in that collision is
        demoted and the caller must fail closed / request manual review.
    """
    out = bridge.copy()
    if out.empty:
        return out, pd.DataFrame(), {"permno_collision_groups":0,"permno_collision_rows":0,"permno_collision_rows_demoted":0,"permno_collision_groups_unresolved":0}
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["permno"] = pd.to_numeric(out.get("permno"), errors="coerce").astype("Int64")
    out["pre_resolution_permno"] = out["permno"]
    b = boundary.copy()
    if not b.empty:
        b["permno"] = pd.to_numeric(b["permno"], errors="coerce").astype("Int64")
        b["boundary_ticker"] = b["boundary_ticker"].astype(str).str.strip().str.upper()
    bmap = {int(r.permno): r for r in b.itertuples(index=False) if pd.notna(r.permno)}
    out["boundary_ticker"] = ""
    out["boundary_date"] = ""
    out["boundary_ticker_count"] = 0
    out["inverse_resolution"] = "NOT_MAPPED"
    out.loc[out["permno"].notna(), "inverse_resolution"] = "KEEP_UNIQUE"

    mapped = out[out["permno"].notna()]
    counts = mapped.groupby("permno", dropna=True).size()
    collision_permnos = [int(x) for x,n in counts.items() if int(n)>1]
    report_rows=[]
    unresolved_groups=0
    demoted=0
    for permno in collision_permnos:
        idx = out.index[out["permno"] == permno].tolist()
        rec = bmap.get(permno)
        bt = str(getattr(rec, "boundary_ticker", "") or "").upper()
        bd = str(getattr(rec, "boundary_date", "") or "")
        btc = int(getattr(rec, "boundary_ticker_count", 0) or 0)
        out.loc[idx, "boundary_ticker"] = bt
        out.loc[idx, "boundary_date"] = bd
        out.loc[idx, "boundary_ticker_count"] = btc
        manual = [i for i in idx if str(out.at[i,"status"]).upper()=="MANUAL_APPROVED"]
        winner = None
        reason = ""
        if len(manual)==1:
            winner=manual[0]; reason="MANUAL_OVERRIDE_WINS"
        elif len(manual)>1:
            reason="UNRESOLVED_MULTIPLE_MANUAL_OVERRIDES"
        elif btc==1 and bt:
            matches=[i for i in idx if out.at[i,"symbol"]==bt]
            if len(matches)==1:
                winner=matches[0]; reason="LATEST_2025_CRSP_TICKER_WINS"
            else:
                reason="UNRESOLVED_BOUNDARY_TICKER_NOT_PRESENT"
        else:
            reason="UNRESOLVED_AMBIGUOUS_2025_BOUNDARY"

        if winner is None:
            unresolved_groups += 1
            for i in idx:
                original = int(out.at[i,"permno"])
                out.at[i,"permno"] = pd.NA
                out.at[i,"status"] = "AMBIGUOUS_PERMNO_COLLISION"
                out.at[i,"inverse_resolution"] = reason
                demoted += 1
                report_rows.append({"permno":original,"symbol":out.at[i,"symbol"],"accepted":False,"resolution":reason,"boundary_ticker":bt,"boundary_date":bd,"boundary_ticker_count":btc,"prior_status":bridge.at[i,"status"]})
        else:
            for i in idx:
                accepted = i==winner
                original = int(out.at[i,"permno"])
                if accepted:
                    out.at[i,"inverse_resolution"] = reason
                else:
                    out.at[i,"permno"] = pd.NA
                    out.at[i,"status"] = "SUPERSEDED_2025_TICKER_ALIAS"
                    out.at[i,"inverse_resolution"] = reason
                    demoted += 1
                report_rows.append({"permno":original,"symbol":out.at[i,"symbol"],"accepted":accepted,"resolution":reason,"boundary_ticker":bt,"boundary_date":bd,"boundary_ticker_count":btc,"prior_status":bridge.at[i,"status"]})

    report = pd.DataFrame(report_rows, columns=["permno","symbol","accepted","resolution","boundary_ticker","boundary_date","boundary_ticker_count","prior_status"])
    stats={
        "permno_collision_groups":len(collision_permnos),
        "permno_collision_rows":int(sum(int(counts.loc[p]) for p in collision_permnos)) if collision_permnos else 0,
        "permno_collision_rows_demoted":int(demoted),
        "permno_collision_groups_unresolved":int(unresolved_groups),
    }
    return out, report, stats


def build_symbol_bridge(recency_start: str = "2025-07-01") -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if not RAW.exists(): raise BridgeError(f"Missing Alpaca raw Parquet: {RAW}")
    if not RAW_MANIFEST.exists(): raise BridgeError(f"Missing Alpaca manifest: {RAW_MANIFEST}")
    files = _normalized_files()
    cols, types = _schema(files)
    date_col = _find_col(cols, DATE_CANDIDATES, "date column")
    permno_col = _find_col(cols, PERMNO_CANDIDATES, "PERMNO column")
    ticker_col = _find_col(cols, TICKER_CANDIDATES, "ticker column")
    price_col = _find_col(cols, PRICE_CANDIDATES, "price column")

    glob = ROOT / "data" / "normalized" / "year=2025" / "*.parquet"
    con = _duckdb().connect()
    try:
        hist = con.execute(f"""
            WITH h AS (
              SELECT
                TRY_CAST({_q(permno_col)} AS BIGINT) AS permno,
                UPPER(TRIM(CAST({_q(ticker_col)} AS VARCHAR))) AS ticker,
                TRY_CAST({_q(date_col)} AS DATE) AS d,
                ABS(TRY_CAST({_q(price_col)} AS DOUBLE)) AS price
              FROM read_parquet({_s(glob)}, union_by_name=true, hive_partitioning=false)
              WHERE TRY_CAST({_q(date_col)} AS DATE) >= DATE { _s(recency_start) }
                AND {_q(ticker_col)} IS NOT NULL
                AND TRY_CAST({_q(permno_col)} AS BIGINT) IS NOT NULL
            ), g AS (
              SELECT ticker, permno, MAX(d) AS last_seen, COUNT(*) AS observations
              FROM h WHERE ticker <> '' GROUP BY 1,2
            )
            SELECT * FROM g ORDER BY ticker, last_seen DESC, permno
        """).df()
        raw_symbols = con.execute(f"SELECT DISTINCT UPPER(TRIM(symbol)) AS symbol FROM read_parquet({_s(RAW)}) ORDER BY 1").df()
        valid_permno_df = con.execute(f"SELECT DISTINCT TRY_CAST({_q(permno_col)} AS BIGINT) AS permno FROM read_parquet({_s(glob)}, union_by_name=true, hive_partitioning=false) WHERE TRY_CAST({_q(permno_col)} AS BIGINT) IS NOT NULL").df()
    finally:
        con.close()
    if hist.empty: raise BridgeError("No recent ticker/PERMNO observations found in 2025 normalized data")
    valid_permnos = set(int(x) for x in valid_permno_df["permno"].dropna().astype(int).tolist())
    overrides = _load_overrides(valid_permnos)

    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in hist.to_dict("records"):
        by_ticker.setdefault(str(row["ticker"]).upper(), []).append(row)

    rows: list[dict[str, Any]] = []
    for symbol in raw_symbols["symbol"].astype(str):
        candidates = by_ticker.get(symbol, [])
        if symbol in overrides:
            rows.append({"symbol": symbol, "permno": overrides[symbol], "status": "MANUAL_APPROVED", "candidate_permnos": str(overrides[symbol]), "last_seen": None, "candidate_count": 1})
        elif len(candidates) == 1:
            c = candidates[0]
            rows.append({"symbol": symbol, "permno": int(c["permno"]), "status": "AUTO_EXACT_RECENT", "candidate_permnos": str(int(c["permno"])), "last_seen": str(c["last_seen"]), "candidate_count": 1})
        elif len(candidates) > 1:
            rows.append({"symbol": symbol, "permno": None, "status": "AMBIGUOUS_RECENT_TICKER", "candidate_permnos": ";".join(str(int(c["permno"])) for c in candidates), "last_seen": str(max(c["last_seen"] for c in candidates)), "candidate_count": len(candidates)})
        else:
            rows.append({"symbol": symbol, "permno": None, "status": "UNMATCHED", "candidate_permnos": "", "last_seen": None, "candidate_count": 0})
    bridge = pd.DataFrame(rows)
    mapped_before = int(bridge.permno.notna().sum())
    boundary = load_2025_permno_boundary()
    bridge, collision_report, collision_stats = resolve_inverse_permno_collisions(bridge, boundary)
    if not collision_report.empty:
        _write_csv(PERMNO_COLLISION_CSV, collision_report.to_dict("records"), list(collision_report.columns))
    else:
        PERMNO_COLLISION_CSV.unlink(missing_ok=True)
    if collision_stats["permno_collision_groups_unresolved"]:
        raise BridgeError(
            f"{collision_stats['permno_collision_groups_unresolved']} inverse PERMNO collision group(s) cannot be resolved by a single manual override or the unique latest 2025 CRSP ticker. Review {PERMNO_COLLISION_CSV}."
        )
    stats = {
        "raw_symbols_with_bars": int(len(bridge)),
        "auto_exact": int((bridge.status == "AUTO_EXACT_RECENT").sum()),
        "manual_approved": int((bridge.status == "MANUAL_APPROVED").sum()),
        "ambiguous": int((bridge.status == "AMBIGUOUS_RECENT_TICKER").sum()),
        "unmatched": int((bridge.status == "UNMATCHED").sum()),
        "mapped_symbols_pre_inverse_resolution": mapped_before,
        "mapped_symbols_pre_anchor": int(bridge.permno.notna().sum()),
        "recency_start": recency_start,
        **collision_stats,
    }
    return bridge, stats, {"resolved": {"date": date_col, "permno": permno_col, "ticker": ticker_col, "price": price_col}, "types": types, "columns": cols}


def _load_asset_map() -> dict[str, dict[str, Any]]:
    if not SYMBOLS.exists(): raise BridgeError(f"Missing Alpaca symbol snapshot: {SYMBOLS}")
    df = pd.read_csv(SYMBOLS, dtype=str).fillna("")
    return {str(r["symbol"]).upper(): dict(r) for _, r in df.iterrows() if str(r.get("symbol", "")).strip()}


def download_anchor(mapped_symbols: list[str], feed: str, adjustment: str, credentials: str | None, batch_size: int = 50) -> tuple[Path, list[dict[str, Any]]]:
    if not mapped_symbols: raise BridgeError("No mapped symbols available for December 2025 anchor download")
    creds = load_credentials(ROOT, credentials)
    session = StdlibSession(); headers = _headers(creds); asset_map = _load_asset_map()
    stage = OUT_DIR / ".anchor_staging" / uuid.uuid4().hex
    parts = stage / "parts"; parts.mkdir(parents=True, exist_ok=True)
    rejected: list[dict[str, Any]] = []
    paths: list[Path] = []
    try:
        batches = list(chunks(mapped_symbols, batch_size))
        for i, batch in enumerate(batches, 1):
            frame, rej = download_batch_with_quarantine(session, headers, batch, asset_map, "2025-12-01", "2025-12-31", feed, adjustment, 10000)
            rejected.extend(rej)
            if not frame.empty:
                p = parts / f"part_{i:05d}.parquet"; write_df_parquet(frame, p); paths.append(p)
            if i == 1 or i % 20 == 0 or i == len(batches):
                print(f"Anchor batch {i}/{len(batches)}: rows={len(frame):,}, rejected={len(rejected):,}")
        if not paths: raise BridgeError("Anchor download returned no December 2025 bars")
        con = _duckdb().connect()
        tmp = ANCHOR_PARQUET.with_name(ANCHOR_PARQUET.name + ".tmp")
        try:
            ANCHOR_PARQUET.parent.mkdir(parents=True, exist_ok=True)
            con.execute(f"COPY (SELECT * FROM read_parquet({_s(parts / '*.parquet')}, union_by_name=true) ORDER BY date,symbol) TO {_s(tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)")
        finally: con.close()
        os.replace(tmp, ANCHOR_PARQUET)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return ANCHOR_PARQUET, rejected


def match_anchors(bridge: pd.DataFrame, schema_map: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = _normalized_files(); glob = ROOT / "data" / "normalized" / "year=2025" / "*.parquet"
    mapped = bridge[bridge.permno.notna()].copy()
    if mapped.empty: raise BridgeError("Bridge has no mapped symbols")
    temp_bridge = OUT_DIR / ".mapped_bridge.parquet"; write_df_parquet(mapped[["symbol", "permno", "status"]], temp_bridge)
    resolved = schema_map["resolved"]
    dcol, pcol, prcol = resolved["date"], resolved["permno"], resolved["price"]
    con = _duckdb().connect()
    try:
        df = con.execute(f"""
          WITH b AS (SELECT symbol, TRY_CAST(permno AS BIGINT) permno, status FROM read_parquet({_s(temp_bridge)})),
          a AS (SELECT UPPER(symbol) symbol, TRY_CAST(date AS DATE) d, TRY_CAST(close AS DOUBLE) alpaca_close FROM read_parquet({_s(ANCHOR_PARQUET)})),
          h AS (SELECT TRY_CAST({_q(pcol)} AS BIGINT) permno, TRY_CAST({_q(dcol)} AS DATE) d, ABS(TRY_CAST({_q(prcol)} AS DOUBLE)) crsp_price FROM read_parquet({_s(glob)}, union_by_name=true, hive_partitioning=false)),
          x AS (
            SELECT b.symbol,b.permno,b.status,a.d,a.alpaca_close,h.crsp_price,
                   ROW_NUMBER() OVER(PARTITION BY b.symbol ORDER BY a.d DESC) rn
            FROM b JOIN a USING(symbol) JOIN h ON h.permno=b.permno AND h.d=a.d
            WHERE a.alpaca_close>0 AND h.crsp_price>0
          )
          SELECT symbol,permno,status,d AS anchor_date,alpaca_close,crsp_price,crsp_price/alpaca_close AS price_scale
          FROM x WHERE rn=1 ORDER BY symbol
        """).df()
    finally:
        con.close(); temp_bridge.unlink(missing_ok=True)
    anchored = set(df.symbol.astype(str))
    bridge2 = bridge.copy(); bridge2["anchor_status"] = bridge2["symbol"].map(lambda s: "PASS" if s in anchored else ("NOT_APPLICABLE" if pd.isna(bridge2.loc[bridge2.symbol==s,"permno"]).all() else "MISSING"))
    # simpler vectorized update
    bridge2["anchor_status"] = "NOT_APPLICABLE"
    bridge2.loc[bridge2.permno.notna(), "anchor_status"] = bridge2.loc[bridge2.permno.notna(), "symbol"].isin(anchored).map({True:"PASS",False:"MISSING"})
    bridge2.loc[(bridge2.permno.notna()) & (bridge2.anchor_status=="MISSING"), "status"] = "ANCHOR_MISSING"
    stats = {"anchor_matches": int(len(df)), "mapped_without_anchor": int(((bridge2.permno.notna()) & (bridge2.anchor_status=="MISSING")).sum())}
    return df, {**stats, "bridge": bridge2}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build audited Alpaca-2026 to recent CRSP PERMNO bridge and same-date December-2025 calibration anchors.")
    ap.add_argument("--recency-start", default="2025-07-01")
    ap.add_argument("--credentials", default=None)
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args(argv)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8", buffering=1) as log:
        oldout, olderr = sys.stdout, sys.stderr; sys.stdout=Tee(oldout,log); sys.stderr=Tee(olderr,log)
        try:
            print("="*78); print("ALPACA 2026 -> CRSP/PERMNO IDENTITY BRIDGE"); print("="*78)
            raw_manifest = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
            if str(raw_manifest.get("feed_used","")).lower() != "sip": raise BridgeError("Raw Alpaca file must use consolidated SIP")
            if "asof=-" not in str(raw_manifest.get("historical_symbol_mapping","")): raise BridgeError("Raw Alpaca manifest does not prove symbol remapping was disabled with asof=-")
            bridge, stats, schema_map = build_symbol_bridge(args.recency_start)
            mapped = bridge[bridge.permno.notna()].copy()
            print(f"Raw symbols with bars: {stats['raw_symbols_with_bars']:,}")
            print(f"Exact recent mappings: {stats['auto_exact']:,}; manual: {stats['manual_approved']:,}; ambiguous: {stats['ambiguous']:,}; unmatched: {stats['unmatched']:,}")
            print(
                f"Inverse PERMNO collisions: groups={stats['permno_collision_groups']:,}; "
                f"rows={stats['permno_collision_rows']:,}; demoted={stats['permno_collision_rows_demoted']:,}; "
                f"unresolved={stats['permno_collision_groups_unresolved']:,}"
            )
            print("Downloading December 2025 Alpaca anchors for mapped incumbents...")
            _, rejected = download_anchor(mapped.symbol.astype(str).tolist(), str(raw_manifest.get("feed_used","sip")), str(raw_manifest.get("adjustment","all")), args.credentials, args.batch_size)
            anchor_df, anchor_info = match_anchors(bridge, schema_map)
            bridge2 = anchor_info.pop("bridge")
            fields = ["symbol","permno","status","candidate_permnos","last_seen","candidate_count","anchor_status","pre_resolution_permno","boundary_ticker","boundary_date","boundary_ticker_count","inverse_resolution"]
            _write_csv(BRIDGE_CSV, bridge2.to_dict("records"), fields)
            _write_csv(UNMATCHED_CSV, bridge2[bridge2.status=="UNMATCHED"].to_dict("records"), fields)
            _write_csv(AMBIG_CSV, bridge2[bridge2.status=="AMBIGUOUS_RECENT_TICKER"].to_dict("records"), fields)

            # Reconcile bridge and anchor counts on the exact accepted symbol/PERMNO keys.
            accepted_keys = bridge2[bridge2.permno.notna()][["symbol", "permno"]].copy()
            accepted_keys["symbol"] = accepted_keys["symbol"].astype(str).str.strip().str.upper()
            accepted_keys["permno"] = pd.to_numeric(accepted_keys["permno"], errors="raise").astype(int)
            anchor_df2 = anchor_df.copy()
            anchor_df2["symbol"] = anchor_df2["symbol"].astype(str).str.strip().str.upper()
            anchor_df2["permno"] = pd.to_numeric(anchor_df2["permno"], errors="raise").astype(int)
            accepted_anchor_df = anchor_df2.merge(accepted_keys.drop_duplicates(), on=["symbol", "permno"], how="inner", validate="one_to_one")
            _write_csv(ANCHOR_MATCH_CSV, accepted_anchor_df.to_dict("records"), ["symbol","permno","status","anchor_date","alpaca_close","crsp_price","price_scale"])
            accepted_mappings = int(len(accepted_keys))
            mapped_anchor = int(len(accepted_anchor_df))
            anchor_rows_excluded_not_accepted = int(len(anchor_df2) - len(accepted_anchor_df))
            if mapped_anchor < 500:
                raise BridgeError(f"Only {mapped_anchor} symbols have a valid CRSP/Alpaca same-date anchor; refusing to build a forward normalized universe")
            payload = {
                "schema_version": 2,
                "status": "PASS",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "raw_alpaca": str(RAW), "raw_alpaca_sha256": sha256_file(RAW),
                "raw_manifest_sha256": sha256_file(RAW_MANIFEST),
                "normalized_2025_files": len(_normalized_files()),
                "historical_schema_columns": schema_map["columns"],
                "resolved_columns": schema_map["resolved"],
                **stats, **anchor_info,
                "anchor_rejected_symbols": len(rejected),
                "accepted_mappings_after_inverse_resolution": accepted_mappings,
                "mapped_symbols_with_anchor": mapped_anchor,
                "anchor_rows_excluded_not_accepted": anchor_rows_excluded_not_accepted,
                "bridge_csv": str(BRIDGE_CSV), "anchor_parquet": str(ANCHOR_PARQUET), "anchor_match_csv": str(ANCHOR_MATCH_CSV),
                "manual_override_file": str(MANUAL), "permno_collision_resolution_csv": str(PERMNO_COLLISION_CSV) if PERMNO_COLLISION_CSV.exists() else None,
                "mapping_policy": "auto-map exact ticker only when it has one recent PERMNO, then enforce one accepted forward ticker per PERMNO; a single manual approval wins, otherwise the unique latest 2025 CRSP ticker wins; unresolved inverse collisions fail closed",
                "new_ipo_policy": "unmatched 2026 symbols are not assigned synthetic PERMNOs; this is conservative and consistent with the strategy's 252-day minimum history requirement",
            }
            MANIFEST.parent.mkdir(parents=True, exist_ok=True); tmp=MANIFEST.with_name(MANIFEST.name+".tmp"); tmp.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8"); os.replace(tmp,MANIFEST)
            print("="*78); print("BRIDGE PASS")
            print(f"Accepted mappings after inverse resolution: {accepted_mappings:,}")
            print(f"Accepted mappings with valid anchor:        {mapped_anchor:,}")
            print(f"Anchor rows excluded as non-accepted keys:  {anchor_rows_excluded_not_accepted:,}")
            print(f"Bridge:   {BRIDGE_CSV}"); print(f"Manifest: {MANIFEST}"); print("="*78)
            return 0
        finally:
            sys.stdout=oldout; sys.stderr=olderr

if __name__ == "__main__": raise SystemExit(main())
