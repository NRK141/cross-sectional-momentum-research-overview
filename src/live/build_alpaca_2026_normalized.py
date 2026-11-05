from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .alpaca_crsp_bridge import (
    ROOT, RAW, RAW_MANIFEST, BRIDGE_CSV, ANCHOR_MATCH_CSV, MANIFEST as BRIDGE_MANIFEST,
    BridgeError, Tee, _duckdb, _q, _s, _schema, _normalized_files, sha256_file,
    load_2025_permno_boundary, resolve_inverse_permno_collisions,
)

OUT = ROOT / "data" / "alpaca" / "2026" / "bridge" / "normalized_year2026_staging.parquet"
MANIFEST = ROOT / "data" / "manifests" / "alpaca_normalized_2026.json"
LOG = ROOT / "logs" / "alpaca_normalized_2026.log"
DUP_REPORT = ROOT / "data" / "alpaca" / "2026" / "bridge" / "alpaca_duplicate_permno_dates_2026.csv"
ALIAS_REPORT = ROOT / "data" / "alpaca" / "2026" / "bridge" / "alpaca_alias_resolution_2026.csv"
CONFLICT_REPORT = ROOT / "data" / "alpaca" / "2026" / "bridge" / "alpaca_conflicting_permno_dates_2026.csv"
PERMNO_RESOLUTION_REPORT = ROOT / "data" / "alpaca" / "2026" / "bridge" / "alpaca_permno_collision_resolution_2026.csv"
UNANCHORED_REPORT = ROOT / "data" / "alpaca" / "2026" / "bridge" / "alpaca_unanchored_mappings_2026.csv"
MAX_UNANCHORED_MAPPINGS = 100
MAX_UNANCHORED_FRACTION = 0.01

DATE_CANDS=("date","dlycaldt","caldt","trade_date","trading_date")
PERMNO_CANDS=("permno","kypermno")
TICKER_CANDS=("ticker","symbol","tic")
PRICE_CANDS=("prc","dlyprc","close","price")
RET_CANDS=("ret","dlyret","return")
VOL_CANDS=("vol","dlyvol","volume")
OPEN_CANDS=("open","openprc","dlyopen")
HIGH_CANDS=("high","askhi","dlyhigh")
LOW_CANDS=("low","bidlo","dlylow")
CLOSE_CANDS=("close",)

PROVENANCE_FORWARD_SPECIALS = {
    "open_is_trade": "TRUE", "high_is_trade": "TRUE", "low_is_trade": "TRUE", "close_is_trade": "TRUE",
    "high_is_quote_fallback": "FALSE", "low_is_quote_fallback": "FALSE", "close_is_bidask_average": "FALSE",
    "high_quote_fallback": "NULL", "low_quote_fallback": "NULL", "ohlc_source": "'ALPACA_SIP_BAR'",
    "ohlc_range_anomaly": "FALSE", "execution_ohlc_usable": "TRUE",
}

EVENT_OR_FORWARD_NULL = {
    "distcd","divamt","facpr","facshr","dclrdt","rcrddt","paydt","acperm","accomp",
    "nwperm","nwcomp","nextdt","disttype","disfacpr","disfacshr","dlret","dlprc","dlstcd",
}


def _find(cols:list[str], cands:tuple[str,...], required:bool=True)->str|None:
    m={c.lower():c for c in cols}
    for x in cands:
        if x in m:return m[x]
    if required: raise BridgeError(f"Required normalized field not found: one of {cands}; columns={cols}")
    return None


def _hist_hashes()->dict[str,str]:
    out={}
    for p in sorted((ROOT/'data'/'normalized').glob('year=*/*.parquet')):
        try: year=int(p.parent.name.split('=',1)[1])
        except Exception: continue
        if year <= 2025:
            out[str(p.relative_to(ROOT)).replace('\\','/')]=sha256_file(p)
    return out




def _check_unanchored_budget(total_candidates:int, unanchored:int, max_count:int=MAX_UNANCHORED_MAPPINGS, max_fraction:float=MAX_UNANCHORED_FRACTION)->float:
    if total_candidates < 0 or unanchored < 0:
        raise BridgeError("Anchor-budget counts must be nonnegative")
    if max_count < 0 or not (0.0 <= max_fraction <= 1.0):
        raise BridgeError("Invalid unanchored-mapping budget")
    frac = (float(unanchored) / float(total_candidates)) if total_candidates else 0.0
    if unanchored > max_count or frac > max_fraction:
        raise BridgeError(
            f"{unanchored} of {total_candidates} accepted mapping(s) lack a valid positive anchor/scale "
            f"({frac:.3%}), exceeding the safety budget of max_count={max_count} and "
            f"max_fraction={max_fraction:.3%}. Review {UNANCHORED_REPORT}; no mapping was guessed."
        )
    return frac

def build_staging(min_mapped_row_fraction:float=0.25)->dict[str,Any]:
    if not BRIDGE_MANIFEST.exists(): raise BridgeError("Bridge manifest missing; run the Foundation 2.2 monthly refresh or src.live.alpaca_crsp_bridge first")
    bm=json.loads(BRIDGE_MANIFEST.read_text(encoding="utf-8"))
    if bm.get("status")!="PASS": raise BridgeError("Bridge manifest is not PASS")
    if sha256_file(RAW)!=bm.get("raw_alpaca_sha256"): raise BridgeError("Raw Alpaca Parquet changed since bridge creation; rebuild the bridge")
    if not BRIDGE_CSV.exists() or not ANCHOR_MATCH_CSV.exists(): raise BridgeError("Bridge/anchor CSV missing")
    files=_normalized_files(); cols, types=_schema(files)
    date_col=_find(cols,DATE_CANDS); permno_col=_find(cols,PERMNO_CANDS); ticker_col=_find(cols,TICKER_CANDS)
    price_col=_find(cols,PRICE_CANDS); ret_col=_find(cols,RET_CANDS); vol_col=_find(cols,VOL_CANDS)
    open_col=_find(cols,OPEN_CANDS,False); high_col=_find(cols,HIGH_CANDS,False); low_col=_find(cols,LOW_CANDS,False); close_col=_find(cols,CLOSE_CANDS,False)
    latest_glob=ROOT/"data"/"normalized"/"year=2025"/"*.parquet"
    duckdb=_duckdb(); con=duckdb.connect()
    tmp=OUT.with_name(OUT.name+f".tmp-{uuid.uuid4().hex}"); OUT.parent.mkdir(parents=True,exist_ok=True)
    try:
        # Patch 1.20 accepted each ticker independently.  A late-2025 ticker change can
        # therefore leave both the historical alias and the year-end ticker mapped to
        # one PERMNO.  Canonicalize the existing bridge locally before the 2026 join so
        # users do not need to redownload December-2025 anchors just to repair identity.
        bridge_df = pd.read_csv(BRIDGE_CSV, dtype=str).fillna("")
        boundary_df = load_2025_permno_boundary()
        canonical_bridge_df, permno_resolution_df, inverse_stats = resolve_inverse_permno_collisions(bridge_df, boundary_df)
        if not permno_resolution_df.empty:
            PERMNO_RESOLUTION_REPORT.parent.mkdir(parents=True, exist_ok=True)
            tmp_report = PERMNO_RESOLUTION_REPORT.with_name(PERMNO_RESOLUTION_REPORT.name + ".tmp")
            permno_resolution_df.to_csv(tmp_report, index=False)
            os.replace(tmp_report, PERMNO_RESOLUTION_REPORT)
        else:
            PERMNO_RESOLUTION_REPORT.unlink(missing_ok=True)
        print(
            f"Bridge inverse-PERMNO canonicalization: groups={inverse_stats['permno_collision_groups']:,}; "
            f"rows={inverse_stats['permno_collision_rows']:,}; demoted={inverse_stats['permno_collision_rows_demoted']:,}; "
            f"unresolved={inverse_stats['permno_collision_groups_unresolved']:,}"
        )
        if int(inverse_stats.get("permno_collision_groups_unresolved", 0)) != 0:
            raise BridgeError(
                f"Existing bridge contains {inverse_stats['permno_collision_groups_unresolved']} unresolved inverse PERMNO collision group(s). "
                f"No arbitrary symbol was chosen. Review {PERMNO_RESOLUTION_REPORT}."
            )
        con.register("bridge_frame", canonical_bridge_df)
        con.execute("CREATE TEMP VIEW bridge_raw AS SELECT UPPER(TRIM(CAST(symbol AS VARCHAR))) symbol, TRY_CAST(permno AS BIGINT) permno, CAST(status AS VARCHAR) status, CAST(anchor_status AS VARCHAR) anchor_status FROM bridge_frame")
        con.execute("CREATE TEMP VIEW bridge_candidates AS SELECT symbol,permno,status,anchor_status FROM bridge_raw WHERE permno IS NOT NULL")
        con.execute(f"CREATE TEMP VIEW anchor AS SELECT UPPER(TRIM(symbol)) symbol, TRY_CAST(permno AS BIGINT) permno, TRY_CAST(anchor_date AS DATE) anchor_date, TRY_CAST(alpaca_close AS DOUBLE) anchor_close, TRY_CAST(price_scale AS DOUBLE) price_scale FROM read_csv_auto({_s(ANCHOR_MATCH_CSV)}, header=true, all_varchar=true)")

        # Fail closed before any large projection. Patch 1.20 joined raw26->bridge with
        # ON and then anchor with USING(symbol,permno), leaving duplicate symbol columns
        # on the left side for DuckDB to bind. Use fully qualified predicates throughout.
        bridge_dups = con.execute("SELECT COUNT(*) FROM (SELECT symbol, COUNT(*) n FROM bridge_candidates GROUP BY 1 HAVING COUNT(*)<>1)").fetchone()[0]
        anchor_dups = con.execute("SELECT COUNT(*) FROM (SELECT symbol,permno,COUNT(*) n FROM anchor GROUP BY 1,2 HAVING COUNT(*)<>1)").fetchone()[0]
        if int(bridge_dups) != 0:
            raise BridgeError(f"Accepted bridge candidates are not one-row-per-symbol: {bridge_dups} duplicate symbol group(s)")
        if int(anchor_dups) != 0:
            raise BridgeError(f"Anchor match file is not one-row-per-symbol/PERMNO: {anchor_dups} duplicate group(s)")

        con.execute("""
          CREATE TEMP VIEW anchor_valid AS
          SELECT symbol,permno,anchor_date,anchor_close,price_scale
          FROM anchor
          WHERE anchor_date IS NOT NULL AND anchor_close>0 AND price_scale>0
        """)
        total_bridge_candidates = int(con.execute("SELECT COUNT(*) FROM bridge_candidates").fetchone()[0])
        missing_anchor = int(con.execute("""
          SELECT COUNT(*)
          FROM bridge_candidates b
          LEFT JOIN anchor_valid a ON a.symbol=b.symbol AND a.permno=b.permno
          WHERE a.symbol IS NULL
        """).fetchone()[0])
        if missing_anchor:
            UNANCHORED_REPORT.parent.mkdir(parents=True, exist_ok=True)
            con.execute(f"""
              COPY (
                SELECT
                  b.symbol,b.permno,b.status,b.anchor_status,
                  a.anchor_date,a.anchor_close,a.price_scale,
                  CASE
                    WHEN a.symbol IS NULL THEN 'NO_ANCHOR_MATCH_ROW'
                    WHEN a.anchor_date IS NULL THEN 'NULL_ANCHOR_DATE'
                    WHEN a.anchor_close IS NULL OR a.anchor_close<=0 THEN 'NONPOSITIVE_ALPACA_ANCHOR_CLOSE'
                    WHEN a.price_scale IS NULL OR a.price_scale<=0 THEN 'NONPOSITIVE_PRICE_SCALE'
                    ELSE 'INVALID_ANCHOR'
                  END AS exclusion_reason
                FROM bridge_candidates b
                LEFT JOIN anchor a ON a.symbol=b.symbol AND a.permno=b.permno
                WHERE NOT EXISTS (
                  SELECT 1 FROM anchor_valid v WHERE v.symbol=b.symbol AND v.permno=b.permno
                )
                ORDER BY b.permno,b.symbol
              ) TO {_s(UNANCHORED_REPORT)} (HEADER, DELIMITER ',')
            """)
        else:
            UNANCHORED_REPORT.unlink(missing_ok=True)
        unanchored_fraction = _check_unanchored_budget(total_bridge_candidates, missing_anchor)
        print(
            f"Anchor eligibility: candidates={total_bridge_candidates:,}; valid={total_bridge_candidates-missing_anchor:,}; "
            f"excluded_unanchored={missing_anchor:,} ({unanchored_fraction:.3%})"
        )

        # Unanchored mappings are not guessed or rescaled from another alias. They are
        # excluded from the forward staging universe and preserved in an audit CSV.
        con.execute("""
          CREATE TEMP VIEW anchor_bridge AS
          SELECT b.symbol,b.permno,b.status,a.anchor_date,a.anchor_close,a.price_scale
          FROM bridge_candidates b
          JOIN anchor_valid a
            ON a.symbol=b.symbol AND a.permno=b.permno
        """)

        con.execute(f"CREATE TEMP VIEW hist AS SELECT * FROM read_parquet({_s(latest_glob)}, union_by_name=true, hive_partitioning=false)")
        con.execute(f"""
          CREATE TEMP VIEW meta AS
          SELECT * EXCLUDE(rn) FROM (
            SELECT *, ROW_NUMBER() OVER(
              PARTITION BY TRY_CAST({_q(permno_col)} AS BIGINT)
              ORDER BY TRY_CAST({_q(date_col)} AS DATE) DESC
            ) rn
            FROM hist
            WHERE TRY_CAST({_q(permno_col)} AS BIGINT) IS NOT NULL
          ) WHERE rn=1
        """)
        con.execute(f"CREATE TEMP VIEW raw26 AS SELECT * FROM read_parquet({_s(RAW)})")
        con.execute(f"""
          CREATE TEMP VIEW ytd_joined AS
          SELECT
            ab.symbol,
            ab.permno,
            TRY_CAST(r.date AS DATE) AS date,
            TRY_CAST(r.close AS DOUBLE) AS close,
            ab.price_scale,
            TRY_CAST(r.open AS DOUBLE) AS open,
            TRY_CAST(r.high AS DOUBLE) AS high,
            TRY_CAST(r.low AS DOUBLE) AS low,
            TRY_CAST(r.volume AS DOUBLE) AS volume,
            NULLIF(TRIM(CAST(r.asset_id AS VARCHAR)), '') AS asset_id,
            LOWER(NULLIF(TRIM(CAST(r.asset_status AS VARCHAR)), '')) AS asset_status
          FROM raw26 r
          JOIN anchor_bridge ab
            ON UPPER(TRIM(CAST(r.symbol AS VARCHAR)))=ab.symbol
        """)
        joined_rows, joined_permnos, joined_dups = con.execute("""
          WITH dup AS (
            SELECT date,permno,COUNT(*) n
            FROM ytd_joined
            GROUP BY 1,2
            HAVING COUNT(*)>1
          )
          SELECT COUNT(*),COUNT(DISTINCT permno),(SELECT COUNT(*) FROM dup)
          FROM ytd_joined
        """).fetchone()
        if int(joined_rows) <= 0 or int(joined_permnos) <= 0:
            raise BridgeError("Alpaca raw rows did not join to any accepted anchored PERMNO")

        # Guardrail context: multiple Alpaca symbols resolve to the same PERMNO on overlapping dates in some alias/rename cases; review the bridge rather than deduplicating when payloads conflict.
        # Multiple Alpaca tickers can legitimately refer to the same underlying Alpaca
        # asset across a rename/alias boundary.  Never keep-first.  For overlapping
        # date/PERMNO observations, collapse only when BOTH vendor identity and the
        # scaled OHLCV payload prove equivalence.  Anything else remains fail-closed.
        equivalent_dups = 0
        conflicting_dups = 0
        if int(joined_dups) > 0:
            DUP_REPORT.parent.mkdir(parents=True, exist_ok=True)
            con.execute("""
              CREATE TEMP VIEW duplicate_group_stats AS
              SELECT
                date,permno,COUNT(*) AS duplicate_count,
                COUNT(DISTINCT asset_id) FILTER (WHERE asset_id IS NOT NULL) AS asset_id_count,
                COUNT(*) FILTER (WHERE asset_id IS NULL) AS null_asset_id_rows,
                MIN(close*price_scale) AS scaled_close_min, MAX(close*price_scale) AS scaled_close_max,
                MIN(open*price_scale) AS scaled_open_min, MAX(open*price_scale) AS scaled_open_max,
                MIN(high*price_scale) AS scaled_high_min, MAX(high*price_scale) AS scaled_high_max,
                MIN(low*price_scale) AS scaled_low_min, MAX(low*price_scale) AS scaled_low_max,
                MIN(volume) AS volume_min, MAX(volume) AS volume_max
              FROM ytd_joined
              GROUP BY 1,2
              HAVING COUNT(*)>1
            """)
            con.execute("""
              CREATE TEMP VIEW duplicate_group_classification AS
              SELECT *,
                CASE
                  WHEN asset_id_count=1 AND null_asset_id_rows=0
                   AND ABS(scaled_close_max-scaled_close_min) <= GREATEST(1e-8, 1e-8*GREATEST(ABS(scaled_close_max),ABS(scaled_close_min)))
                   AND ABS(scaled_open_max-scaled_open_min) <= GREATEST(1e-8, 1e-8*GREATEST(ABS(scaled_open_max),ABS(scaled_open_min)))
                   AND ABS(scaled_high_max-scaled_high_min) <= GREATEST(1e-8, 1e-8*GREATEST(ABS(scaled_high_max),ABS(scaled_high_min)))
                   AND ABS(scaled_low_max-scaled_low_min) <= GREATEST(1e-8, 1e-8*GREATEST(ABS(scaled_low_max),ABS(scaled_low_min)))
                   AND ABS(volume_max-volume_min) <= 0.5
                  THEN 'EQUIVALENT_VENDOR_ALIAS'
                  ELSE 'CONFLICTING_OBSERVATION'
                END AS classification
              FROM duplicate_group_stats
            """)
            equivalent_dups, conflicting_dups = con.execute("""
              SELECT
                COUNT(*) FILTER (WHERE classification='EQUIVALENT_VENDOR_ALIAS'),
                COUNT(*) FILTER (WHERE classification='CONFLICTING_OBSERVATION')
              FROM duplicate_group_classification
            """).fetchone()
            con.execute(f"""
              COPY (
                SELECT y.date,y.permno,y.symbol,y.asset_id,y.asset_status,y.close,y.open,y.high,y.low,y.volume,y.price_scale,
                       c.duplicate_count,c.classification
                FROM ytd_joined y
                JOIN duplicate_group_classification c USING(date,permno)
                ORDER BY y.date,y.permno,y.symbol
              ) TO {_s(DUP_REPORT)} (HEADER, DELIMITER ',')
            """)
            con.execute(f"""
              COPY (
                SELECT y.date,y.permno,y.symbol,y.asset_id,y.asset_status,y.close,y.open,y.high,y.low,y.volume,y.price_scale,
                       c.duplicate_count,c.asset_id_count,c.null_asset_id_rows,c.scaled_close_min,c.scaled_close_max,
                       c.scaled_open_min,c.scaled_open_max,c.scaled_high_min,c.scaled_high_max,c.scaled_low_min,c.scaled_low_max,
                       c.volume_min,c.volume_max
                FROM ytd_joined y
                JOIN duplicate_group_classification c USING(date,permno)
                WHERE c.classification='CONFLICTING_OBSERVATION'
                ORDER BY y.date,y.permno,y.symbol
              ) TO {_s(CONFLICT_REPORT)} (HEADER, DELIMITER ',')
            """)
            if int(conflicting_dups) != 0:
                raise BridgeError(
                    f"Raw Alpaca -> bridge join creates {joined_dups} duplicate date/PERMNO group(s): "
                    f"{equivalent_dups} are provably equivalent same-asset aliases, but {conflicting_dups} conflict. "
                    f"No arbitrary deduplication was performed. Review {CONFLICT_REPORT}."
                )

        # Select one row only for duplicate groups already proven equivalent.  Prefer an
        # active Alpaca alias, then the latest 2025 CRSP ticker, then the alias with more
        # 2026 observations, then lexical order for deterministic reproducibility.
        con.execute(f"""
          CREATE TEMP VIEW ytd_ranked AS
          WITH counts AS (
            SELECT permno,symbol,COUNT(*) AS symbol_rows
            FROM ytd_joined GROUP BY 1,2
          )
          SELECT y.*,
                 COUNT(*) OVER(PARTITION BY y.date,y.permno) AS duplicate_count,
                 ROW_NUMBER() OVER(
                   PARTITION BY y.date,y.permno
                   ORDER BY
                     CASE WHEN y.asset_status='active' THEN 1 ELSE 0 END DESC,
                     CASE WHEN UPPER(TRIM(CAST(m.{_q(ticker_col)} AS VARCHAR)))=y.symbol THEN 1 ELSE 0 END DESC,
                     c.symbol_rows DESC,
                     y.symbol ASC
                 ) AS alias_rank
          FROM ytd_joined y
          JOIN counts c ON c.permno=y.permno AND c.symbol=y.symbol
          JOIN meta m ON TRY_CAST(m.{_q(permno_col)} AS BIGINT)=y.permno
        """)
        con.execute("""
          CREATE TEMP VIEW ytd_resolved AS
          SELECT * EXCLUDE(duplicate_count,alias_rank)
          FROM ytd_ranked
          WHERE duplicate_count=1 OR alias_rank=1
        """)
        if int(joined_dups) > 0:
            con.execute(f"""
              COPY (
                SELECT
                  r.date,r.permno,r.symbol AS chosen_symbol,r.asset_id,r.asset_status,
                  string_agg(j.symbol, '|' ORDER BY j.symbol) AS overlapping_aliases,
                  COUNT(*) AS duplicate_count,
                  'same asset_id + equivalent scaled OHLCV; deterministic alias preference' AS resolution_reason
                FROM ytd_ranked r
                JOIN ytd_joined j ON j.date=r.date AND j.permno=r.permno
                WHERE r.duplicate_count>1 AND r.alias_rank=1
                GROUP BY r.date,r.permno,r.symbol,r.asset_id,r.asset_status
                ORDER BY r.date,r.permno
              ) TO {_s(ALIAS_REPORT)} (HEADER, DELIMITER ',')
            """)
        else:
            ALIAS_REPORT.unlink(missing_ok=True); DUP_REPORT.unlink(missing_ok=True); CONFLICT_REPORT.unlink(missing_ok=True)

        resolved_rows, resolved_dups = con.execute("""
          WITH d AS (SELECT date,permno,COUNT(*) n FROM ytd_resolved GROUP BY 1,2 HAVING COUNT(*)>1)
          SELECT COUNT(*),(SELECT COUNT(*) FROM d) FROM ytd_resolved
        """).fetchone()
        if int(resolved_dups) != 0:
            raise BridgeError(f"Alias resolution invariant failed: {resolved_dups} duplicate date/PERMNO group(s) remain")

        con.execute("""
          CREATE TEMP VIEW seq AS
          SELECT
            ab.symbol,ab.permno,ab.anchor_date AS date,ab.anchor_close AS close,ab.price_scale,
            NULL::DOUBLE AS open,NULL::DOUBLE AS high,NULL::DOUBLE AS low,NULL::DOUBLE AS volume,0 AS is_ytd
          FROM anchor_bridge ab
          UNION ALL
          SELECT
            y.symbol,y.permno,y.date,y.close,y.price_scale,y.open,y.high,y.low,y.volume,1 AS is_ytd
          FROM ytd_resolved y
        """)
        con.execute("""
          CREATE TEMP VIEW dyn AS
          SELECT *, close/LAG(close) OVER(PARTITION BY permno ORDER BY date)-1.0 AS alpaca_ret
          FROM seq
        """)
        special={
            date_col: "d.date", permno_col:"d.permno", ticker_col:"d.symbol",
            price_col:"d.close*d.price_scale", ret_col:"d.alpaca_ret", vol_col:"d.volume",
        }
        if open_col:special[open_col]="d.open*d.price_scale"
        if high_col:special[high_col]="d.high*d.price_scale"
        if low_col:special[low_col]="d.low*d.price_scale"
        if close_col and close_col != price_col:special[close_col]="d.close*d.price_scale"
        by_lower={c.lower():c for c in cols}
        for canonical,base in PROVENANCE_FORWARD_SPECIALS.items():
            if canonical in by_lower: special[by_lower[canonical]]=base
        expr=[]
        for c in cols:
            typ=types[c]; cl=c.lower()
            if c in special: base=special[c]
            elif cl in EVENT_OR_FORWARD_NULL: base="NULL"
            elif cl in {"cfacpr","cfacshr"}: base="1.0"
            else: base=f"m.{_q(c)}"
            expr.append(f"TRY_CAST({base} AS {typ}) AS {_q(c)}")
        projection=",\n              ".join(expr)
        con.execute(f"""
          COPY (
            SELECT {projection}
            FROM dyn d
            JOIN meta m ON TRY_CAST(m.{_q(permno_col)} AS BIGINT)=d.permno
            WHERE d.is_ytd=1
            ORDER BY TRY_CAST(d.date AS DATE), d.permno
          ) TO {_s(tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        desc=con.execute(f"DESCRIBE SELECT * FROM read_parquet({_s(tmp)}, hive_partitioning=false)").fetchall()
        out_cols=[str(r[0]) for r in desc]
        if out_cols!=cols: raise BridgeError(f"Staging schema columns differ from historical normalized schema: {out_cols} vs {cols}")
        stats=con.execute(f"""
          WITH x AS (SELECT * FROM read_parquet({_s(tmp)}, hive_partitioning=false)),
          d AS (SELECT TRY_CAST({_q(date_col)} AS DATE) date, TRY_CAST({_q(permno_col)} AS BIGINT) permno,
                       TRY_CAST({_q(price_col)} AS DOUBLE) price, TRY_CAST({_q(ret_col)} AS DOUBLE) ret,
                       TRY_CAST({_q(vol_col)} AS DOUBLE) vol FROM x),
          dup AS (SELECT date,permno,COUNT(*) n FROM d GROUP BY 1,2 HAVING COUNT(*)>1)
          SELECT COUNT(*),COUNT(DISTINCT permno),MIN(date),MAX(date),
                 COUNT(*) FILTER(WHERE date IS NULL OR permno IS NULL),
                 COUNT(*) FILTER(WHERE price IS NULL OR price<=0),
                 COUNT(*) FILTER(WHERE ret IS NULL OR NOT isfinite(ret)),
                 COUNT(*) FILTER(WHERE vol IS NULL OR vol<0),
                 (SELECT COUNT(*) FROM dup)
          FROM d
        """).fetchone()
        ohlc_stats=(0,0,0,0,0,0,0,0)
        provenance_stats=(0,0,0,0)
        if open_col and high_col and low_col and close_col:
            ohlc_stats=con.execute(f"""
              SELECT
                COUNT(*) FILTER(WHERE TRY_CAST({_q(open_col)} AS DOUBLE) IS NULL OR TRY_CAST({_q(open_col)} AS DOUBLE)<=0),
                COUNT(*) FILTER(WHERE TRY_CAST({_q(high_col)} AS DOUBLE) IS NULL OR TRY_CAST({_q(high_col)} AS DOUBLE)<=0),
                COUNT(*) FILTER(WHERE TRY_CAST({_q(low_col)} AS DOUBLE) IS NULL OR TRY_CAST({_q(low_col)} AS DOUBLE)<=0),
                COUNT(*) FILTER(WHERE TRY_CAST({_q(close_col)} AS DOUBLE) IS NULL OR TRY_CAST({_q(close_col)} AS DOUBLE)<=0),
                COUNT(*) FILTER(WHERE TRY_CAST({_q(low_col)} AS DOUBLE)>TRY_CAST({_q(high_col)} AS DOUBLE)),
                COUNT(*) FILTER(WHERE TRY_CAST({_q(open_col)} AS DOUBLE)<TRY_CAST({_q(low_col)} AS DOUBLE) OR TRY_CAST({_q(open_col)} AS DOUBLE)>TRY_CAST({_q(high_col)} AS DOUBLE)),
                COUNT(*) FILTER(WHERE TRY_CAST({_q(close_col)} AS DOUBLE)<TRY_CAST({_q(low_col)} AS DOUBLE) OR TRY_CAST({_q(close_col)} AS DOUBLE)>TRY_CAST({_q(high_col)} AS DOUBLE)),
                COUNT(*) FILTER(WHERE NOT isfinite(TRY_CAST({_q(close_col)} AS DOUBLE)))
              FROM read_parquet({_s(tmp)}, hive_partitioning=false)
            """).fetchone()
            if any(int(x)!=0 for x in ohlc_stats):
                raise BridgeError(
                    f"Canonical 2026 OHLC validation failed: bad_open={ohlc_stats[0]}, bad_high={ohlc_stats[1]}, "
                    f"bad_low={ohlc_stats[2]}, bad_close={ohlc_stats[3]}, low_gt_high={ohlc_stats[4]}, "
                    f"open_outside_range={ohlc_stats[5]}, close_outside_range={ohlc_stats[6]}, nonfinite_close={ohlc_stats[7]}"
                )
            prov_required=[x for x in PROVENANCE_FORWARD_SPECIALS if x in by_lower]
            if prov_required:
                if len(prov_required)!=len(PROVENANCE_FORWARD_SPECIALS):
                    missing=sorted(set(PROVENANCE_FORWARD_SPECIALS)-set(prov_required))
                    raise BridgeError(f"Historical template has partial OHLC provenance schema; missing={missing}")
                provenance_stats=con.execute(f"""
                  SELECT
                    COUNT(*) FILTER(WHERE NOT COALESCE(TRY_CAST({_q(by_lower['open_is_trade'])} AS BOOLEAN),FALSE)
                                         OR NOT COALESCE(TRY_CAST({_q(by_lower['high_is_trade'])} AS BOOLEAN),FALSE)
                                         OR NOT COALESCE(TRY_CAST({_q(by_lower['low_is_trade'])} AS BOOLEAN),FALSE)
                                         OR NOT COALESCE(TRY_CAST({_q(by_lower['close_is_trade'])} AS BOOLEAN),FALSE)),
                    COUNT(*) FILTER(WHERE COALESCE(TRY_CAST({_q(by_lower['high_is_quote_fallback'])} AS BOOLEAN),FALSE)
                                         OR COALESCE(TRY_CAST({_q(by_lower['low_is_quote_fallback'])} AS BOOLEAN),FALSE)
                                         OR COALESCE(TRY_CAST({_q(by_lower['close_is_bidask_average'])} AS BOOLEAN),FALSE)),
                    COUNT(*) FILTER(WHERE {_q(by_lower['high_quote_fallback'])} IS NOT NULL OR {_q(by_lower['low_quote_fallback'])} IS NOT NULL),
                    COUNT(*) FILTER(WHERE CAST({_q(by_lower['ohlc_source'])} AS VARCHAR)<>'ALPACA_SIP_BAR')
                  FROM read_parquet({_s(tmp)}, hive_partitioning=false)
                """).fetchone()
                if any(int(x)!=0 for x in provenance_stats):
                    raise BridgeError(f"2026 Alpaca OHLC provenance validation failed: {provenance_stats}")
        raw_rows=int(con.execute(f"SELECT COUNT(*) FROM read_parquet({_s(RAW)})").fetchone()[0])
    finally: con.close()
    rows,symbols,mind,maxd,nullkeys,badprice,badret,badvol,dups=stats
    mapped_fraction=float(rows)/raw_rows if raw_rows else 0.0
    if int(rows)<=0 or int(symbols)<=0: tmp.unlink(missing_ok=True); raise BridgeError("Normalized staging is empty")
    if any(int(x)!=0 for x in (nullkeys,badprice,badret,badvol,dups)):
        tmp.unlink(missing_ok=True); raise BridgeError(f"Normalized staging validation failed: nullkeys={nullkeys}, badprice={badprice}, badret={badret}, badvol={badvol}, duplicate_groups={dups}")
    if not mind or mind.year!=2026 or not maxd or maxd.year!=2026:
        tmp.unlink(missing_ok=True); raise BridgeError(f"Normalized staging date bounds invalid: {mind}..{maxd}")
    if mapped_fraction < min_mapped_row_fraction:
        tmp.unlink(missing_ok=True); raise BridgeError(f"Mapped normalized rows are only {mapped_fraction:.2%} of raw Alpaca rows, below {min_mapped_row_fraction:.2%}; inspect bridge diagnostics rather than silently shrinking the universe")
    os.replace(tmp,OUT)
    raw_manifest=json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    payload={
        "schema_version":4,"status":"PASS","generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_raw":str(RAW),"source_raw_sha256":sha256_file(RAW),"bridge_manifest_sha256":sha256_file(BRIDGE_MANIFEST),
        "historical_template_year":2025,"historical_schema_columns":cols,"resolved_columns":{"date":date_col,"permno":permno_col,"ticker":ticker_col,"price":price_col,"return":ret_col,"volume":vol_col,"open":open_col,"high":high_col,"low":low_col,"close":close_col},
        "rows":int(rows),"permnos":int(symbols),"first_date":mind.isoformat(),"last_date":maxd.isoformat(),"raw_rows":raw_rows,"mapped_row_fraction":mapped_fraction,
        "preflight":{"inverse_permno_resolution":inverse_stats,"inverse_permno_resolution_csv":str(PERMNO_RESOLUTION_REPORT) if PERMNO_RESOLUTION_REPORT.exists() else None,"accepted_bridge_symbol_duplicate_groups":int(bridge_dups),"anchor_symbol_permno_duplicate_groups":int(anchor_dups),"accepted_bridge_candidates":int(total_bridge_candidates),"accepted_bridge_rows_excluded_unanchored":int(missing_anchor),"unanchored_mapping_fraction":unanchored_fraction,"unanchored_mapping_budget":{"max_count":MAX_UNANCHORED_MAPPINGS,"max_fraction":MAX_UNANCHORED_FRACTION},"unanchored_mapping_report":str(UNANCHORED_REPORT) if UNANCHORED_REPORT.exists() else None,"joined_ytd_rows":int(joined_rows),"joined_ytd_permnos":int(joined_permnos),"joined_ytd_duplicate_date_permno_groups":int(joined_dups),"equivalent_alias_duplicate_groups":int(equivalent_dups),"conflicting_duplicate_groups":int(conflicting_dups),"resolved_ytd_rows":int(resolved_rows),"resolved_duplicate_groups_remaining":int(resolved_dups)},
        "duplicate_date_permno_groups":int(dups),"null_key_rows":int(nullkeys),"invalid_price_rows":int(badprice),"invalid_return_rows":int(badret),"invalid_volume_rows":int(badvol),
        "canonical_ohlc_validation":{"enabled":bool(open_col and high_col and low_col and close_col),"bad_open":int(ohlc_stats[0]),"bad_high":int(ohlc_stats[1]),"bad_low":int(ohlc_stats[2]),"bad_close":int(ohlc_stats[3]),"low_gt_high":int(ohlc_stats[4]),"open_outside_range":int(ohlc_stats[5]),"close_outside_range":int(ohlc_stats[6]),"nonfinite_close":int(ohlc_stats[7])},
        "ohlc_provenance_validation":{"template_has_full_schema":all(k in {c.lower() for c in cols} for k in PROVENANCE_FORWARD_SPECIALS),"invalid_trade_flags":int(provenance_stats[0]),"invalid_quote_flags":int(provenance_stats[1]),"unexpected_quote_values":int(provenance_stats[2]),"wrong_source":int(provenance_stats[3])},
        "staging_parquet":str(OUT),"staging_sha256":sha256_file(OUT),"historical_normalized_hashes_before":_hist_hashes(),
        "price_bridge":"Alpaca adjusted OHLC is multiplied by a per-symbol scale calibrated on the latest same-date December 2025 CRSP/Alpaca anchor; relative Alpaca price path is preserved while the CRSP price level remains continuous at the vendor boundary.",
        "return_bridge":"2026 return field is computed from the Alpaca adjusted close ratio, including the December 2025 Alpaca anchor as the lag for the first 2026 bar.",
        "metadata_bridge":"non-market normalized fields are carried from the latest 2025 CRSP PERMNO observation; event/delisting fields are nulled; cfacpr/cfacshr are set to 1 when present because the bridged market-price series is already adjusted/scaled.",
        "alpaca_adjustment":raw_manifest.get("adjustment"),"alpaca_feed":raw_manifest.get("feed_used"),
        "duplicate_resolution":{"policy":"collapse only overlapping date/PERMNO observations that share one non-null Alpaca asset_id and equivalent scaled OHLCV; conflicting observations fail closed","all_duplicate_rows_csv":str(DUP_REPORT) if DUP_REPORT.exists() else None,"alias_resolution_csv":str(ALIAS_REPORT) if ALIAS_REPORT.exists() else None,"conflicting_rows_csv":str(CONFLICT_REPORT) if CONFLICT_REPORT.exists() else None},
        "guardrail":"This is a staging artifact only. It does not modify data/normalized until src.live.install_alpaca_2026_normalized passes its independent checks.",
    }
    MANIFEST.parent.mkdir(parents=True,exist_ok=True); tmpm=MANIFEST.with_name(MANIFEST.name+".tmp"); tmpm.write_text(json.dumps(payload,indent=2),encoding="utf-8"); os.replace(tmpm,MANIFEST)
    return payload


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--min-mapped-row-fraction",type=float,default=0.25); args=ap.parse_args()
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open("w",encoding="utf-8",buffering=1) as log:
        oldout,olderr=sys.stdout,sys.stderr
        sys.stdout=Tee(oldout,log); sys.stderr=Tee(olderr,log)
        try:
            print("="*78); print("ALPACA 2026 NORMALIZED STAGING BUILD"); print("="*78)
            p=build_staging(args.min_mapped_row_fraction)
            print(f"Rows: {p['rows']:,}; PERMNOs: {p['permnos']:,}; mapped raw rows: {p['mapped_row_fraction']:.2%}")
            print(f"Dates: {p['first_date']} -> {p['last_date']}")
            print(f"Staging: {p['staging_parquet']}"); print(f"Manifest: {MANIFEST}"); print(f"Log: {LOG}"); print("="*78)
        except Exception:
            print("="*78); print("ALPACA 2026 NORMALIZED STAGING BUILD FAILED"); traceback.print_exc(); print(f"Full log: {LOG}"); print("="*78)
            raise
        finally:
            sys.stdout=oldout; sys.stderr=olderr

if __name__=="__main__": main()
