from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from src.common.config import load_config, root_path
from src.common.logging_utils import log


BASE_REQUIRED_COLUMNS = {
    "date", "permno", "ticker", "close", "ret1", "ret5", "adv20", "sma50",
    "sma100", "sma200", "momentum_3_1", "momentum_6_1", "momentum_12_1",
    "trend_strength", "rsi2", "sector",
}
ELIGIBLE_REQUIRED_COLUMNS = BASE_REQUIRED_COLUMNS | {"eligible"}
# 550 calendar days comfortably exceeds the 252-trading-day longest feature window.
WARMUP_CALENDAR_DAYS = 550
MIN_PARQUET_BYTES = 64


def q(s):
    return "'" + str(s).replace("'", "''") + "'"


def _files_for_base(out: Path) -> list[Path]:
    return sorted(out.glob("year=*/*.parquet"))


def _files_for_eligible(out: Path) -> list[Path]:
    return sorted((out / "eligible").glob("eligible_year=*/*.parquet"))


def _status(out: Path, manifest: Path) -> dict:
    base = _files_for_base(out) if out.exists() else []
    eligible = _files_for_eligible(out) if out.exists() else []
    return {
        "manifest_exists": manifest.exists(),
        "base_files": len(base),
        "eligible_files": len(eligible),
        "base_ready": bool(base),
        "eligible_ready": bool(eligible),
    }


def _configure_duckdb(con, cfg: dict, temp: Path) -> None:
    comp = cfg.get("compute", {})
    con.execute(f"SET threads={int(comp.get('threads', 4))}")
    if comp.get("memory_limit"):
        con.execute(f"SET memory_limit={q(comp['memory_limit'])}")
    con.execute(f"SET temp_directory={q(str(temp))}")
    con.execute(
        "SET preserve_insertion_order="
        + ("true" if comp.get("preserve_insertion_order", False) else "false")
    )


def _normalized_stats(con, norm: Path) -> tuple:
    glob = str(norm).replace("\\", "/") + "/**/*.parquet"
    return con.execute(
        f"SELECT MIN(date), MAX(date), COUNT(*) FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true)"
    ).fetchone()


def _expected_years(con, cfg: dict, norm: Path) -> list[int]:
    n = _normalized_stats(con, norm)
    if n[0] is None or n[1] is None:
        raise RuntimeError("Normalized parquet store contains no rows.")
    first = int(str(cfg["data"]["feature_start_date"])[:4])
    last = int(str(n[1])[:4])
    if last < first:
        raise RuntimeError(f"Normalized data ends in {last}, before feature start year {first}.")
    return list(range(first, last + 1))


def _year_from_parent(path: Path, prefix: str) -> int | None:
    name = path.parent.name
    if not name.startswith(prefix):
        return None
    try:
        return int(name.split("=", 1)[1])
    except Exception:
        return None


def _parquet_schema(con, path: Path) -> set[str]:
    cur = con.execute(
        f"SELECT * FROM read_parquet({q(str(path).replace('\\','/'))}, union_by_name=true) LIMIT 0"
    )
    return {d[0] for d in cur.description}


def _validate_parquet_file(
    con,
    path: Path,
    expected_year: int,
    required_columns: set[str],
) -> tuple[bool, str, tuple | None]:
    if not path.exists():
        return False, "missing", None
    try:
        size = path.stat().st_size
    except OSError as e:
        return False, f"stat_failed:{e}", None
    if size < MIN_PARQUET_BYTES:
        return False, f"too_small:{size}_bytes", None
    try:
        cols = _parquet_schema(con, path)
        missing = sorted(required_columns.difference(cols))
        if missing:
            return False, "missing_columns:" + ",".join(missing), None
        p = str(path).replace("\\", "/")
        stats = con.execute(
            f"SELECT COUNT(*), MIN(date), MAX(date) FROM read_parquet({q(p)}, union_by_name=true)"
        ).fetchone()
        rows, min_date, max_date = int(stats[0]), stats[1], stats[2]
        if rows <= 0 or min_date is None or max_date is None:
            return False, "empty", stats
        if int(str(min_date)[:4]) != expected_year or int(str(max_date)[:4]) != expected_year:
            return False, f"wrong_date_range:{min_date}..{max_date}", stats
        return True, "ok", stats
    except Exception as e:
        return False, f"unreadable:{type(e).__name__}:{e}", None


def _scan_base_partitions(con, cfg: dict, norm: Path, out: Path) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for year in _expected_years(con, cfg, norm):
        path = out / f"year={year}" / "data_0.parquet"
        ok, reason, stats = _validate_parquet_file(con, path, year, BASE_REQUIRED_COLUMNS)
        result[year] = {"path": path, "ok": ok, "reason": reason, "stats": stats}
    return result


def _scan_eligible_partitions(con, cfg: dict, norm: Path, out: Path) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for year in _expected_years(con, cfg, norm):
        path = out / "eligible" / f"eligible_year={year}" / "data_0.parquet"
        ok, reason, stats = _validate_parquet_file(con, path, year, ELIGIBLE_REQUIRED_COLUMNS)
        result[year] = {"path": path, "ok": ok, "reason": reason, "stats": stats}
    return result


def _log_bad_partitions(label: str, scan: dict[int, dict]) -> None:
    bad = [(y, d["reason"]) for y, d in scan.items() if not d["ok"]]
    if not bad:
        return
    log(f"{label}: {len(bad)} missing/corrupt year partition(s):")
    for year, reason in bad:
        # Keep errors readable on Windows console; full details remain in log file.
        text = str(reason).replace("\n", " ")
        if len(text) > 260:
            text = text[:257] + "..."
        log(f"  {year}: {text}")


def _aggregate_partition_stats(scan: dict[int, dict]) -> tuple | None:
    valid = [d["stats"] for d in scan.values() if d["ok"] and d["stats"] is not None]
    if not valid or len(valid) != len(scan):
        return None
    rows = sum(int(s[0]) for s in valid)
    min_date = min(s[1] for s in valid)
    max_date = max(s[2] for s in valid)
    return min_date, max_date, rows, len(valid)


def _base_stats(con, cfg: dict, norm: Path, out: Path) -> tuple | None:
    scan = _scan_base_partitions(con, cfg, norm, out)
    agg = _aggregate_partition_stats(scan)
    if agg is None:
        return None
    return agg


def _eligible_stats(con, cfg: dict, norm: Path, out: Path) -> tuple | None:
    scan = _scan_eligible_partitions(con, cfg, norm, out)
    agg = _aggregate_partition_stats(scan)
    if agg is None:
        return None
    min_date, max_date, rows, year_count = agg
    eligible_rows = 0
    for year, d in scan.items():
        p = str(d["path"]).replace("\\", "/")
        eligible_rows += int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet({q(p)}, union_by_name=true) WHERE eligible"
            ).fetchone()[0]
        )
    return min_date, max_date, rows, eligible_rows, year_count


def _base_is_current(con, cfg: dict, norm: Path, out: Path) -> bool:
    scan = _scan_base_partitions(con, cfg, norm, out)
    return bool(scan) and all(d["ok"] for d in scan.values())


def _eligible_is_current(con, cfg: dict, norm: Path, out: Path) -> bool:
    bscan = _scan_base_partitions(con, cfg, norm, out)
    escan = _scan_eligible_partitions(con, cfg, norm, out)
    if not bscan or set(bscan) != set(escan):
        return False
    for year in bscan:
        b = bscan[year]
        e = escan[year]
        if not b["ok"] or not e["ok"]:
            return False
        if int(b["stats"][0]) != int(e["stats"][0]):
            return False
    return True


def _atomic_replace_dir(staged_dir: Path, final_dir: Path) -> None:
    backup = final_dir.with_name(final_dir.name + ".repair_backup")
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    if final_dir.exists():
        final_dir.replace(backup)
    try:
        staged_dir.replace(final_dir)
    except Exception:
        if backup.exists() and not final_dir.exists():
            backup.replace(final_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def _build_base_year(con, cfg: dict, norm: Path, out: Path, year: int) -> None:
    import pandas as pd

    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")
    warmup_start = pd.Timestamp(year_start) - pd.to_timedelta(WARMUP_CALENDAR_DAYS, unit="D")

    glob = str(norm).replace("\\", "/") + "/**/*.parquet"
    stage_root = out / ".repair_tmp"
    staged_dir = stage_root / f"year={year}"
    staged_file = staged_dir / "data_0.parquet"
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"Rebuilding base feature year {year} atomically "
        f"(warm-up starts {warmup_start.date()})..."
    )
    sql = f"""
    COPY (
      WITH source AS (
        SELECT date,permno,ticker,close,ret,volume,dollar_volume,primary_exch,
               security_type,security_subtype,share_type,sic
        FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true)
        WHERE date BETWEEN DATE {q(str(warmup_start.date()))} AND DATE {q(str(year_end.date()))}
      ), base0 AS (
        SELECT date,permno,ticker,close,ret,volume,dollar_volume,primary_exch,security_type,security_subtype,share_type,sic,
               COALESCE(ret, close/LAG(close,1) OVER w - 1.0) AS ret1,
               COUNT(close) OVER w AS history_days,
               LAG(close,1) OVER w AS close_lag1,
               LAG(close,2) OVER w AS close_lag2,
               LAG(close,3) OVER w AS close_lag3,
               LAG(close,5) OVER w AS close_lag5,
               LAG(close,21) OVER w AS close_lag21,
               LAG(close,63) OVER w AS close_lag63,
               LAG(close,126) OVER w AS close_lag126,
               LAG(close,252) OVER w AS close_lag252
        FROM source
        WINDOW w AS (PARTITION BY permno ORDER BY date)
      ), b AS (
        SELECT *,
               AVG(close) OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) sma10,
               AVG(close) OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) sma20,
               AVG(close) OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) sma50,
               AVG(close) OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) sma100,
               AVG(close) OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) sma200,
               AVG(dollar_volume) OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) adv20,
               STDDEV_SAMP(ret1) OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) * SQRT(252.0) vol20,
               MAX(close) OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) high20,
               MAX(close) OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) high252,
               AVG(GREATEST(COALESCE(ret1,0),0)) OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) gain2,
               AVG(GREATEST(-COALESCE(ret1,0),0)) OVER (PARTITION BY permno ORDER BY date ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) loss2
        FROM base0
      ), f AS (
        SELECT *,
          close/close_lag2-1 AS ret2,
          close/close_lag3-1 AS ret3,
          close/close_lag5-1 AS ret5,
          close/close_lag21-1 AS ret21,
          close/close_lag63-1 AS ret63,
          close/close_lag126-1 AS ret126,
          close/close_lag252-1 AS ret252,
          close_lag21/close_lag63-1 AS momentum_3_1,
          close_lag21/close_lag126-1 AS momentum_6_1,
          close_lag21/close_lag252-1 AS momentum_12_1,
          close/high20-1 AS distance_20d_high,
          close/high252-1 AS distance_52w_high,
          close/sma200-1 AS trend_strength,
          CASE WHEN loss2=0 AND gain2>0 THEN 100.0 WHEN loss2=0 THEN 50.0 ELSE 100.0-(100.0/(1.0+gain2/loss2)) END AS rsi2
        FROM b
      )
      SELECT *,
        CASE
          WHEN sic BETWEEN 100 AND 999 THEN 'Agriculture'
          WHEN sic BETWEEN 1000 AND 1499 THEN 'Mining'
          WHEN sic BETWEEN 1500 AND 1799 THEN 'Construction'
          WHEN sic BETWEEN 2000 AND 3999 THEN 'Manufacturing'
          WHEN sic BETWEEN 4000 AND 4999 THEN 'Transportation/Utilities'
          WHEN sic BETWEEN 5000 AND 5199 THEN 'Wholesale'
          WHEN sic BETWEEN 5200 AND 5999 THEN 'Retail'
          WHEN sic BETWEEN 6000 AND 6799 THEN 'Finance'
          WHEN sic BETWEEN 7000 AND 8999 THEN 'Services'
          ELSE 'Other' END AS sector
      FROM f
      WHERE date BETWEEN DATE {q(str(year_start.date()))} AND DATE {q(str(year_end.date()))}
    ) TO {q(str(staged_file).replace('\\','/'))} (
      FORMAT PARQUET,
      COMPRESSION ZSTD,
      ROW_GROUP_SIZE 250000
    )
    """
    con.execute(sql)
    ok, reason, stats = _validate_parquet_file(con, staged_file, year, BASE_REQUIRED_COLUMNS)
    if not ok:
        raise RuntimeError(f"Newly rebuilt base feature year {year} failed validation: {reason}")

    final_dir = out / f"year={year}"
    _atomic_replace_dir(staged_dir, final_dir)
    log(f"  year {year} PASS: {int(stats[0]):,} rows; {stats[1]} .. {stats[2]}")


def _build_base(con, cfg: dict, norm: Path, out: Path, years: list[int] | None = None) -> None:
    expected = _expected_years(con, cfg, norm)
    targets = expected if years is None else [y for y in years if y in expected]
    if not targets:
        return
    log(f"Building base momentum/pullback features for {len(targets)} year(s): {targets}")
    for year in targets:
        _build_base_year(con, cfg, norm, out, year)
    tmp = out / ".repair_tmp"
    if tmp.exists() and not any(tmp.iterdir()):
        tmp.rmdir()


def _build_eligible_year(con, cfg: dict, out: Path, year: int) -> None:
    universe = cfg["universe"]
    allowed = ",".join(q(x) for x in universe["allowed_primary_exch"])
    st = ",".join(q(x) for x in universe["allowed_security_type"])
    sst = ",".join(q(x) for x in universe["allowed_security_subtype"])
    sht = ",".join(q(x) for x in universe["allowed_share_type"])

    source = out / f"year={year}" / "data_0.parquet"
    eligible_root = out / "eligible"
    stage_root = eligible_root / ".repair_tmp"
    staged_dir = stage_root / f"eligible_year={year}"
    staged_file = staged_dir / "data_0.parquet"
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    log(f"Rebuilding eligible feature year {year} atomically...")
    eq = f"""
    COPY (
      SELECT *,
        (close >= {float(universe['min_price'])}
         AND adv20 >= {float(universe['min_adv20_dollars'])}
         AND history_days >= {int(universe['min_history_days'])}
         AND (primary_exch IS NULL OR primary_exch IN ({allowed}))
         AND (security_type IS NULL OR UPPER(security_type) IN ({st}))
         AND (security_subtype IS NULL OR UPPER(security_subtype) IN ({sst}))
         AND (share_type IS NULL OR UPPER(share_type) IN ({sht}))) AS eligible
      FROM read_parquet({q(str(source).replace('\\','/'))}, union_by_name=true)
    ) TO {q(str(staged_file).replace('\\','/'))} (
      FORMAT PARQUET,
      COMPRESSION ZSTD,
      ROW_GROUP_SIZE 250000
    )
    """
    con.execute(eq)
    ok, reason, stats = _validate_parquet_file(con, staged_file, year, ELIGIBLE_REQUIRED_COLUMNS)
    if not ok:
        raise RuntimeError(f"Newly rebuilt eligible feature year {year} failed validation: {reason}")

    final_dir = eligible_root / f"eligible_year={year}"
    _atomic_replace_dir(staged_dir, final_dir)
    log(f"  eligible year {year} PASS: {int(stats[0]):,} rows")


def _build_eligible(con, cfg: dict, norm: Path, out: Path, years: list[int] | None = None) -> None:
    expected = _expected_years(con, cfg, norm)
    targets = expected if years is None else [y for y in years if y in expected]
    if not targets:
        return
    (out / "eligible").mkdir(parents=True, exist_ok=True)
    log(f"Building eligible feature layer for {len(targets)} year(s): {targets}")
    for year in targets:
        _build_eligible_year(con, cfg, out, year)
    tmp = out / "eligible" / ".repair_tmp"
    if tmp.exists() and not any(tmp.iterdir()):
        tmp.rmdir()


def _write_manifest(con, cfg: dict, norm: Path, out: Path, manifest: Path) -> dict:
    stats = _eligible_stats(con, cfg, norm, out)
    if stats is None:
        raise RuntimeError("Cannot mark feature stage complete: eligible parquet store is missing/incomplete.")
    if not _base_is_current(con, cfg, norm, out):
        raise RuntimeError("Cannot mark feature stage complete: one or more base feature years are invalid.")
    if not _eligible_is_current(con, cfg, norm, out):
        raise RuntimeError("Eligible feature store failed coverage/row-count validation against base features.")
    payload = {
        "stage": "features",
        "release": "Patch 1.5 - Atomic Year Repair",
        "completed_at": datetime.now().isoformat(),
        "min_date": str(stats[0]),
        "max_date": str(stats[1]),
        "rows": int(stats[2]),
        "eligible_rows": int(stats[3] or 0),
        "year_count": int(stats[4]),
        "integrity_validated": True,
        "atomic_year_partitions": True,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(manifest)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Rebuild all base features and eligible years.")
    ap.add_argument(
        "--repair",
        action="store_true",
        help="Validate every year independently and rebuild only missing/corrupt feature partitions.",
    )
    args = ap.parse_args()

    cfg = load_config()
    norm = root_path(cfg, cfg["data"]["normalized_dir"])
    out = root_path(cfg, cfg["data"]["feature_dir"])
    manifests = root_path(cfg, cfg["data"]["manifest_dir"])
    temp = root_path(cfg, cfg["data"]["temp_dir"])
    manifest = manifests / "features_success.json"

    if not list(norm.rglob("*.parquet")):
        raise FileNotFoundError(f"Normalized parquet not found under {norm}. Run 01_build_data.cmd first.")

    out.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)

    import duckdb

    con = duckdb.connect()
    _configure_duckdb(con, cfg, temp)

    s = _status(out, manifest)
    log(
        "Feature-store preflight: "
        f"manifest={s['manifest_exists']}, base_files={s['base_files']}, "
        f"eligible_files={s['eligible_files']}"
    )

    expected_years = _expected_years(con, cfg, norm)

    if args.force:
        log("--force requested: rebuilding all generated feature years atomically.")
        if manifest.exists():
            manifest.unlink()
        base_targets = expected_years
        eligible_targets = expected_years
    else:
        bscan = _scan_base_partitions(con, cfg, norm, out)
        _log_bad_partitions("Base feature preflight", bscan)
        base_targets = [y for y, d in bscan.items() if not d["ok"]]

        if manifest.exists() and not base_targets:
            escan0 = _scan_eligible_partitions(con, cfg, norm, out)
            if all(d["ok"] for d in escan0.values()) and _eligible_is_current(con, cfg, norm, out):
                payload = _write_manifest(con, cfg, norm, out, manifest)
                log(
                    "Feature stage already complete and physically validated: "
                    f"{payload['rows']:,} rows, {payload['eligible_rows']:,} eligible."
                )
                return
            log("Stale features_success.json detected; physical feature store is incomplete/stale. Repairing.")
            manifest.unlink()
        elif manifest.exists() and base_targets:
            log("features_success.json cannot be trusted because base year validation failed; removing marker.")
            manifest.unlink()

        if base_targets:
            log(
                "Repair mode will rebuild ONLY invalid base feature year(s); "
                "normalized CRSP will not be modified."
            )
            _build_base(con, cfg, norm, out, base_targets)
        else:
            log("All base feature years validated; no base recomputation required.")

        # Rescan base after repairs before touching eligible data.
        bscan = _scan_base_partitions(con, cfg, norm, out)
        bad_after = [y for y, d in bscan.items() if not d["ok"]]
        if bad_after:
            _log_bad_partitions("Base feature post-repair validation", bscan)
            raise RuntimeError(f"Base feature repair still has invalid years: {bad_after}")

        escan = _scan_eligible_partitions(con, cfg, norm, out)
        _log_bad_partitions("Eligible feature preflight", escan)
        eligible_targets = [y for y, d in escan.items() if not d["ok"]]

        # If a base year was rebuilt, its eligible derivative must also be refreshed even if an
        # old eligible file happens to be readable.
        eligible_targets = sorted(set(eligible_targets).union(base_targets))

    if args.force:
        _build_base(con, cfg, norm, out, base_targets)

    if eligible_targets:
        _build_eligible(con, cfg, norm, out, eligible_targets)
    else:
        log("All eligible feature years validated; no eligible recomputation required.")

    payload = _write_manifest(con, cfg, norm, out, manifest)
    log(
        f"Feature store complete and validated: {payload['rows']:,} rows; "
        f"eligible rows {payload['eligible_rows']:,}; years {payload['year_count']}."
    )


if __name__ == "__main__":
    main()
