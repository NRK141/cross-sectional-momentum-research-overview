from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .identity import ROOT, LIVE_BACKTEST, LIVE_SCREENER, validate_active_identity

OUT = ROOT / "outputs" / "diagnostics" / "foundation2_2"
DATE_CANDIDATES = ("date", "dlycaldt", "caldt", "trade_date", "trading_date")


def _sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _duckdb():
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        raise RuntimeError("DuckDB is required for forward-data readiness checks") from exc
    return duckdb


def _discover_date_column(columns: list[str]) -> str:
    lookup = {c.lower(): c for c in columns}
    for c in DATE_CANDIDATES:
        if c in lookup:
            return lookup[c]
    raise RuntimeError(f"No supported date column found. Columns include: {columns[:30]}")


def _duckdb_relation_stats(source_sql: str) -> dict[str, Any]:
    con = _duckdb().connect(database=":memory:")
    try:
        desc = con.execute(f"DESCRIBE SELECT * FROM {source_sql} LIMIT 0").fetchall()
        cols = [str(r[0]) for r in desc]
        date_col = _discover_date_column(cols)
        ident = '"' + date_col.replace('"', '""') + '"'
        row = con.execute(
            f"SELECT COUNT(*)::BIGINT, MIN(TRY_CAST({ident} AS DATE)), MAX(TRY_CAST({ident} AS DATE)) FROM {source_sql}"
        ).fetchone()
        return {
            "status": "PASS",
            "date_column": date_col,
            "rows": int(row[0]),
            "min_date": row[1].isoformat() if row[1] else None,
            "max_date": row[2].isoformat() if row[2] else None,
        }
    finally:
        con.close()


def raw_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    out = _duckdb_relation_stats(
        f"read_csv_auto({_sql_literal(path.as_posix())}, header=true, all_varchar=true)"
    )
    out["path"] = str(path)
    return out


def parquet_stats(glob_path: Path) -> dict[str, Any]:
    files = [Path(x) for x in glob.glob(str(glob_path), recursive=True)]
    if not files:
        return {"status": "MISSING", "glob": str(glob_path)}
    out = _duckdb_relation_stats(
        f"read_parquet({_sql_literal(glob_path.as_posix())}, union_by_name=true, hive_partitioning=true)"
    )
    out["glob"] = str(glob_path)
    out["files_detected"] = len(files)
    return out


def csv_date_stats(path: Path, column: str = "date") -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    df = pd.read_csv(path, usecols=[column])
    d = pd.to_datetime(df[column], errors="raise")
    return {
        "status": "PASS",
        "path": str(path),
        "rows": int(len(df)),
        "min_date": d.min().date().isoformat() if len(d) else None,
        "max_date": d.max().date().isoformat() if len(d) else None,
    }


def effective_month_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    df = pd.read_csv(path, usecols=["effective_month"])
    d = pd.to_datetime(df["effective_month"].astype(str), format="%Y-%m", errors="raise")
    return {
        "status": "PASS",
        "path": str(path),
        "rows": int(len(df)),
        "min_effective_month": d.min().strftime("%Y-%m") if len(d) else None,
        "max_effective_month": d.max().strftime("%Y-%m") if len(d) else None,
    }


def _date_ge(value: str | None, threshold: pd.Timestamp) -> bool:
    return bool(value and pd.Timestamp(value) >= threshold)


def build_report(target_start: str = "2026-01-01") -> dict[str, Any]:
    identity = validate_active_identity()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8-sig")) or {}
    raw = Path(str(cfg.get("data", {}).get("raw_crsp_csv", "")))
    target = pd.Timestamp(target_start)

    layers: dict[str, Any] = {
        "raw_crsp": raw_stats(raw),
        "raw_alpaca_2026": parquet_stats(ROOT / "data" / "alpaca" / "2026" / "alpaca_us_equities_2026_ytd.parquet"),
        "normalized": parquet_stats(ROOT / "data" / "normalized" / "year=*" / "*.parquet"),
        "features_base": parquet_stats(ROOT / "data" / "features" / "year=*" / "*.parquet"),
        "features_eligible": parquet_stats(ROOT / "data" / "features" / "eligible" / "eligible_year=*" / "*.parquet"),
        "benchmark_SPY": csv_date_stats(ROOT / "data" / "benchmarks" / "spy_daily.csv"),
        "benchmark_QQQ": csv_date_stats(ROOT / "data" / "benchmarks" / "qqq_daily.csv"),
        "benchmark_DIA": csv_date_stats(ROOT / "data" / "benchmarks" / "dia_daily.csv"),
        "production_equity": csv_date_stats(LIVE_BACKTEST / "equity_curve.csv"),
        "production_screener": effective_month_stats(LIVE_SCREENER / "monthly_rankings.csv"),
    }

    wrds_raw_ready = layers["raw_crsp"].get("status") == "PASS" and _date_ge(layers["raw_crsp"].get("max_date"), target)
    alpaca_raw_ready = layers["raw_alpaca_2026"].get("status") == "PASS" and _date_ge(layers["raw_alpaca_2026"].get("max_date"), target)
    raw_ready = wrds_raw_ready or alpaca_raw_ready
    raw_source_kind = "wrds" if wrds_raw_ready else ("alpaca" if alpaca_raw_ready else None)
    build_layers = ("normalized", "features_base", "features_eligible", "benchmark_SPY", "benchmark_QQQ", "benchmark_DIA")
    downstream_ready = all(
        layers[k].get("status") == "PASS" and _date_ge(layers[k].get("max_date"), target)
        for k in build_layers
    )
    prod_ready = layers["production_equity"].get("status") == "PASS" and _date_ge(layers["production_equity"].get("max_date"), target)

    if not identity.get("pass"):
        status = "IDENTITY_FAIL"
    elif not raw_ready:
        status = "RAW_FORWARD_DATA_NOT_READY"
    elif not downstream_ready:
        status = "BUILD_REQUIRED"
    elif not prod_ready:
        status = "PRODUCTION_BACKTEST_REQUIRED"
    else:
        status = "PRODUCTION_OUTPUTS_READY"

    return {
        "status": status,
        "target_start": target.date().isoformat(),
        "production_identity": identity,
        "layers": layers,
        "flags": {
            "raw_forward_available": raw_ready,
            "raw_forward_source_kind": raw_source_kind,
            "wrds_raw_forward_available": wrds_raw_ready,
            "alpaca_raw_forward_available": alpaca_raw_ready,
            "normalized_features_benchmarks_forward_ready": downstream_ready,
            "production_forward_output_ready": prod_ready,
        },
        "guardrails": [
            "Foundation 2.2 is the only production strategy.",
            "This readiness check never mutates raw or normalized data.",
            "A forward refresh may append new observations but may not rewrite evidence already protected by the F2.2 production prefix lock.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect F2.2 production forward-data coverage without modifying the project.")
    ap.add_argument("--target-start", default="2026-01-01")
    ap.add_argument("--require-raw-forward", action="store_true")
    args = ap.parse_args()
    report = build_report(args.target_start)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "forward_data_readiness.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 78)
    print("FOUNDATION 2.2 PRODUCTION FORWARD-DATA READINESS")
    print("=" * 78)
    print(f"Status:       {report['status']}")
    print(f"Target start: {report['target_start']}")
    for name, rec in report["layers"].items():
        mx = rec.get("max_date") or rec.get("max_effective_month") or "-"
        print(f"{name:24s} {rec.get('status','?'):8s} max={mx}")
    print(f"Report: {out}")
    print("=" * 78)
    if args.require_raw_forward and not report["flags"]["raw_forward_available"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
