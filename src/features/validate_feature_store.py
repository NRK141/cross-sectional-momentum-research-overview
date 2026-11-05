from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "DuckDB is required for feature-store validation. Run 00_setup_env.cmd first."
    ) from exc


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _read_expr(paths: list[Path]) -> str:
    if not paths:
        raise ValueError("No parquet files supplied")
    items = ", ".join(_sql_string(p.as_posix()) for p in paths)
    return f"read_parquet([{items}], union_by_name=true, hive_partitioning=false)"


def _resolve_key_columns(columns: list[str]) -> tuple[str, str]:
    by_lower = {c.lower(): c for c in columns}
    date_candidates = (
        "date",
        "dlycaldt",
        "caldt",
        "trade_date",
        "trading_date",
    )
    permno_candidates = ("permno", "kypermno")
    date_col = next((by_lower[c] for c in date_candidates if c in by_lower), None)
    permno_col = next((by_lower[c] for c in permno_candidates if c in by_lower), None)
    if date_col is None or permno_col is None:
        raise RuntimeError(
            "Feature-store schema must contain a date column and PERMNO. "
            f"Columns found: {columns}"
        )
    return date_col, permno_col


def _discover_columns(con: Any, paths: list[Path]) -> list[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {_read_expr(paths[:1])}").fetchall()
    return [str(row[0]) for row in rows]


def _partition_stats(
    con: Any, paths: list[Path], date_col: str, permno_col: str
) -> dict[str, int]:
    expr = _read_expr(paths)
    d = _q(date_col)
    p = _q(permno_col)
    row = con.execute(
        f"""
        WITH src AS (
            SELECT {d} AS k_date, {p} AS k_permno
            FROM {expr}
        ), dup AS (
            SELECT k_date, k_permno, COUNT(*) AS n
            FROM src
            GROUP BY 1, 2
            HAVING COUNT(*) > 1
        )
        SELECT
            (SELECT COUNT(*) FROM src),
            (SELECT COUNT(*) FROM dup),
            COALESCE((SELECT SUM(n - 1) FROM dup), 0),
            (SELECT COUNT(*) FROM src WHERE k_date IS NULL),
            (SELECT COUNT(*) FROM src WHERE k_permno IS NULL)
        """
    ).fetchone()
    return {
        "rows": int(row[0]),
        "duplicate_key_groups": int(row[1]),
        "duplicate_extra_rows": int(row[2]),
        "null_date_rows": int(row[3]),
        "null_permno_rows": int(row[4]),
    }


def _date_bounds(con: Any, paths: list[Path], date_col: str) -> tuple[str | None, str | None]:
    expr = _read_expr(paths)
    d = _q(date_col)
    row = con.execute(f"SELECT MIN({d}), MAX({d}) FROM {expr}").fetchone()
    return (
        None if row[0] is None else str(row[0]),
        None if row[1] is None else str(row[1]),
    )


def _layer_report(
    con: Any,
    layer_name: str,
    files: list[Path],
    partition_dirs: list[Path],
) -> dict[str, Any]:
    if not files:
        return {
            "layer": layer_name,
            "status": "FAIL",
            "failures": [f"No parquet files found for {layer_name} layer"],
            "parquet_files": 0,
            "rows": 0,
            "duplicate_key_groups": 0,
            "duplicate_extra_rows": 0,
            "null_date_rows": 0,
            "null_permno_rows": 0,
            "date_min": None,
            "date_max": None,
            "yearly": [],
        }

    columns = _discover_columns(con, files)
    date_col, permno_col = _resolve_key_columns(columns)
    overall = _partition_stats(con, files, date_col, permno_col)
    date_min, date_max = _date_bounds(con, files, date_col)

    yearly: list[dict[str, Any]] = []
    for year_dir in partition_dirs:
        year_files = sorted(year_dir.glob("*.parquet"))
        if not year_files:
            continue
        stats = _partition_stats(con, year_files, date_col, permno_col)
        yearly.append(
            {
                "partition": year_dir.name,
                "parquet_files": len(year_files),
                **stats,
            }
        )

    failures: list[str] = []
    if overall["duplicate_key_groups"]:
        failures.append(
            f"{overall['duplicate_key_groups']:,} duplicate date/PERMNO key groups remain"
        )
    if overall["null_date_rows"]:
        failures.append(f"{overall['null_date_rows']:,} rows have null date keys")
    if overall["null_permno_rows"]:
        failures.append(f"{overall['null_permno_rows']:,} rows have null PERMNO keys")

    return {
        "layer": layer_name,
        "status": "PASS" if not failures else "FAIL",
        "date_column": date_col,
        "permno_column": permno_col,
        "parquet_files": len(files),
        "date_min": date_min,
        "date_max": date_max,
        **overall,
        "failures": failures,
        "yearly": yearly,
    }


def _write_duplicate_sample(
    con: Any,
    layer_name: str,
    paths: list[Path],
    date_col: str,
    permno_col: str,
    writer: csv.writer,
    limit: int = 250,
) -> None:
    expr = _read_expr(paths)
    d = _q(date_col)
    p = _q(permno_col)
    rows = con.execute(
        f"""
        SELECT {d} AS date_key, {p} AS permno_key, COUNT(*) AS row_count
        FROM {expr}
        GROUP BY 1, 2
        HAVING COUNT(*) > 1
        ORDER BY row_count DESC, date_key, permno_key
        LIMIT {int(limit)}
        """
    ).fetchall()
    for row in rows:
        writer.writerow([layer_name, row[0], row[1], row[2]])


def validate_feature_store(project_root: Path) -> dict[str, Any]:
    feature_root = project_root / "data" / "features"
    manifest_dir = project_root / "data" / "manifests"
    logs_dir = project_root / "logs"

    base_dirs = sorted(p for p in feature_root.glob("year=*") if p.is_dir())
    base_files = sorted(p for d in base_dirs for p in d.glob("*.parquet"))

    eligible_root = feature_root / "eligible"
    eligible_dirs = sorted(
        p for p in eligible_root.glob("eligible_year=*") if p.is_dir()
    )
    eligible_files = sorted(p for d in eligible_dirs for p in d.glob("*.parquet"))

    con = duckdb.connect(database=":memory:")
    try:
        base = _layer_report(con, "base", base_files, base_dirs)
        eligible = _layer_report(con, "eligible", eligible_files, eligible_dirs)
        status = "PASS" if base["status"] == "PASS" and eligible["status"] == "PASS" else "FAIL"

        report: dict[str, Any] = {
            "status": status,
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_root": str(feature_root),
            "base": base,
            "eligible": eligible,
            "failures": [
                *(f"base: {x}" for x in base["failures"]),
                *(f"eligible: {x}" for x in eligible["failures"]),
            ],
        }

        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "feature_store_integrity.json"
        manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        duplicate_sample = logs_dir / "feature_duplicate_keys.csv"
        if base["duplicate_key_groups"] or eligible["duplicate_key_groups"]:
            logs_dir.mkdir(parents=True, exist_ok=True)
            with duplicate_sample.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["layer", "date", "PERMNO", "row_count"])
                if base["duplicate_key_groups"]:
                    _write_duplicate_sample(
                        con,
                        "base",
                        base_files,
                        base["date_column"],
                        base["permno_column"],
                        writer,
                    )
                if eligible["duplicate_key_groups"]:
                    _write_duplicate_sample(
                        con,
                        "eligible",
                        eligible_files,
                        eligible["date_column"],
                        eligible["permno_column"],
                        writer,
                    )
        elif duplicate_sample.exists():
            duplicate_sample.unlink()

        return report
    finally:
        con.close()


def _print_layer(layer: dict[str, Any]) -> None:
    print(
        f"{layer['layer'].upper():<10} status={layer['status']:<4} "
        f"files={layer['parquet_files']:,} rows={layer['rows']:,} "
        f"dup_groups={layer['duplicate_key_groups']:,} "
        f"dup_extra={layer['duplicate_extra_rows']:,} "
        f"range={layer['date_min']} -> {layer['date_max']}"
    )
    for item in layer["yearly"]:
        if item["duplicate_key_groups"] or item["null_date_rows"] or item["null_permno_rows"]:
            print(
                f"  {item['partition']}: rows={item['rows']:,}, "
                f"dup_groups={item['duplicate_key_groups']:,}, "
                f"dup_extra={item['duplicate_extra_rows']:,}"
            )


def _print_report(report: dict[str, Any]) -> None:
    print("=" * 78)
    print("FEATURE STORE INTEGRITY VALIDATION - BASE + ELIGIBLE")
    print("=" * 78)
    print(f"Overall status: {report['status']}")
    _print_layer(report["base"])
    _print_layer(report["eligible"])
    print("=" * 78)
    if report["status"] == "PASS":
        print("Both feature layers are safe for ranking: date/PERMNO keys are unique.")
    else:
        print("Feature store is NOT safe for ranking.")
        for failure in report["failures"]:
            print(f"- {failure}")
        print("Duplicate sample: logs\\feature_duplicate_keys.csv")
    print("Integrity manifest: data\\manifests\\feature_store_integrity.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate both generated feature layers before ranking."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()

    try:
        report = validate_feature_store(args.project_root.resolve())
    except Exception as exc:
        print(f"FEATURE STORE VALIDATION ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    _print_report(report)
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
