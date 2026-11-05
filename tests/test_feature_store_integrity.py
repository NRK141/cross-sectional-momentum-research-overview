from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("duckdb")
import duckdb

from src.features.validate_feature_store import validate_feature_store


def _write(path: Path, rows_sql: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(
            f"COPY ({rows_sql}) TO '{path.as_posix()}' (FORMAT PARQUET)"
        )
    finally:
        con.close()


def test_validator_checks_base_and_eligible_layers(tmp_path: Path) -> None:
    base = tmp_path / "data" / "features" / "year=2020" / "data_0.parquet"
    eligible = (
        tmp_path
        / "data"
        / "features"
        / "eligible"
        / "eligible_year=2020"
        / "data_0.parquet"
    )
    _write(
        base,
        "SELECT * FROM (VALUES (DATE '2020-01-02', 10001), (DATE '2020-01-03', 10001)) t(date, PERMNO)",
    )
    _write(
        eligible,
        "SELECT * FROM (VALUES (DATE '2020-01-02', 10001), (DATE '2020-01-02', 10001)) t(date, PERMNO)",
    )

    report = validate_feature_store(tmp_path)
    assert report["status"] == "FAIL"
    assert report["base"]["status"] == "PASS"
    assert report["eligible"]["status"] == "FAIL"
    assert report["eligible"]["duplicate_key_groups"] == 1
