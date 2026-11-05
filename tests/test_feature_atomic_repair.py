from pathlib import Path

import pytest

from src.features.build_features import BASE_REQUIRED_COLUMNS, _validate_parquet_file


def test_tiny_parquet_is_detected_before_duckdb_open(tmp_path: Path):
    p = tmp_path / "year=2016" / "data_0.parquet"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    ok, reason, _ = _validate_parquet_file(None, p, 2016, BASE_REQUIRED_COLUMNS)
    assert not ok
    assert reason.startswith("too_small")


def test_atomic_repair_symbols_available():
    # Structural regression test that stays runnable in artifact sandboxes without DuckDB.
    from src.features import build_features as bf
    assert callable(bf._scan_base_partitions)
    assert callable(bf._build_base_year)
    assert callable(bf._atomic_replace_dir)
    assert bf.WARMUP_CALENDAR_DAYS >= 500
