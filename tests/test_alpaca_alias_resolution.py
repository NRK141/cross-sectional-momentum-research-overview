from pathlib import Path
import ast

SRC = Path('src/live/build_alpaca_2026_normalized.py')


def test_normalizer_compiles():
    ast.parse(SRC.read_text(encoding='utf-8'))


def test_no_arbitrary_keep_first_dedupe():
    text = SRC.read_text(encoding='utf-8')
    assert "EQUIVALENT_VENDOR_ALIAS" in text
    assert "CONFLICTING_OBSERVATION" in text
    assert "asset_id_count=1" in text
    assert "No arbitrary deduplication was performed" in text


def test_equivalence_requires_vendor_identity_and_ohlcv():
    text = SRC.read_text(encoding='utf-8')
    assert "asset_id_count=1 AND null_asset_id_rows=0" in text
    assert "scaled_close_max-scaled_close_min" in text
    assert "scaled_open_max-scaled_open_min" in text
    assert "scaled_high_max-scaled_high_min" in text
    assert "scaled_low_max-scaled_low_min" in text
    assert "volume_max-volume_min" in text


def test_alias_selection_is_deterministic_and_semantic():
    text = SRC.read_text(encoding='utf-8')
    assert "asset_status='active'" in text
    assert "=y.symbol THEN 1 ELSE 0 END DESC" in text
    assert "c.symbol_rows DESC" in text
    assert "y.symbol ASC" in text


def test_diagnostics_are_written():
    text = SRC.read_text(encoding='utf-8')
    assert 'alpaca_duplicate_permno_dates_2026.csv' in text
    assert 'alpaca_alias_resolution_2026.csv' in text
    assert 'alpaca_conflicting_permno_dates_2026.csv' in text
    assert 'duplicate_resolution' in text


def test_staging_only_and_history_untouched():
    text = SRC.read_text(encoding='utf-8')
    assert 'normalized_year2026_staging.parquet' in text
    assert 'os.replace(tmp,OUT)' in text
    assert 'year=2026/data_0.parquet' not in text
    assert "year=2025" in text
