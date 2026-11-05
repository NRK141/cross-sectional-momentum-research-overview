from pathlib import Path
import ast


def test_normalizer_compiles():
    p=Path('src/live/build_alpaca_2026_normalized.py')
    ast.parse(p.read_text(encoding='utf-8'))


def test_ambiguous_using_join_removed():
    text=Path('src/live/build_alpaca_2026_normalized.py').read_text(encoding='utf-8')
    assert 'JOIN anchor a USING(symbol,permno)' not in text
    assert 'ON a.symbol=b.symbol AND a.permno=b.permno' in text
    assert 'ON UPPER(TRIM(CAST(r.symbol AS VARCHAR)))=ab.symbol' in text


def test_duplicate_date_permno_is_fail_closed():
    text=Path('src/live/build_alpaca_2026_normalized.py').read_text(encoding='utf-8')
    assert 'joined_ytd_duplicate_date_permno_groups' in text
    assert 'multiple Alpaca symbols resolve to the same PERMNO on overlapping dates' in text
    assert 'review the bridge rather than deduplicating' in text


def test_normalizer_has_dedicated_log_and_traceback():
    text=Path('src/live/build_alpaca_2026_normalized.py').read_text(encoding='utf-8')
    assert 'alpaca_normalized_2026.log' in text
    assert 'traceback.print_exc()' in text
    launcher=Path('47_refresh_and_run_alpaca_2026_forward.cmd').read_text(encoding='utf-8')
    assert 'src.live.build_alpaca_2026_normalized' in launcher


def test_staging_normalizer_does_not_install_or_modify_history():
    text=Path('src/live/build_alpaca_2026_normalized.py').read_text(encoding='utf-8')
    assert 'normalized_year2026_staging.parquet' in text
    assert "year=2025" in text
    assert "os.replace(tmp,OUT)" in text
    assert "year=2026/data_0.parquet" not in text
