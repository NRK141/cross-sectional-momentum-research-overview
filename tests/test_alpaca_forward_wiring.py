from pathlib import Path
import importlib


def test_bridge_column_resolution_case_insensitive(monkeypatch):
    m=importlib.import_module('src.live.build_alpaca_2026_normalized')
    cols=['DlyCalDt','PERMNO','Ticker','DlyPrc','DlyRet','DlyVol']
    assert m._find(cols,m.DATE_CANDS)=='DlyCalDt'
    assert m._find(cols,m.PRICE_CANDS)=='DlyPrc'
    assert m._find(cols,m.RET_CANDS)=='DlyRet'


def test_bridge_policy_is_exact_and_fail_closed():
    text=Path('src/live/alpaca_crsp_bridge.py').read_text(encoding='utf-8')
    assert 'AUTO_EXACT_RECENT' in text
    assert 'AMBIGUOUS_RECENT_TICKER' in text
    assert 'UNMATCHED' in text
    assert 'MANUAL_APPROVED' in text
    assert 'synthetic PERMNO' in text or 'synthetic PERMNOs' in text


def test_normalized_install_only_targets_2026():
    text=Path('src/live/install_alpaca_2026_normalized.py').read_text(encoding='utf-8')
    assert "year=2026" in text
    assert 'historical_hashes_unchanged' in text
    assert 'year <= 2025' in text


def test_full_monthly_launcher_orders_pipeline():
    text=Path('47_refresh_and_run_alpaca_2026_forward.cmd').read_text(encoding='utf-8')
    expected=[
        'src.live.download_alpaca_2026_ytd',
        'src.live.alpaca_crsp_bridge',
        'src.live.build_alpaca_2026_normalized',
        'src.live.install_alpaca_2026_normalized',
        'src.live.extend_benchmarks_from_alpaca',
        'src.features.build_features',
        'src.features.validate_feature_store',
        'src.production.run_cycle',
    ]
    pos=[text.index(x) for x in expected]
    assert pos==sorted(pos)


def test_forward_cycle_does_not_run_wrds_builder_for_alpaca():
    text=Path('src/production/run_cycle.py').read_text(encoding='utf-8')
    assert 'source_kind == "wrds"' in text
    assert 'source_kind == "alpaca"' in text
    assert 'WRDS-only 01_build_data.cmd intentionally skipped' in text
