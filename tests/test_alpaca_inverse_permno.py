from pathlib import Path
import ast
import pandas as pd

from src.live.alpaca_crsp_bridge import resolve_inverse_permno_collisions

BRIDGE = Path('src/live/alpaca_crsp_bridge.py')
NORMALIZER = Path('src/live/build_alpaca_2026_normalized.py')


def _boundary():
    return pd.DataFrame([
        {'permno': 10001, 'boundary_ticker': 'NEW', 'boundary_date': '2025-12-31', 'boundary_ticker_count': 1},
        {'permno': 10002, 'boundary_ticker': 'KEEP', 'boundary_date': '2025-12-31', 'boundary_ticker_count': 1},
        {'permno': 10003, 'boundary_ticker': 'MISSING', 'boundary_date': '2025-12-31', 'boundary_ticker_count': 1},
    ])


def test_sources_compile():
    ast.parse(BRIDGE.read_text(encoding='utf-8'))
    ast.parse(NORMALIZER.read_text(encoding='utf-8'))


def test_latest_2025_boundary_ticker_wins_inverse_collision():
    bridge = pd.DataFrame([
        {'symbol':'OLD','permno':10001,'status':'AUTO_EXACT_RECENT','anchor_status':'PASS'},
        {'symbol':'NEW','permno':10001,'status':'AUTO_EXACT_RECENT','anchor_status':'PASS'},
        {'symbol':'SOLO','permno':20000,'status':'AUTO_EXACT_RECENT','anchor_status':'PASS'},
    ])
    resolved, report, stats = resolve_inverse_permno_collisions(bridge, _boundary())
    new = resolved.loc[resolved.symbol.eq('NEW')].iloc[0]
    old = resolved.loc[resolved.symbol.eq('OLD')].iloc[0]
    assert int(new.permno) == 10001
    assert pd.isna(old.permno)
    assert old.status == 'SUPERSEDED_2025_TICKER_ALIAS'
    assert new.inverse_resolution == 'LATEST_2025_CRSP_TICKER_WINS'
    assert stats['permno_collision_groups'] == 1
    assert stats['permno_collision_rows_demoted'] == 1
    assert stats['permno_collision_groups_unresolved'] == 0
    assert set(report.symbol) == {'OLD','NEW'}


def test_single_manual_override_wins_collision():
    bridge = pd.DataFrame([
        {'symbol':'OLD','permno':10002,'status':'AUTO_EXACT_RECENT','anchor_status':'PASS'},
        {'symbol':'MAN','permno':10002,'status':'MANUAL_APPROVED','anchor_status':'PASS'},
    ])
    resolved, _, stats = resolve_inverse_permno_collisions(bridge, _boundary())
    assert int(resolved.loc[resolved.symbol.eq('MAN'),'permno'].iloc[0]) == 10002
    assert pd.isna(resolved.loc[resolved.symbol.eq('OLD'),'permno'].iloc[0])
    assert resolved.loc[resolved.symbol.eq('MAN'),'inverse_resolution'].iloc[0] == 'MANUAL_OVERRIDE_WINS'
    assert stats['permno_collision_groups_unresolved'] == 0


def test_no_boundary_match_fails_closed_by_demoting_entire_group():
    bridge = pd.DataFrame([
        {'symbol':'A','permno':10003,'status':'AUTO_EXACT_RECENT','anchor_status':'PASS'},
        {'symbol':'B','permno':10003,'status':'AUTO_EXACT_RECENT','anchor_status':'PASS'},
    ])
    resolved, report, stats = resolve_inverse_permno_collisions(bridge, _boundary())
    assert resolved.permno.isna().all()
    assert set(resolved.status) == {'AMBIGUOUS_PERMNO_COLLISION'}
    assert stats['permno_collision_groups_unresolved'] == 1
    assert report.accepted.eq(False).all()


def test_multiple_manual_overrides_do_not_choose_arbitrarily():
    bridge = pd.DataFrame([
        {'symbol':'A','permno':10002,'status':'MANUAL_APPROVED','anchor_status':'PASS'},
        {'symbol':'B','permno':10002,'status':'MANUAL_APPROVED','anchor_status':'PASS'},
    ])
    resolved, _, stats = resolve_inverse_permno_collisions(bridge, _boundary())
    assert resolved.permno.isna().all()
    assert stats['permno_collision_groups_unresolved'] == 1


def test_normalizer_canonicalizes_before_raw_join_and_keeps_secondary_guard():
    text = NORMALIZER.read_text(encoding='utf-8')
    assert 'resolve_inverse_permno_collisions(bridge_df, boundary_df)' in text
    assert text.index('resolve_inverse_permno_collisions(bridge_df, boundary_df)') < text.index('CREATE TEMP VIEW ytd_joined')
    assert 'CONFLICTING_OBSERVATION' in text
    assert 'No arbitrary deduplication was performed' in text
    assert 'alpaca_permno_collision_resolution_2026.csv' in text


def test_bridge_future_builds_apply_inverse_uniqueness_before_anchor_download():
    text = BRIDGE.read_text(encoding='utf-8')
    assert 'load_2025_permno_boundary()' in text
    assert 'resolve_inverse_permno_collisions(bridge, boundary)' in text
    assert 'LATEST_2025_CRSP_TICKER_WINS' in text
    assert 'SUPERSEDED_2025_TICKER_ALIAS' in text
    assert 'AMBIGUOUS_PERMNO_COLLISION' in text
