from pathlib import Path
import ast
import pytest

from src.live.alpaca_crsp_bridge import BridgeError
from src.live.build_alpaca_2026_normalized import _check_unanchored_budget

NORMALIZER = Path('src/live/build_alpaca_2026_normalized.py')
REFRESH = Path('47_refresh_and_run_alpaca_2026_forward.cmd')


def test_anchor_safety_sources_compile():
    ast.parse(NORMALIZER.read_text(encoding='utf-8'))


def test_single_unanchored_mapping_is_within_small_budget():
    frac = _check_unanchored_budget(9969, 1)
    assert 0 < frac < 0.001


def test_unanchored_count_budget_fails_closed():
    with pytest.raises(BridgeError):
        _check_unanchored_budget(10000, 101)


def test_unanchored_fraction_budget_fails_closed():
    with pytest.raises(BridgeError):
        _check_unanchored_budget(1000, 11)


def test_normalizer_excludes_unanchored_instead_of_fabricating_scale():
    text = NORMALIZER.read_text(encoding='utf-8')
    assert 'alpaca_unanchored_mappings_2026.csv' in text
    assert 'CREATE TEMP VIEW anchor_valid AS' in text
    assert 'JOIN anchor_valid a' in text
    assert 'excluded_unanchored=' in text
    assert 'no mapping was guessed' in text
    # The old unconditional one-row failure is gone.
    assert 'accepted bridge row(s) do not have one valid positive anchor/scale' not in text


def test_monthly_refresh_invokes_staging_before_install():
    text = REFRESH.read_text(encoding='utf-8')
    build = text.index('src.live.build_alpaca_2026_normalized')
    install = text.index('src.live.install_alpaca_2026_normalized')
    assert build < install
