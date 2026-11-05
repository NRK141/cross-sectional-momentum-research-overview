"""Mark tests requiring deliberately omitted local artifacts."""
from pathlib import Path
import pytest

ROOT = Path(__file__).parent
LAUNCHER_TESTS = {
    "tests/test_alpaca_anchor_safety.py::test_monthly_refresh_invokes_staging_before_install",
    "tests/test_alpaca_forward_wiring.py::test_full_monthly_launcher_orders_pipeline",
    "tests/test_alpaca_normalized_pipeline.py::test_normalizer_has_dedicated_log_and_traceback",
}
SOBOL_TESTS = {
    "research_stochastic/tests/test_mc_v280.py::test_sobol_same_context_is_reproducible",
    "research_stochastic/tests/test_mc_v280.py::test_qe_paths_are_positive_and_finite",
    "research_stochastic/tests/test_mc_v280.py::test_sobol_scramble_is_reproducible_and_balanced",
    "research_stochastic/tests/test_mc_v280.py::test_legacy_forecast_record_preserves_model_fields_needed_by_live_risk_allocator",
}

def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid in LAUNCHER_TESTS and not (ROOT / "47_refresh_and_run_alpaca_2026_forward.cmd").is_file():
            item.add_marker(pytest.mark.skip(reason="private operational launcher omitted from public source"))
        if item.nodeid in SOBOL_TESTS and not (ROOT / "research_stochastic/mc_v280/sobol_joe_kuo_512.npz").is_file():
            item.add_marker(pytest.mark.skip(reason="Sobol direction-number data omitted from public source"))
