from __future__ import annotations

# Exact active repository surface for the finalized Foundation 2.2 production tree.
# The two transient cleanup files are allowed only so the one-time finalizer can run
# the final production verifier before deleting itself.
ROOT_ALLOWED_FILES = {
    ".gitignore",
    "00_setup_env.cmd",
    "01_build_data.cmd",
    "04_run_all.cmd",
    "05_run_f2_2_stock_screener.cmd",
    "18_recover_production_state.cmd",
    "47_refresh_and_run_alpaca_2026_forward.cmd",
    "80_run_foundation2_2_production.cmd",
    "81_verify_foundation2_2_production.cmd",
    "82_foundation2_2_operational_suite.cmd",
    "open_latest_log.cmd",
    "config.yaml",
    "PRODUCTION_SPEC.json",
    "STOCK_SCREENER_SPEC.json",
    "README.md",
    "requirements.txt",
}

TRANSIENT_CLEANUP_FILES = {
    "100_finalize_f2_2_production_cleanup.cmd",
    "_finalize_f2_2_cleanup.py",
}

SRC_ALLOWED_FILES = {
    "__init__.py",
    "backtest/__init__.py",
    "backtest/engine.py",
    "backtest/run_backtest.py",
    "common/__init__.py",
    "common/config.py",
    "common/logging_utils.py",
    "common/metrics.py",
    "common/run_logged.py",
    "common/schema.py",
    "common/sic.py",
    "data/__init__.py",
    "data/build_benchmarks.py",
    "data/build_normalized.py",
    "data/inspect_source.py",
    "data/verify_data.py",
    "features/__init__.py",
    "features/build_features.py",
    "features/validate_feature_store.py",
    "live/__init__.py",
    "live/alpaca_crsp_bridge.py",
    "live/build_alpaca_2026_normalized.py",
    "live/download_alpaca_2026_ytd.py",
    "live/extend_benchmarks_from_alpaca.py",
    "live/install_alpaca_2026_normalized.py",
    "production/__init__.py",
    "production/analyze_outputs.py",
    "production/canonicalize_outputs.py",
    "production/forward_data_readiness.py",
    "production/forward_prefix_lock.py",
    "production/identity.py",
    "production/production_health.py",
    "production/production_state.py",
    "production/prospective_monitor.py",
    "production/run_cycle.py",
    "production/surface.py",
    "production/verify.py",
    "screener/__init__.py",
    "screener/build_monthly_rankings.py",
    "screener/ranking.py",
    "screener/stock_focus.py",
}

TEST_ALLOWED_FILES = {
    "test_alpaca_2026_downloader.py",
    "test_alpaca_alias_resolution.py",
    "test_alpaca_anchor_safety.py",
    "test_alpaca_forward_wiring.py",
    "test_alpaca_inverse_permno.py",
    "test_alpaca_normalized_pipeline.py",
    "test_engine.py",
    "test_f2_2_production_surface.py",
    "test_feature_atomic_repair.py",
    "test_feature_store_integrity.py",
    "test_feature_store_repair.py",
    "test_metrics.py",
    "test_ranking.py",
    "test_schema.py",
    "test_stock_focus_screener.py",
}

# Manifests retained because the active production/data/live modules create or consume them.
MANIFEST_ALLOWED_FILES = {
    "alpaca_2026_ytd.json",
    "alpaca_benchmark_extension_2026.json",
    "alpaca_crsp_bridge_2026.json",
    "alpaca_normalized_2026.json",
    "alpaca_normalized_2026_install.json",
    "benchmarks_success.json",
    "data_verification.json",
    "features_success.json",
    "feature_store_integrity.json",
    "foundation2_2_main_forward_prefix_lock.json",
    "f2_2_production_cycle.json",
    "normalized_success.json",
    "screener_success.json",
    "stock_screener_success.json",
}

OUTPUT_ALLOWED_TOP_DIRS = {"backtest", "baselines", "diagnostics", "screener", "stock_screener"}
BASELINE_ALLOWED_DIRS = {"foundation2_2_production_v0_2_2"}
DIAGNOSTIC_ALLOWED_DIRS = {"foundation2_2"}
