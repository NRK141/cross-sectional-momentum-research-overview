from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
from itertools import product
from pathlib import Path
import json
import hashlib
import math
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data_access import (
    HistoryStore,
    load_config,
    load_feature_calendar,
    iter_feature_quote_history_chunks,
    load_feature_rows_for_dates,
    load_feature_market_breadth_for_dates,
    load_normalized_calibration_backfill,
    load_spec,
    verify_frozen_production_identity,
)
from reporting import portfolio_metrics, write_json
from run_stochastic_standalone import (
    _calibration_at,
    _drift_prefilter_pass,
    _prefilter_score,
    _terminal_forecast,
    _risk_allocation_score,
    _ranking_score,
    _signal_alpha_frame,
    _alpha_columns_for_model,
    _risk_exit,
    _rebalance_date_sets,
    _last_full_horizon_signal_date,
)
from stochastic_core import (
    Calibration,
    adaptive_entry_forecasts_for_stops,
    regularize_calibration_cross_section,
    sha256_file,
    stable_seed,
)

PATCH = "2.9.3"
SCORE_START = pd.Timestamp("2015-01-01")
SCORE_END = pd.Timestamp("2024-12-31")
FOLDS = [
    (pd.Timestamp("2015-01-01"), pd.Timestamp("2016-12-31"), "2015_2016"),
    (pd.Timestamp("2017-01-01"), pd.Timestamp("2018-12-31"), "2017_2018"),
    (pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31"), "2019_2020"),
    (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"), "2021_2022"),
    (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"), "2023_2024"),
]

ALPHA_MODELS = ("BALANCED", "PULLBACK_HEAVY", "HUMP_PULLBACK")


# Patch 2.9.3 storage/checkpoint contract.  The stochastic engine itself is
# changed price/feature/alpha semantics; prior per-date parts CANNOT be reused.
LEGACY_REUSE_PATCH = "2.9.1"


def _disk_free_gb(path: Path) -> float:
    path = Path(path)
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return float(shutil.disk_usage(probe).free) / float(1024 ** 3)


def _assert_disk_space(path: Path, minimum_free_gb: float, label: str) -> None:
    free_gb = _disk_free_gb(path)
    if free_gb < float(minimum_free_gb):
        raise RuntimeError(
            f"Insufficient free disk for {label}: {free_gb:.2f} GiB free at {Path(path)}; "
            f"Patch {PATCH} requires at least {float(minimum_free_gb):.2f} GiB. "
            "Free space or change STOCHASTIC_RESEARCH_SPEC.json storage.research_output_root before retrying."
        )


def _atomic_to_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write a CSV atomically so disk-full/interruption cannot bless a truncated checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        df.to_csv(tmp, index=index)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _atomic_copy(src: Path, dst: Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _valid_panel_part(path: Path, stop_pcts: list[float]) -> bool:
    if not Path(path).exists() or Path(path).stat().st_size <= 0:
        return False
    required = {
        "date", "permno", "forecast_stop_pct", "alpha_balanced", "alpha_hump_pullback", "alpha_pct_hump_pullback",
        "alpha_trend_heavy", "alpha_pullback_heavy", "risk_volatility",
        "mc_p_stop", "mc_p_stop_se", "mc_converged", "forecast_spec_hash",
    }
    try:
        x = pd.read_csv(path, low_memory=False)
    except Exception:
        return False
    if not required.issubset(x.columns) or x.empty:
        return False
    stops = sorted(round(float(v), 8) for v in pd.to_numeric(x["forecast_stop_pct"], errors="coerce").dropna().unique())
    expected = sorted(round(float(v), 8) for v in stop_pcts)
    if stops != expected:
        return False
    counts = x.groupby("forecast_stop_pct")["permno"].nunique().tolist()
    if not counts or len(set(int(v) for v in counts)) != 1:
        return False
    if x.duplicated(["permno", "forecast_stop_pct"]).any():
        return False
    return True


def _valid_prefilter_part(path: Path, *, minimum_rows: int = 1) -> bool:
    if not Path(path).exists() or Path(path).stat().st_size <= 0:
        return False
    required = {
        "date", "permno", "ticker", "close", "persistent_drift",
        "alpha_prefilter_score", "alpha_balanced", "alpha_trend_heavy",
        "alpha_pullback_heavy", "breadth_above_sma200",
    }
    try:
        x = pd.read_csv(path, low_memory=False)
    except Exception:
        return False
    return required.issubset(x.columns) and len(x) >= int(minimum_rows)


def _legacy_checkpoint_candidates(spec: dict) -> list[Path]:
    storage = spec.get("storage", {})
    raw = list(storage.get("legacy_patch_2_9_1_checkpoint_candidates", []))
    # Also support the original C: relative optimizer location if it was copied rather than moved.
    raw.append(str(ROOT / "outputs" / "research" / "heston_merton_optimizer_2015_2024_patch_2_9_1"))
    out: list[Path] = []
    seen: set[str] = set()
    for value in raw:
        p = Path(str(value))
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _find_legacy_checkpoint_source(spec: dict, out: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    for candidate in _legacy_checkpoint_candidates(spec):
        try:
            if candidate.resolve() == Path(out).resolve():
                continue
        except Exception:
            pass
        panel_dir = candidate / "optimizer_panel_parts"
        pre_dir = candidate / "optimizer_prefilter_parts"
        if not panel_dir.is_dir() and not pre_dir.is_dir():
            continue
        marker = candidate / "OPTIMIZER_STARTED.json"
        if marker.exists():
            try:
                m = json.loads(marker.read_text(encoding="utf-8"))
                if str(m.get("patch")) != LEGACY_REUSE_PATCH:
                    continue
            except Exception:
                continue
        score = len(list(panel_dir.glob("*.csv"))) if panel_dir.is_dir() else 0
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best else None


def _import_legacy_291_checkpoints(out: Path, spec: dict, grid: dict) -> dict:
    # The 2.9.3 corporate-action price basis and signal/stop grid invalidate 2.9.1 parts.
    # Explicitly forbid manual or accidental reuse, even if shape checks pass.
    raise RuntimeError("Patch 2.9.3 prohibits importing 2.9.1/2.9.2 forecasts: economic-price/alpha model changed")
    """Import only proven-complete 2.9.1 date parts into the fresh 2.9.3 D: run.

    A prefilter CSV from the disk-full date is never trusted merely because it exists:
    legacy prefilters are imported only when the same date also has a structurally complete
    four-stop panel part.  The prefilter is compacted to top-N + held-name buffer.
    """
    marker = Path(out) / "IMPORTED_PATCH_2_9_1_CHECKPOINTS.json"
    if marker.exists():
        try:
            return json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            pass
    source = _find_legacy_checkpoint_source(spec, out)
    summary = {
        "patch": PATCH, "source_patch": LEGACY_REUSE_PATCH,
        "source": str(source) if source else None,
        "valid_panel_parts_imported": 0, "compact_prefilter_parts_imported": 0,
        "invalid_or_incomplete_panel_parts_skipped": 0,
        "legacy_prefilter_rows_read": 0, "compact_prefilter_rows_written": 0,
    }
    if source is None:
        # Do not seal a negative discovery result: if the moved D: checkpoint folder
        # is attached/copied later, the next launch should be able to discover it.
        print("[RESUME] No Patch 2.9.1 checkpoint source found; building the Patch 2.9.3 panel from scratch.")
        return summary

    stop_pcts = [float(x) for x in grid["variables"]["daily_close_stop_pct"]]
    panel_cfg = grid["common_forecast_panel"]
    keep_n = int(panel_cfg["prefilter_n"]) + int(panel_cfg.get("prefilter_checkpoint_buffer", 16))
    dst_panels = Path(out) / "optimizer_panel_parts"
    dst_prefilters = Path(out) / "optimizer_prefilter_parts"
    dst_panels.mkdir(parents=True, exist_ok=True)
    dst_prefilters.mkdir(parents=True, exist_ok=True)
    src_panels = source / "optimizer_panel_parts"
    src_prefilters = source / "optimizer_prefilter_parts"
    if src_panels.is_dir():
        for src_part in sorted(src_panels.glob("*.csv")):
            if not _valid_panel_part(src_part, stop_pcts):
                summary["invalid_or_incomplete_panel_parts_skipped"] += 1
                continue
            src_pre = src_prefilters / src_part.name
            pre = None
            if _valid_prefilter_part(src_pre):
                try:
                    pre = pd.read_csv(src_pre, parse_dates=["date"], low_memory=False)
                except Exception:
                    pre = None
            # Strengthen legacy completeness validation: a row-boundary-truncated CSV
            # could still parse and contain all four stops.  Require the panel to have
            # exactly the number of names the old top-N stage should have produced.
            try:
                panel_probe = pd.read_csv(src_part, usecols=["permno"], low_memory=False)
                actual_names = int(panel_probe["permno"].nunique())
            except Exception:
                summary["invalid_or_incomplete_panel_parts_skipped"] += 1
                continue
            expected_names = min(int(panel_cfg["prefilter_n"]), int(len(pre))) if pre is not None else int(panel_cfg["prefilter_n"])
            if actual_names != expected_names:
                summary["invalid_or_incomplete_panel_parts_skipped"] += 1
                continue
            dst_part = dst_panels / src_part.name
            if not _valid_panel_part(dst_part, stop_pcts):
                _atomic_copy(src_part, dst_part)
            summary["valid_panel_parts_imported"] += 1
            if pre is None:
                continue
            summary["legacy_prefilter_rows_read"] += int(len(pre))
            if len(pre):
                pre = pre.sort_values(["alpha_prefilter_score", "permno"], ascending=[False, False]).head(keep_n).reset_index(drop=True)
            dst_pre = dst_prefilters / src_pre.name
            _atomic_to_csv(pre, dst_pre, index=False)
            summary["compact_prefilter_parts_imported"] += 1
            summary["compact_prefilter_rows_written"] += int(len(pre))
    write_json(marker, summary)
    print(
        f"[RESUME] Imported {summary['valid_panel_parts_imported']} valid Patch 2.9.1 panel dates from {source}. "
        f"Compacted {summary['compact_prefilter_parts_imported']} prefilter checkpoints to <= {keep_n} rows/date; "
        f"skipped {summary['invalid_or_incomplete_panel_parts_skipped']} incomplete panel part(s)."
    )
    return summary



@dataclass
class ScreenPosition:
    permno: int
    ticker: str
    signal_source: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    entry_notional: float
    entry_cost: float
    held_days: int = 1
    last_seen_price: float | None = None
    last_seen_date: pd.Timestamp | None = None
    missing_sessions: int = 0
    negative_drift_rechecks: int = 0


def _load_grid() -> dict:
    path = HERE / "DEVELOPMENT_OPTIMIZER_GRID.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_data(spec: dict, cfg: dict):
    """Build optimizer state without ever materializing the full feature table.

    Memory footprint is bounded by one yearly four-column quote chunk plus the
    compact per-security HistoryStore. Signal-date feature rows are loaded later
    only for explicit month/week scan dates.
    """
    window = spec["experiment_windows"]["development"]
    calendar = load_feature_calendar(cfg, window["history_start"], window["score_end"])
    if len(calendar) == 0:
        raise RuntimeError("No feature-store trading dates loaded for optimizer development window")
    if pd.Timestamp(calendar.max()) > SCORE_END:
        raise RuntimeError("OPTIMIZER FIREWALL BREACH: post-2024 feature date loaded")

    feature_min = pd.Timestamp(calendar.min())
    required = int(spec["model"]["max_calibration_observations"])
    feature_pre = int((calendar < SCORE_START).sum())
    missing = max(0, required - feature_pre)
    backfill = pd.DataFrame(columns=["date", "permno", "ret1", "close"])
    if missing:
        backfill = load_normalized_calibration_backfill(
            cfg, before_date=feature_min, global_sessions_needed=missing
        )

    def frames():
        if len(backfill):
            yield backfill
        yield from iter_feature_quote_history_chunks(
            cfg, feature_min, pd.Timestamp(window["score_end"])
        )

    print("[MEMORY] Building compact HistoryStore from calibration backfill + yearly 4-column feature chunks...")
    history = HistoryStore.from_frames(frames())
    backfill_sessions = int(pd.to_datetime(backfill["date"]).nunique()) if len(backfill) else 0
    warm = feature_pre + backfill_sessions
    if warm < required:
        raise RuntimeError(f"Optimizer calibration warm-up only {warm} sessions; need {required}")
    score_dates = pd.DatetimeIndex(calendar[(calendar >= SCORE_START) & (calendar <= SCORE_END)])
    if len(score_dates) == 0:
        raise RuntimeError("No scored trading dates in 2015-2024")
    return score_dates, history, {
        "feature_min_loaded": str(feature_min.date()),
        "feature_max_loaded": str(pd.Timestamp(calendar.max()).date()),
        "feature_pre2015_sessions": feature_pre,
        "normalized_backfill_sessions": backfill_sessions,
        "combined_pre2015_sessions": warm,
        "calibration_min_loaded": str(pd.Timestamp(backfill["date"].min()).date()) if len(backfill) else str(feature_min.date()),
        "history_store_permnos": int(len(history.dates)),
        "data_mode": "MEMORY_BOUNDED_RETURN_CONSISTENT_HISTORY_PLUS_SIGNAL_DATE_FEATURE_ROWS",
    }


def _exact_price(history: HistoryStore, permno: int, date: pd.Timestamp) -> float | None:
    p = int(permno)
    if p not in history.dates:
        return None
    arr = history.dates[p]
    d = np.datetime64(pd.Timestamp(date).to_datetime64())
    i = int(np.searchsorted(arr, d, side="left"))
    if i >= len(arr) or arr[i] != d:
        return None
    x = float(history.closes[p][i])
    return x if math.isfinite(x) and x > 0 else None


def _current_or_last_price(history: HistoryStore, permno: int, date: pd.Timestamp, fallback: float) -> float:
    p = int(permno)
    if p not in history.dates:
        return float(fallback)
    arr = history.dates[p]
    d = np.datetime64(pd.Timestamp(date).to_datetime64())
    i = int(np.searchsorted(arr, d, side="right")) - 1
    if i < 0:
        return float(fallback)
    x = float(history.closes[p][i])
    return x if math.isfinite(x) and x > 0 else float(fallback)


def _stop_key(x: float) -> float:
    return round(float(x), 6)


def _panel_spec_hash(spec: dict, grid: dict) -> str:
    payload = {
        "patch": PATCH,
        "model": spec.get("model", {}),
        "standalone_mc_quality": {
            "require_mc_converged": spec.get("standalone", {}).get("require_mc_converged"),
            "max_mc_mean_se": spec.get("standalone", {}).get("max_mc_mean_se"),
        },
        "panel": grid.get("common_forecast_panel", {}),
        "stops": grid.get("variables", {}).get("daily_close_stop_pct", []),
        "execution": "signal_t0_next_close_t1_close_stop_from_t2_actual_close_fill",
        "regularization": {
            "method": "same_date_empirical_bayes_drift_plus_jump_reliability_v2",
            "max_mu_reliability_weight": spec.get("model", {}).get("drift_max_eb_weight", 0.35),
        },
        "alpha": spec.get("alpha_model", spec.get("standalone", {}).get("alpha_research", {})),
        "regime": "all-universe breadth_above_sma200 signal-close only",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _regularized_calibration_from_record(raw: Calibration, rec: dict) -> Calibration:
    """Rehydrate same-date regularized parameters from a persisted prefilter part."""
    def val(name: str, fallback: float) -> float:
        try:
            x = float(rec.get(name, fallback))
            return x if math.isfinite(x) else fallback
        except Exception:
            return fallback
    return replace(
        raw,
        mu=val("persistent_drift", raw.mu),
        lambda_jump=val("lambda_jump_reg", raw.lambda_jump),
        jump_mu_log=val("jump_mu_log_reg", raw.jump_mu_log),
        jump_sigma_log=val("jump_sigma_log_reg", raw.jump_sigma_log),
        mu_raw=val("mu_raw", raw.mu_raw if math.isfinite(raw.mu_raw) else raw.mu),
        mu_se=val("mu_se", raw.mu_se),
        mu_shrink_weight=val("mu_shrink_weight", raw.mu_shrink_weight),
        jump_shrink_weight=val("jump_shrink_weight", raw.jump_shrink_weight),
        mu_prior=val("mu_prior", raw.mu_prior),
    )


def _build_common_panel(
    all_dates: pd.DatetimeIndex,
    history: HistoryStore,
    spec: dict,
    grid: dict,
    out: Path,
    cfg: dict,
) -> pd.DataFrame:
    """Build a stop-aware screening forecast cube from signal-date feature rows.

    Patch 2.9.3 simulates one RQMC path set per security/date and evaluates every
    optimizer close-stop level from those same paths.  Drift sign is no longer a
    binary prefilter; same-date drift/jump estimates are regularized before the
    top-N soft prefilter is formed.
    """
    panel_path = out / "optimizer_forecast_panel.csv"
    meta_path = out / "optimizer_forecast_panel_meta.json"
    parts = out / "optimizer_panel_parts"
    pre_parts = out / "optimizer_prefilter_parts"
    panel_hash = _panel_spec_hash(spec, grid)
    storage_cfg = spec.get("storage", {})
    _assert_disk_space(out, float(storage_cfg.get("minimum_free_gb_checkpoint", 5.0)), "optimizer checkpointing")
    # Patch 2.9.3 requires a clean forecast cube: return-adjusted economic prices,
    # recomputed technical features and an expanded stop/alpha grid.
    # Corporate-action/economic-price and new-alpha semantics invalidate ALL 2.9.x forecasts.
    # Never import previous panel/pre-filter parts even when their shapes match.
    if panel_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("patch") == PATCH
            and meta.get("score_end") == "2024-12-31"
            and meta.get("panel_spec_hash") == panel_hash
        ):
            print(f"[PANEL] Reusing completed Patch {PATCH} forecast cube: {panel_path}")
            return pd.read_csv(panel_path, parse_dates=["date"])

    parts.mkdir(parents=True, exist_ok=True)
    pre_parts.mkdir(parents=True, exist_ok=True)
    month_end, week_end = _rebalance_date_sets(all_dates)
    # Patch 2.9.3 fixes entry discovery to month-end after the 2.8.3 search showed
    # every top-50 screen configuration converged on MONTH_END_ONLY.  Do not pay
    # Monte Carlo cost for weekly dates that no optimizer combination can use.
    scan_dates = sorted(month_end)
    last_signal = _last_full_horizon_signal_date(all_dates, int(spec["standalone"]["max_holding_days"]))
    scan_dates = [pd.Timestamp(d) for d in scan_dates if last_signal is not None and pd.Timestamp(d) <= last_signal]
    panel_cfg = grid["common_forecast_panel"]
    prefilter_n = int(panel_cfg["prefilter_n"])
    prefilter_buffer = int(panel_cfg.get("prefilter_checkpoint_buffer", spec.get("storage", {}).get("prefilter_checkpoint_buffer", 16)))
    prefilter_keep_n = prefilter_n + max(prefilter_buffer, int(max(grid["variables"].get("max_active_positions", [5]))))
    n_paths = int(panel_cfg["n_paths"])
    horizon = int(panel_cfg.get("forecast_horizon_days", spec["standalone"]["forecast_horizon_days"]))
    stop_pcts = [float(x) for x in grid["variables"]["daily_close_stop_pct"]]

    cal_cache: dict = {}
    signal_feature_anomalies: list[dict] = []
    done = 0
    dates_by_year: dict[int, list[pd.Timestamp]] = {}
    for d in scan_dates:
        dates_by_year.setdefault(d.year, []).append(d)

    for year in sorted(dates_by_year):
        batch_dates = dates_by_year[year]
        need_feature = [
            d for d in batch_dates
            if not _valid_prefilter_part(pre_parts / f"{d.strftime('%Y%m%d')}.csv")
        ]
        feature = load_feature_rows_for_dates(cfg, need_feature) if need_feature else pd.DataFrame()
        breadth_df = history.economic_breadth(need_feature) if need_feature else pd.DataFrame()
        groups = {pd.Timestamp(k): g.copy() for k, g in feature.groupby("date", sort=False)} if len(feature) else {}
        breadth_map = {pd.Timestamp(r["date"]): r for r in breadth_df.to_dict("records")} if len(breadth_df) else {}
        for date in batch_dates:
            pre_path = pre_parts / f"{date.strftime('%Y%m%d')}.csv"
            part = parts / f"{date.strftime('%Y%m%d')}.csv"
            if not _valid_prefilter_part(pre_path):
                gg = groups.get(date, pd.DataFrame()).copy()
                # Price-based alpha features are recomputed from return-consistent
                # daily history to eliminate split distortions. Ineligible upstream
                # rows remain ineligible; eligibility provenance is audited separately.
                if len(gg):
                    repairs = [history.repaired_signal_features(int(perm), date) for perm in gg["permno"]]
                    # Preserve an audit of extreme feature-store discrepancies.
                    # We cannot certify the UPSTREAM `eligible` calculation without
                    # upstream feature-generation source; do not conceal that limit.
                    for src_row, repaired in zip(gg.to_dict("records"), repairs):
                        if repaired is None: continue
                        def diff(a, b):
                            try: return abs(float(a)-float(b))
                            except (ValueError,TypeError):return float('nan')
                        source_trend = (float(src_row["close"])/float(src_row["sma200"]) - 1.0) if pd.notna(src_row.get("sma200")) and float(src_row["sma200"])>0 else float('nan')
                        corrected_trend = repaired["close"]/repaired["sma200"]-1.0
                        ret5_gap = diff(src_row.get("ret5"), repaired["ret5"])
                        trend_gap = abs(source_trend-corrected_trend)
                        if ((np.isfinite(ret5_gap) and ret5_gap>0.15) or
                            (np.isfinite(trend_gap) and trend_gap>0.15)):
                            signal_feature_anomalies.append({"date":str(date.date()),"permno":int(src_row["permno"]),
                                    "ticker":str(src_row.get("ticker","")),"source_ret5":src_row.get("ret5"),
                                    "repaired_ret5":repaired["ret5"],"source_close_sma200_gap":source_trend,
                                    "repaired_close_sma200_gap":corrected_trend})
                    for key in ("close", "sma20", "sma50", "sma100", "sma200", "ret5", "rsi2", "distance_20d_high"):
                        gg[key] = [r[key] if r is not None else np.nan for r in repairs]
                    gg = gg.dropna(subset=["close", "sma200", "ret5", "rsi2"])
                alpha_frame = _signal_alpha_frame(gg.set_index("permno", drop=False)) if len(gg) else pd.DataFrame()
                breadth_rec = breadth_map.get(date, {})
                breadth200 = float(breadth_rec.get("breadth_above_sma200", np.nan))
                breadth100 = float(breadth_rec.get("breadth_above_sma100", np.nan))
                breadth_ret5 = float(breadth_rec.get("breadth_ret5_positive", np.nan))
                raw_rows: list[tuple[dict, Calibration]] = []
                raw_map: dict[int, Calibration] = {}
                if len(gg):
                    for rec in gg.to_dict("records"):
                        perm = int(rec["permno"])
                        px = float(rec["close"])
                        if not math.isfinite(px) or px <= 0:
                            continue
                        cal = _calibration_at(perm, date, history, spec, cal_cache)
                        if cal is None:
                            continue
                        raw_rows.append((rec, cal))
                        raw_map[perm] = cal
                reg_map, reg_diag = regularize_calibration_cross_section(
                    raw_map,
                    jump_prior_strength=float(spec.get("model", {}).get("jump_prior_strength", 8.0)),
                    max_mu_reliability_weight=float(spec.get("model", {}).get("drift_max_eb_weight", 0.35)),
                )
                rows = []
                for rec, raw in raw_rows:
                    perm = int(rec["permno"])
                    cal = reg_map.get(perm, raw)
                    cal_cache[(perm, pd.Timestamp(date))] = cal
                    rows.append({
                        "date": date,
                        "permno": perm,
                        "ticker": str(rec.get("ticker", "")),
                        "sector": str(rec.get("sector", "Other")),
                        "close": float(rec["close"]),
                        "persistent_drift": float(_prefilter_score(cal, spec)),
                        "mu_raw": float(cal.mu_raw if math.isfinite(cal.mu_raw) else raw.mu),
                        "mu_se": float(cal.mu_se),
                        "mu_shrink_weight": float(cal.mu_shrink_weight),
                        "mu_prior": float(cal.mu_prior),
                        "lambda_jump_reg": float(cal.lambda_jump),
                        "jump_mu_log_reg": float(cal.jump_mu_log),
                        "jump_sigma_log_reg": float(cal.jump_sigma_log),
                        "jump_shrink_weight": float(cal.jump_shrink_weight),
                        "regularization_n": int(reg_diag.get("n", 0)),
                        "breadth_above_sma200": breadth200,
                        "breadth_above_sma100": breadth100,
                        "breadth_ret5_positive": breadth_ret5,
                        **(alpha_frame.loc[perm].to_dict() if perm in alpha_frame.index else {}),
                    })
                pre = pd.DataFrame(rows)
                if len(pre):
                    pre = pre.sort_values(["alpha_prefilter_score", "permno"], ascending=[False, False]).reset_index(drop=True)
                # Persist only what exact replay can consume: top-N plus enough buffer
                # to replace every simultaneously held name.  The old full-universe
                # per-date CSVs were the primary disk-growth defect.
                pre = pre.head(prefilter_keep_n).reset_index(drop=True) if len(pre) else pre
                _assert_disk_space(out, float(storage_cfg.get("minimum_free_gb_checkpoint", 5.0)), "prefilter checkpoint")
                _atomic_to_csv(pre, pre_path, index=False)
            else:
                pre = pd.read_csv(pre_path, parse_dates=["date"], low_memory=False)

            if not _valid_panel_part(part, stop_pcts):
                pre_top = pre.head(prefilter_n) if len(pre) else pre
                rows = []
                for rec in pre_top.to_dict("records"):
                    perm = int(rec["permno"])
                    raw = _calibration_at(perm, date, history, spec, cal_cache)
                    if raw is None:
                        continue
                    cal = _regularized_calibration_from_record(raw, rec)
                    cal_cache[(perm, pd.Timestamp(date))] = cal
                    context_key = f"optimizer_panel_v290|{pd.Timestamp(date).date()}|h{horizon}"
                    forecasts = adaptive_entry_forecasts_for_stops(
                        float(rec["close"]), cal, spec, horizon_days=horizon,
                        requested_paths=n_paths, stop_pcts=stop_pcts,
                        purpose="optimizer_panel_v290", context_key=context_key,
                    )
                    for stop in stop_pcts:
                        f = forecasts[_stop_key(stop)] if _stop_key(stop) in forecasts else forecasts[float(stop)]
                        risk_score, risk_vol = _risk_allocation_score(f, spec)
                        stop_ucb = float(f.get("mc_p_stop", 1.0)) + float(spec["standalone"].get("mc_stop_risk_z", 1.0)) * float(f.get("mc_p_stop_se", 0.0))
                        rows.append({
                            "date": date,
                            "permno": perm,
                            "ticker": str(rec["ticker"]),
                            "sector": str(rec.get("sector", "Other")),
                            "is_month_end": bool(date in month_end),
                            "is_week_end": bool(date in week_end),
                            "forecast_stop_pct": float(stop),
                            "persistent_drift": float(rec["persistent_drift"]),
                            "alpha_balanced": float(rec.get("alpha_balanced", 0.5)),
                            "alpha_trend_heavy": float(rec.get("alpha_trend_heavy", 0.5)),
                            "alpha_pullback_heavy": float(rec.get("alpha_pullback_heavy", 0.5)),
                            "alpha_hump_pullback": float(rec.get("alpha_hump_pullback", 0.5)),
                            "alpha_pct_balanced": float(rec.get("alpha_pct_balanced", 0.5)),
                            "alpha_pct_trend_heavy": float(rec.get("alpha_pct_trend_heavy", 0.5)),
                            "alpha_pct_pullback_heavy": float(rec.get("alpha_pct_pullback_heavy", 0.5)),
                            "alpha_pct_hump_pullback": float(rec.get("alpha_pct_hump_pullback", 0.5)),
                            "alpha_prefilter_score": float(rec.get("alpha_prefilter_score", 0.5)),
                            "breadth_above_sma200": float(rec.get("breadth_above_sma200", np.nan)),
                            "breadth_above_sma100": float(rec.get("breadth_above_sma100", np.nan)),
                            "breadth_ret5_positive": float(rec.get("breadth_ret5_positive", np.nan)),
                            "risk_allocation_score": float(risk_score),
                            "risk_volatility": float(risk_vol),
                            "mc_stop_probability_ucb": float(stop_ucb),
                            **cal.to_dict(),
                            **f,
                        })
                _assert_disk_space(out, float(storage_cfg.get("minimum_free_gb_checkpoint", 5.0)), "forecast-panel checkpoint")
                _atomic_to_csv(pd.DataFrame(rows), part, index=False)
            done += 1
            if done % int(panel_cfg.get("checkpoint_every_scan_dates", 20)) == 0:
                print(f"[PANEL] {done}/{len(scan_dates)} scan dates checkpointed; cal_cache={len(cal_cache):,}")
        del feature, groups, breadth_df, breadth_map

    _atomic_to_csv(pd.DataFrame(signal_feature_anomalies, columns=["date","permno","ticker","source_ret5","repaired_ret5","source_close_sma200_gap","repaired_close_sma200_gap"]),
                   out / "signal_feature_source_discrepancies.csv", index=False)
    print(f"[FEATURE-AUDIT] {len(signal_feature_anomalies)} eligible source-feature discrepancies >15 percentage points. Upstream eligibility provenance remains unverified.")
    frames = []
    for d in scan_dates:
        part = parts / f"{d.strftime('%Y%m%d')}.csv"
        if not _valid_panel_part(part, stop_pcts):
            raise RuntimeError(f"Missing or incomplete optimizer panel part: {part}")
        x = pd.read_csv(part, parse_dates=["date"], low_memory=False)
        frames.append(x)
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _assert_disk_space(out, float(storage_cfg.get("minimum_free_gb_checkpoint", 5.0)), "completed forecast panel")
    _atomic_to_csv(panel, panel_path, index=False)
    hashes = sorted(set(panel.get("forecast_spec_hash", pd.Series(dtype=str)).dropna().astype(str))) if len(panel) else []
    write_json(meta_path, {
        "patch": PATCH,
        "score_start": "2015-01-01",
        "score_end": "2024-12-31",
        "scan_dates": len(scan_dates),
        "rows": len(panel),
        "n_paths_requested": n_paths,
        "prefilter_n": prefilter_n,
        "stop_pcts": stop_pcts,
        "panel_spec_hash": panel_hash,
        "forecast_spec_hashes": hashes,
        "post_2024_rows": int((panel["date"] > SCORE_END).sum()) if len(panel) else 0,
        "memory_mode": "RET_CONSISTENT_ECONOMIC_PRICE_SIGNAL_FEATURES_BREADTH_STOP_AWARE_RQMC_CUBE",
        "prefilter_parts_dir": str(pre_parts),
        "prefilter_checkpoint_rows_max": int(prefilter_keep_n),
        "storage_output_dir": str(out),
        "atomic_checkpoint_writes": True,
        "legacy_2_9_1_checkpoint_import": True,
        "drift_prefilter": "ALPHA_PREFILTER_HM_RISK_ONLY_DRIFT_RELIABILITY_CAPPED",
    })
    print(f"[PANEL] Completed {len(panel):,} stop-aware candidate forecasts across {len(scan_dates)} scan dates.")
    return panel


class _PrefilterStore:
    def __init__(self, parts_dir: Path, max_cache: int = 64):
        self.parts_dir = Path(parts_dir)
        self.max_cache = int(max_cache)
        self.cache: dict[pd.Timestamp, pd.DataFrame] = {}
        self.order: list[pd.Timestamp] = []

    def get(self, date: pd.Timestamp) -> pd.DataFrame:
        date = pd.Timestamp(date)
        if date in self.cache:
            return self.cache[date]
        path = self.parts_dir / f"{date.strftime('%Y%m%d')}.csv"
        if not path.exists():
            return pd.DataFrame(columns=["date", "permno", "ticker", "sector", "close", "persistent_drift", "mu_raw", "mu_se", "mu_shrink_weight", "mu_prior", "lambda_jump_reg", "jump_mu_log_reg", "jump_sigma_log_reg", "jump_shrink_weight"])
        x = pd.read_csv(path, parse_dates=["date"])
        self.cache[date] = x
        self.order.append(date)
        while len(self.order) > self.max_cache:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return x


def _panel_structures(panel: pd.DataFrame) -> dict[tuple[pd.Timestamp, float], dict]:
    out = {}
    if panel.empty:
        return out
    for (dt, stop), g in panel.groupby(["date", "forecast_stop_pct"], sort=True):
        g = g.reset_index(drop=True)
        item = {
            "permno": g["permno"].astype(int).to_numpy(),
            "ticker": g["ticker"].astype(str).to_numpy(dtype=object),
            "sector": g["sector"].astype(str).to_numpy(dtype=object),
            "alpha_balanced": pd.to_numeric(g["alpha_balanced"], errors="coerce").to_numpy(float),
            "alpha_trend_heavy": pd.to_numeric(g["alpha_trend_heavy"], errors="coerce").to_numpy(float),
            "alpha_pullback_heavy": pd.to_numeric(g["alpha_pullback_heavy"], errors="coerce").to_numpy(float),
            "alpha_hump_pullback": pd.to_numeric(g["alpha_hump_pullback"], errors="coerce").to_numpy(float),
            "alpha_pct_balanced": pd.to_numeric(g["alpha_pct_balanced"], errors="coerce").to_numpy(float),
            "alpha_pct_trend_heavy": pd.to_numeric(g["alpha_pct_trend_heavy"], errors="coerce").to_numpy(float),
            "alpha_pct_pullback_heavy": pd.to_numeric(g["alpha_pct_pullback_heavy"], errors="coerce").to_numpy(float),
            "alpha_pct_hump_pullback": pd.to_numeric(g["alpha_pct_hump_pullback"], errors="coerce").to_numpy(float),
            "mc_p_stop": pd.to_numeric(g["mc_p_stop"], errors="coerce").to_numpy(float),
            "mc_p_stop_se": pd.to_numeric(g.get("mc_p_stop_se", pd.Series([0.0] * len(g))), errors="coerce").to_numpy(float),
            "mc_stop_probability_ucb": pd.to_numeric(g["mc_stop_probability_ucb"], errors="coerce").to_numpy(float),
            "risk_volatility": pd.to_numeric(g["risk_volatility"], errors="coerce").to_numpy(float),
            "mc_mean_se": pd.to_numeric(g.get("mc_mean_se", pd.Series([np.nan] * len(g))), errors="coerce").to_numpy(float),
            "mc_converged": g.get("mc_converged", pd.Series([True] * len(g))).astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(bool),
            "breadth_above_sma200": float(pd.to_numeric(g["breadth_above_sma200"], errors="coerce").iloc[0]),
            "is_month_end": str(g["is_month_end"].iloc[0]).strip().lower() in {"true", "1", "yes"},
        }
        item["orders"] = {
            "BALANCED": np.argsort(-np.nan_to_num(item["alpha_balanced"], nan=-np.inf), kind="stable"),
            "TREND_HEAVY": np.argsort(-np.nan_to_num(item["alpha_trend_heavy"], nan=-np.inf), kind="stable"),
            "PULLBACK_HEAVY": np.argsort(-np.nan_to_num(item["alpha_pullback_heavy"], nan=-np.inf), kind="stable"),
            "HUMP_PULLBACK": np.argsort(-np.nan_to_num(item["alpha_hump_pullback"], nan=-np.inf), kind="stable"),
        }
        out[(pd.Timestamp(dt), _stop_key(float(stop)))] = item
    return out


def _dollar_pf(trades: pd.DataFrame) -> float | None:
    if trades.empty or "net_pnl_usd" not in trades:
        return None
    x = pd.to_numeric(trades["net_pnl_usd"], errors="coerce").dropna()
    gp = float(x[x > 0].sum())
    gl = float(-x[x < 0].sum())
    if gl <= 0:
        return None if gp <= 0 else float("inf")
    return float(gp / gl)


def _trade_return_pf(trades: pd.DataFrame) -> float | None:
    if trades.empty:
        return None
    x = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    gp = float(x[x > 0].sum())
    gl = float(-x[x < 0].sum())
    return float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else None)


def _fold_metrics(eq: pd.DataFrame, trades: pd.DataFrame) -> tuple[dict, list[dict]]:
    fold_rows = []
    for a, b, label in FOLDS:
        e = eq[(eq["date"] >= a) & (eq["date"] <= b)].copy()
        t = trades[(trades["exit_date"] >= a) & (trades["exit_date"] <= b)].copy() if len(trades) else trades
        if len(e) >= 2:
            pm = portfolio_metrics(e["equity"])
        else:
            pm = {"cagr": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "total_return": np.nan}
        fold_rows.append({
            "fold": label,
            "fold_start": str(a.date()),
            "fold_end": str(b.date()),
            "total_return": pm.get("total_return"),
            "cagr": pm.get("cagr"),
            "sharpe": pm.get("sharpe"),
            "max_drawdown": pm.get("max_drawdown"),
            "closed_trades": int(len(t)),
            "dollar_profit_factor": _dollar_pf(t),
        })
    f = pd.DataFrame(fold_rows)
    valid_sh = pd.to_numeric(f["sharpe"], errors="coerce").dropna()
    valid_ret = pd.to_numeric(f["total_return"], errors="coerce").dropna()
    agg = {
        "median_fold_sharpe": float(valid_sh.median()) if len(valid_sh) else np.nan,
        "worst_fold_return": float(valid_ret.min()) if len(valid_ret) else np.nan,
        "positive_fold_fraction": float((valid_ret > 0).mean()) if len(valid_ret) else np.nan,
        "folds_with_trades": int((pd.to_numeric(f["closed_trades"], errors="coerce") > 0).sum()),
    }
    return agg, fold_rows


def _screen_replay(combo: dict, all_dates: pd.DatetimeIndex, panels: dict, history: HistoryStore, cost: float, last_signal: pd.Timestamp | None):
    initial = 100000.0
    cash = initial
    positions: dict[int, ScreenPosition] = {}
    pending: list[dict] = []
    trades = []
    equity = []
    max_pos = int(combo["max_active_positions"])
    target_weight = 1.0 / max_pos
    min_weight = target_weight * 0.5
    stop_pct = float(combo["daily_close_stop_pct"])
    alpha_model = str(combo["alpha_model"]).upper()
    alpha_col, alpha_pct_col = _alpha_columns_for_model(alpha_model)
    missing_before_exit = 2

    for date in all_dates:
        date = pd.Timestamp(date)
        # exits first
        for perm, pos in list(positions.items()):
            px = _exact_price(history, perm, date)
            reason = None
            exit_px = None
            if px is None:
                pos.missing_sessions += 1
                if pos.missing_sessions >= missing_before_exit:
                    exit_px = float(pos.last_seen_price)
                    reason = "SECURITY_DISCONTINUATION"
            else:
                pos.last_seen_price = px
                pos.last_seen_date = date
                pos.missing_sessions = 0
                pos.held_days += 1
                if px <= pos.entry_price * (1.0 - stop_pct):
                    reason = "CLOSE_STOP_PROXY"
                    exit_px = px
                elif pos.held_days >= 20:
                    reason = "MODEL_HORIZON"
                    exit_px = px
            if reason:
                proceeds = pos.shares * float(exit_px)
                fee = proceeds * cost
                cash += proceeds - fee
                invested = pos.entry_notional + pos.entry_cost
                pnl = proceeds - fee - invested
                trades.append({
                    "permno": perm,
                    "ticker": pos.ticker,
                    "signal_source": pos.signal_source,
                    "entry_date": pos.entry_date,
                    "exit_date": date,
                    "entry_price": pos.entry_price,
                    "exit_price": float(exit_px),
                    "price_basis": "CRSP_RET_REINVESTED_WEALTH_INDEX",
                    "entry_weight": pos.entry_notional / (pos.entry_notional / target_weight) if target_weight > 0 else target_weight,
                    "net_pnl_usd": pnl,
                    "net_return": pnl / invested if invested > 0 else 0.0,
                    "exit_reason": reason,
                })
                del positions[perm]

        # pending next-close entries after exits
        for order in pending:
            perm = int(order["permno"])
            if perm in positions or len(positions) >= max_pos:
                continue
            px = _exact_price(history, perm, date)
            if px is None:
                continue
            open_value = sum(p.shares * _current_or_last_price(history, p.permno, date, p.last_seen_price or p.entry_price) for p in positions.values())
            eq_now = cash + open_value
            gross_room = max(0.0, eq_now - open_value)
            target = eq_now * target_weight
            cash_room = max(0.0, cash / (1.0 + cost))
            notional = min(target, gross_room, cash_room)
            if eq_now <= 0 or notional / eq_now < min_weight - 1e-12:
                continue
            shares = notional / px
            fee = notional * cost
            cash -= notional + fee
            positions[perm] = ScreenPosition(
                permno=perm,
                ticker=str(order["ticker"]),
                signal_source=str(order["signal_source"]),
                entry_date=date,
                entry_price=px,
                shares=shares,
                entry_notional=notional,
                entry_cost=fee,
                held_days=1,
                last_seen_price=px,
                last_seen_date=date,
            )
        pending = []

        # discover new candidates only when there is a vacancy
        vacancies = max_pos - len(positions)
        pdata = panels.get((date, _stop_key(stop_pct)))
        scan = False
        source = None
        if vacancies > 0 and pdata is not None and (last_signal is None or date <= last_signal):
            scan = bool(pdata["is_month_end"])
            source = "MONTH_END" if scan else None
        if scan and float(pdata["breadth_above_sma200"]) >= float(combo["min_market_breadth_200"]):
            alpha_pct = pdata[alpha_pct_col]
            mask = (
                (alpha_pct >= float(combo["min_alpha_percentile"]))
                & (pdata["mc_stop_probability_ucb"] <= float(combo["max_mc_stop_probability"]))
                & (pdata["risk_volatility"] <= float(combo["max_risk_volatility"]))
            )
            # Numerical quality is fixed rather than optimized; alpha ranking never uses MC mean.
            mask = mask & pdata["mc_converged"] & (pdata["mc_mean_se"] <= 0.0015)
            held = set(positions)
            for idx in pdata["orders"][alpha_model]:
                if len(pending) >= vacancies:
                    break
                if not bool(mask[int(idx)]):
                    continue
                perm = int(pdata["permno"][int(idx)])
                if perm in held or any(int(x["permno"]) == perm for x in pending):
                    continue
                pending.append({
                    "permno": perm,
                    "ticker": str(pdata["ticker"][int(idx)]),
                    "signal_source": source,
                })

        open_value = sum(p.shares * _current_or_last_price(history, p.permno, date, p.last_seen_price or p.entry_price) for p in positions.values())
        total = cash + open_value
        equity.append({
            "date": date,
            "equity": total,
            "cash": cash,
            "gross_exposure": open_value / total if total > 0 else 0.0,
            "positions": len(positions),
        })

    eq = pd.DataFrame(equity)
    tr = pd.DataFrame(trades)
    return eq, tr


def _combo_catalog(grid: dict) -> pd.DataFrame:
    v = grid["variables"]
    keys = [
        "max_active_positions",
        "daily_close_stop_pct",
        "alpha_model",
        "min_alpha_percentile",
        "max_mc_stop_probability",
        "max_risk_volatility",
        "min_market_breadth_200",
    ]
    rows = []
    for i, vals in enumerate(product(*(v[k] for k in keys)), 1):
        row = {"combo_id": f"G{i:04d}"}
        row.update(dict(zip(keys, vals)))
        rows.append(row)
    out = pd.DataFrame(rows)
    expected = int(grid["combination_count"])
    if len(out) != expected:
        raise RuntimeError(f"Grid combination_count={expected}, actual={len(out)}")
    return out



def _trade_signature(trades: pd.DataFrame) -> str:
    if trades is None or trades.empty:
        return "EMPTY"
    cols = [c for c in ("permno", "entry_date", "exit_date", "exit_reason") if c in trades.columns]
    x = trades[cols].copy().sort_values(cols, kind="stable")
    payload = x.astype(str).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _screen_metrics(combo: dict, eq: pd.DataFrame, trades: pd.DataFrame) -> tuple[dict, list[dict]]:
    pm = portfolio_metrics(eq["equity"])
    fold_agg, fold_rows = _fold_metrics(eq, trades)
    stops = int((trades.get("exit_reason", pd.Series(dtype=str)) == "CLOSE_STOP_PROXY").sum()) if len(trades) else 0
    trade_signature = _trade_signature(trades)
    row = {
        **combo,
        "trade_signature": trade_signature,
        **{f"full_{k}": v for k, v in pm.items()},
        "closed_trades": int(len(trades)),
        "win_rate": float((pd.to_numeric(trades["net_return"], errors="coerce") > 0).mean()) if len(trades) else 0.0,
        "trade_return_profit_factor": _trade_return_pf(trades),
        "portfolio_dollar_profit_factor": _dollar_pf(trades),
        "average_net_trade_return": float(pd.to_numeric(trades["net_return"], errors="coerce").mean()) if len(trades) else 0.0,
        "stop_trades": stops,
        "stop_rate": float(stops / len(trades)) if len(trades) else 0.0,
        "average_gross_exposure": float(pd.to_numeric(eq["gross_exposure"], errors="coerce").mean()),
        "max_gross_exposure": float(pd.to_numeric(eq["gross_exposure"], errors="coerce").max()),
        **fold_agg,
    }
    for f in fold_rows:
        f["combo_id"] = combo["combo_id"]
    return row, fold_rows


def _apply_robust_scores(df: pd.DataFrame, weights: dict, min_trades: int) -> pd.DataFrame:
    x = df.copy()
    trades_ok = pd.to_numeric(x["closed_trades"], errors="coerce").fillna(0) >= int(min_trades)
    ret_ok = pd.to_numeric(x["full_total_return"], errors="coerce").fillna(-np.inf) > 0.0
    pf_ok = pd.to_numeric(x["portfolio_dollar_profit_factor"], errors="coerce").replace([np.inf], 999.0).fillna(0.0) > 1.0
    folds_ok = pd.to_numeric(x["folds_with_trades"], errors="coerce").fillna(0) >= len(FOLDS)
    consistency_ok = pd.to_numeric(x["positive_fold_fraction"], errors="coerce").fillna(0.0) >= 0.60
    x["optimizer_eligible"] = trades_ok & ret_ok & pf_ok & folds_ok & consistency_ok
    metrics = {
        "full_sharpe_percentile": ("full_sharpe", True),
        "median_fold_sharpe_percentile": ("median_fold_sharpe", True),
        "portfolio_dollar_profit_factor_percentile": ("portfolio_dollar_profit_factor", True),
        "max_drawdown_percentile": ("full_max_drawdown", True),  # less negative is larger/better
        "worst_fold_return_percentile": ("worst_fold_return", True),
        "positive_fold_fraction_percentile": ("positive_fold_fraction", True),
    }
    for outcol, (col, asc) in metrics.items():
        vals = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        x[outcol] = vals.rank(pct=True, ascending=asc, method="average")
    x["robust_score"] = 0.0
    for k, w in weights.items():
        x["robust_score"] += float(w) * pd.to_numeric(x[k], errors="coerce").fillna(0.0)
    x.loc[~x["optimizer_eligible"], "robust_score"] = -np.inf
    x = x.sort_values(["robust_score", "full_sharpe", "full_cagr"], ascending=False).reset_index(drop=True)
    x["optimizer_rank"] = np.arange(1, len(x) + 1)
    return x



def _select_diverse_finalists(screen: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select unique realized screen strategies, round-robin across alpha families."""
    eligible = screen[np.isfinite(pd.to_numeric(screen["robust_score"], errors="coerce"))].copy()
    eligible = eligible.drop_duplicates(subset=["trade_signature"], keep="first")
    if eligible.empty:
        return eligible
    chosen = []
    used = set()
    # Round-robin prevents a numerically duplicated alpha family from consuming all exact slots.
    for rank_within in range(len(eligible)):
        added = False
        for family in ALPHA_MODELS:
            fam = eligible[eligible["alpha_model"].astype(str).str.upper() == family]
            if rank_within >= len(fam):
                continue
            rec = fam.iloc[rank_within]
            cid = str(rec["combo_id"])
            if cid in used:
                continue
            chosen.append(rec)
            used.add(cid)
            added = True
            if len(chosen) >= int(n):
                return pd.DataFrame(chosen).reset_index(drop=True)
        if not added and len(chosen) >= len(eligible):
            break
    # Fill any remaining slots globally by robust score, still unique by trade signature.
    for _, rec in eligible.iterrows():
        cid = str(rec["combo_id"])
        if cid in used:
            continue
        chosen.append(rec)
        used.add(cid)
        if len(chosen) >= int(n):
            break
    return pd.DataFrame(chosen).reset_index(drop=True)

def _apply_combo_to_spec(base: dict, combo: dict) -> dict:
    spec = deepcopy(base)
    s = spec["standalone"]
    n = int(combo["max_active_positions"])
    w = 1.0 / n
    s.update({
        "max_active_positions": n,
        "max_new_entries_per_rebalance": n,
        "weekly_refill_max_new_entries": 0,
        "target_gross_exposure": 1.0,
        "max_gross_exposure": 1.0,
        "max_sector_weight": 1.0,
        "fixed_position_weight": w,
        "max_position_weight": w,
        "min_position_weight": w * 0.5,
        "position_sizing": "FIXED_EQUAL_WEIGHT",
        "emergency_stop_pct": float(combo["daily_close_stop_pct"]),
        "stop_execution_model": "DAILY_CLOSE_STOP_PROXY",
        "entry_schedule": "MONTH_END_ONLY",
        "alpha_model": str(combo["alpha_model"]),
        "min_alpha_percentile": float(combo["min_alpha_percentile"]),
        "max_mc_stop_probability": float(combo["max_mc_stop_probability"]),
        "max_risk_volatility": float(combo["max_risk_volatility"]),
        "min_market_breadth_200": float(combo["min_market_breadth_200"]),
        "prefilter_n": int(base["standalone"].get("prefilter_n", 200)),
        "weekly_vacancy_refill_enabled": False,
        "min_probability_positive": None,
        "min_expected_shortfall_5": None,
        "max_probability_loss_10": None,
    })
    return spec


def _exact_metrics(combo: dict, eq: pd.DataFrame, trades: pd.DataFrame) -> tuple[dict, list[dict]]:
    pm = portfolio_metrics(eq["equity"])
    fold_agg, fold_rows = _fold_metrics(eq, trades)
    er = trades.get("exit_reason", pd.Series(dtype=str)) if len(trades) else pd.Series(dtype=str)
    stops = int((er == "EMERGENCY_STOP").sum())
    trade_signature = _trade_signature(trades)
    row = {
        **combo,
        "trade_signature": trade_signature,
        **{f"full_{k}": v for k, v in pm.items()},
        "closed_trades": int(len(trades)),
        "win_rate": float((pd.to_numeric(trades["net_return"], errors="coerce") > 0).mean()) if len(trades) else 0.0,
        "trade_return_profit_factor": _trade_return_pf(trades),
        "portfolio_dollar_profit_factor": _dollar_pf(trades),
        "average_net_trade_return": float(pd.to_numeric(trades["net_return"], errors="coerce").mean()) if len(trades) else 0.0,
        "stop_trades": stops,
        "stop_rate": float(stops / len(trades)) if len(trades) else 0.0,
        "average_gross_exposure": float(pd.to_numeric(eq["gross_exposure"], errors="coerce").mean()),
        "max_gross_exposure": float(pd.to_numeric(eq["gross_exposure"], errors="coerce").max()),
        **fold_agg,
    }
    for f in fold_rows:
        f["combo_id"] = combo["combo_id"]
    return row, fold_rows



def _exact_entry_candidates(
    *,
    combo: dict,
    combo_spec: dict,
    date: pd.Timestamp,
    source: str,
    positions: dict[int, ScreenPosition],
    prefilters: _PrefilterStore,
    history: HistoryStore,
    cal_cache: dict,
    forecast_cache: dict,
    collect_details: bool,
    forecast_rows: list[dict],
) -> list[dict]:
    """Exact entry forecasts from the dynamic top-N prefilter after held-name exclusion."""
    pre = prefilters.get(date)
    if pre.empty:
        return []
    s = combo_spec["standalone"]
    breadth = float(pd.to_numeric(pre.get("breadth_above_sma200", pd.Series([np.nan])), errors="coerce").iloc[0])
    if not math.isfinite(breadth) or breadth < float(combo["min_market_breadth_200"]):
        return []
    alpha_model = str(combo["alpha_model"]).upper()
    alpha_col, alpha_pct_col = _alpha_columns_for_model(alpha_model)
    held = set(positions)
    x = pre[~pre["permno"].astype(int).isin(held)].head(int(s["prefilter_n"]))
    purpose = "standalone_month_end_v9"
    n_paths = int(s["entry_n_paths"])
    rows = []
    for rec in x.to_dict("records"):
        perm = int(rec["permno"])
        ticker = str(rec["ticker"])
        raw_cal = _calibration_at(perm, date, history, combo_spec, cal_cache)
        if raw_cal is None:
            continue
        reg_cal = _regularized_calibration_from_record(raw_cal, rec)
        cal_cache[(perm, pd.Timestamp(date))] = reg_cal
        f = _terminal_forecast(
            perm, ticker, date, float(rec["close"]), history, combo_spec,
            cal_cache, forecast_cache,
            horizon_days=int(combo_spec["standalone"]["forecast_horizon_days"]),
            n_paths=n_paths, purpose=purpose, cal_override=reg_cal,
        )
        if f is None:
            continue
        risk_score, risk_vol = _risk_allocation_score(f, combo_spec)
        score = float(rec.get(alpha_col, 0.5))
        alpha_pct = float(rec.get(alpha_pct_col, 0.5))
        stop_ucb = float(f.get("mc_p_stop", 1.0)) + float(s.get("mc_stop_risk_z", 1.0)) * float(f.get("mc_p_stop_se", 0.0))
        passed = (
            alpha_pct >= float(combo["min_alpha_percentile"])
            and stop_ucb <= float(combo["max_mc_stop_probability"])
            and risk_vol <= float(combo["max_risk_volatility"])
            and (not bool(s.get("require_mc_converged", True)) or bool(f.get("mc_converged", False)))
            and (s.get("max_mc_mean_se") is None or float(f.get("mc_mean_se", float("inf"))) <= float(s["max_mc_mean_se"]))
        )
        row = {
            "date": date,
            "permno": perm,
            "ticker": ticker,
            "sector": str(rec.get("sector", "Other")),
            "signal_source": source,
            "persistent_drift": float(rec["persistent_drift"]),
            "alpha_model": alpha_model,
            "alpha_score": float(score),
            "alpha_percentile": float(alpha_pct),
            "breadth_above_sma200": breadth,
            "mc_stop_probability_ucb": float(stop_ucb),
            "model_score": float(score),
            "risk_allocation_score": float(risk_score),
            "risk_volatility": float(risk_vol),
            "pass": bool(passed),
            **f,
        }
        rows.append(row)
        if collect_details:
            forecast_rows.append(dict(row, forecast_type=("MONTH_END_ENTRY" if source == "MONTH_END" else "WEEKLY_REFILL_ENTRY")))
    passed_rows = [r for r in rows if r["pass"]]
    passed_rows.sort(key=lambda r: (float(r["model_score"]), int(r["permno"])), reverse=True)
    return passed_rows


def _exact_replay(
    combo: dict,
    all_dates: pd.DatetimeIndex,
    prefilters: _PrefilterStore,
    history: HistoryStore,
    cfg: dict,
    base_spec: dict,
    last_signal: pd.Timestamp | None,
    cal_cache: dict,
    forecast_cache: dict,
    *,
    collect_details: bool = False,
):
    """Memory-bounded exact finalist replay.

    Entry forecasts use the same requested 1024/512 minimum path budgets as
    the standalone engine, with adaptive RQMC escalation when precision requires it. Position rechecks are also stochastic/exact. Daily
    prices come from the compact HistoryStore, so no full feature DataFrame is
    required in RAM.
    """
    spec = _apply_combo_to_spec(base_spec, combo)
    s = spec["standalone"]
    cost = float(cfg["execution"]["transaction_cost_bps_per_side"]) / 10000.0
    initial = float(s["initial_capital"])
    cash = initial
    positions: dict[int, ScreenPosition] = {}
    pending: list[dict] = []
    trades: list[dict] = []
    events: list[dict] = []
    forecast_rows: list[dict] = []
    equity: list[dict] = []
    month_end, week_end = _rebalance_date_sets(all_dates)
    max_pos = int(combo["max_active_positions"])
    target_weight = 1.0 / max_pos
    min_weight = target_weight * 0.5
    stop_pct = float(combo["daily_close_stop_pct"])
    missing_before_exit = 2

    for date in all_dates:
        date = pd.Timestamp(date)
        # Exits first, matching the repaired live execution ordering.
        for perm, pos in list(positions.items()):
            px = _exact_price(history, perm, date)
            reason = None
            risk_reason = ""
            exit_px = None
            if px is None:
                pos.missing_sessions += 1
                if collect_details:
                    events.append({"date": date, "ticker": pos.ticker, "permno": perm, "event": "SECURITY_QUOTE_MISSING", "missing_global_sessions": pos.missing_sessions})
                if pos.missing_sessions >= missing_before_exit:
                    exit_px = float(pos.last_seen_price)
                    reason = "SECURITY_DISCONTINUATION"
            else:
                pos.last_seen_price = px
                pos.last_seen_date = date
                pos.missing_sessions = 0
                pos.held_days += 1
                if px <= pos.entry_price * (1.0 - stop_pct):
                    reason = "EMERGENCY_STOP"
                    exit_px = px
                elif pos.held_days >= int(s["max_holding_days"]):
                    reason = "MODEL_HORIZON"
                    exit_px = px
                elif pos.held_days % int(s["recheck_every_days"]) == 0:
                    remaining = max(1, int(s["max_holding_days"]) - pos.held_days)
                    f = _terminal_forecast(
                        perm, pos.ticker, date, px, history, spec,
                        cal_cache, forecast_cache,
                        horizon_days=remaining,
                        n_paths=int(s["recheck_n_paths"]),
                        purpose="standalone_recheck_v9",
                    )
                    if f is not None:
                        exit_now, risk_reason, streak = _risk_exit(f, spec, pos.negative_drift_rechecks)
                        pos.negative_drift_rechecks = streak
                        if collect_details:
                            forecast_rows.append({
                                "date": date, "ticker": pos.ticker, "permno": perm,
                                "forecast_type": "POSITION_RECHECK", "signal_source": "POSITION_RECHECK",
                                "horizon_days": remaining, **f, "pass": not exit_now,
                                "reason": risk_reason, "negative_drift_rechecks": streak,
                            })
                        if exit_now:
                            reason = "MODEL_RISK_EXIT"
                            exit_px = px
            if reason:
                if exit_px is None:
                    exit_px = float(pos.last_seen_price)
                proceeds = pos.shares * float(exit_px)
                fee = proceeds * cost
                cash += proceeds - fee
                invested = pos.entry_notional + pos.entry_cost
                pnl = proceeds - fee - invested
                trades.append({
                    "permno": perm, "ticker": pos.ticker, "signal_source": pos.signal_source,
                    "entry_date": pos.entry_date, "exit_date": date,
                    "entry_price": pos.entry_price, "exit_price": float(exit_px),
                    "price_basis": "CRSP_RET_REINVESTED_WEALTH_INDEX", "unit_type": "SYNTHETIC_WEALTH_UNITS",
                    "entry_notional_usd": pos.entry_notional, "entry_cost_usd": pos.entry_cost,
                    "exit_cost_usd": fee, "net_pnl_usd": pnl,
                    "net_return": pnl / invested if invested > 0 else 0.0,
                    "exit_reason": reason, "risk_reason": risk_reason,
                })
                if collect_details:
                    events.append({"date": date, "ticker": pos.ticker, "permno": perm, "event": "EXIT", "exit_reason": reason, "price": float(exit_px)})
                del positions[perm]

        # Previously-known orders execute at today's close after exits free capacity.
        for order in pending:
            perm = int(order["permno"])
            if perm in positions or len(positions) >= max_pos:
                continue
            px = _exact_price(history, perm, date)
            if px is None:
                continue
            open_value = sum(
                p.shares * _current_or_last_price(history, p.permno, date, p.last_seen_price or p.entry_price)
                for p in positions.values()
            )
            eq_now = cash + open_value
            gross_room = max(0.0, eq_now - open_value)
            target = eq_now * target_weight
            cash_room = max(0.0, cash / (1.0 + cost))
            notional = min(target, gross_room, cash_room)
            if eq_now <= 0 or notional / eq_now < min_weight - 1e-12:
                continue
            fee = notional * cost
            shares = notional / px
            cash -= notional + fee
            positions[perm] = ScreenPosition(
                permno=perm, ticker=str(order["ticker"]), signal_source=str(order["signal_source"]),
                entry_date=date, entry_price=px, shares=shares,
                entry_notional=notional, entry_cost=fee, held_days=1,
                last_seen_price=px, last_seen_date=date, negative_drift_rechecks=0,
            )
            if collect_details:
                events.append({"date": date, "ticker": str(order["ticker"]), "permno": perm, "event": "ENTRY", "signal_source": order["signal_source"], "price": px, "model_score": order["model_score"]})
        pending = []

        vacancies = max_pos - len(positions)
        source = None
        if vacancies > 0 and (last_signal is None or date <= last_signal) and date in month_end:
            source = "MONTH_END"
        if source is not None:
            candidates = _exact_entry_candidates(
                combo=combo, combo_spec=spec, date=date, source=source,
                positions=positions, prefilters=prefilters, history=history,
                cal_cache=cal_cache, forecast_cache=forecast_cache,
                collect_details=collect_details, forecast_rows=forecast_rows,
            )
            held = set(positions)
            for cand in candidates:
                if len(pending) >= vacancies:
                    break
                perm = int(cand["permno"])
                if perm in held or any(int(o["permno"]) == perm for o in pending):
                    continue
                pending.append(cand)
                if collect_details:
                    events.append({"date": date, "ticker": cand["ticker"], "permno": perm, "event": "SIGNAL", "signal_source": source, "model_score": cand["model_score"]})

        open_value = sum(
            p.shares * _current_or_last_price(history, p.permno, date, p.last_seen_price or p.entry_price)
            for p in positions.values()
        )
        total = cash + open_value
        equity.append({
            "date": date, "equity": total, "cash": cash,
            "gross_exposure": open_value / total if total > 0 else 0.0,
            "positions": len(positions),
        })

    unresolved = [
        {"permno": p.permno, "ticker": p.ticker, "entry_date": p.entry_date}
        for p in positions.values()
    ]
    return pd.DataFrame(equity), pd.DataFrame(trades), pd.DataFrame(events), pd.DataFrame(forecast_rows), unresolved


def main() -> None:
    t0 = time.time()
    identity = verify_frozen_production_identity(ROOT)
    cfg = load_config(ROOT)
    spec = load_spec(ROOT)
    if str(spec.get("patch")) != PATCH:
        raise RuntimeError(f"Expected Patch {PATCH}, found {spec.get('patch')}")
    grid = _load_grid()
    if grid.get("patch") != PATCH:
        raise RuntimeError("Optimizer grid patch mismatch")

    out = ROOT / spec["outputs"]["optimizer"]
    out.mkdir(parents=True, exist_ok=True)
    storage_cfg = spec.get("storage", {})
    _assert_disk_space(out, float(storage_cfg.get("minimum_free_gb_start", 20.0)), "optimizer startup")
    print(f"[STORAGE] Optimizer output: {out}")
    print(f"[STORAGE] Free space: {_disk_free_gb(out):.1f} GiB; atomic checkpoints + compact prefilters enabled.")
    write_json(out / "OPTIMIZER_STARTED.json", {
        "patch": PATCH,
        "status": "RUNNING",
        "started_utc": pd.Timestamp.utcnow().isoformat(),
        "combination_count": int(grid["combination_count"]),
        "development_window": "2015-2024",
        "holdout_loaded": False,
        "production_identity": identity,
    })
    print("=" * 120)
    print("PATCH 2.9.3 - MOMENTUM-PULLBACK ALPHA + HESTON-MERTON QE-M/RQMC RISK OPTIMIZER")
    print("2015-2024 ONLY. 2025+ IS NOT LOADED OR SCORED.")
    print(f"Grid: {grid['combination_count']:,} combinations; exact finalists: {grid['screening']['exact_finalists']}; final top 3 saved.")
    print("=" * 120)

    all_dates, history, warm = _prepare_data(spec, cfg)
    print(f"[DATA] memory-bounded mode; scored sessions={len(all_dates):,}; max feature date={warm['feature_max_loaded']}; calibration warm-up={warm['combined_pre2015_sessions']} sessions; HistoryStore PERMNOs={warm['history_store_permnos']:,}.")
    action_rows = history.corporate_action_events or []
    unresolved_rows = history.unresolved_return_events or []
    _atomic_to_csv(pd.DataFrame(action_rows, columns=["permno", "date", "raw_price_ratio", "economic_return", "return_to_raw_ratio", "classification"]),
                   out / "corporate_action_price_discontinuities.csv", index=False)
    _atomic_to_csv(pd.DataFrame(unresolved_rows, columns=["permno", "date", "raw_price_ratio", "return"]),
                   out / "unresolved_price_return_events.csv", index=False)
    print(f"[CORP] Return-consistent wealth price index active; {len(action_rows)} raw/return discrepancies audited; {len(unresolved_rows)} unresolved return observations flagged.")
    panel = _build_common_panel(all_dates, history, spec, grid, out, cfg)
    if len(panel) and pd.Timestamp(panel["date"].max()) > SCORE_END:
        raise RuntimeError("Optimizer panel contains post-2024 data")
    panels = _panel_structures(panel)

    last_signal = _last_full_horizon_signal_date(all_dates, 20)
    cost = float(cfg["execution"]["transaction_cost_bps_per_side"]) / 10000.0
    catalog = _combo_catalog(grid)
    _atomic_to_csv(catalog, out / "optimizer_combination_catalog.csv", index=False)

    partial_path = out / "optimizer_results_partial.csv"
    fold_partial_path = out / "optimizer_fold_results_partial.csv"
    existing = pd.read_csv(partial_path) if partial_path.exists() else pd.DataFrame()
    done_ids = set(existing["combo_id"].astype(str)) if len(existing) else set()
    result_rows = existing.to_dict("records") if len(existing) else []
    fold_rows_all = pd.read_csv(fold_partial_path).to_dict("records") if fold_partial_path.exists() else []
    progress_every = int(grid["screening"].get("progress_every_combinations", 50))

    for idx, rec in catalog.iterrows():
        combo = rec.to_dict()
        if combo["combo_id"] in done_ids:
            continue
        eq, tr = _screen_replay(combo, all_dates, panels, history, cost, last_signal)
        row, folds = _screen_metrics(combo, eq, tr)
        result_rows.append(row)
        fold_rows_all.extend(folds)
        n_done = len(result_rows)
        if n_done % progress_every == 0 or n_done == len(catalog):
            _atomic_to_csv(pd.DataFrame(result_rows), partial_path, index=False)
            _atomic_to_csv(pd.DataFrame(fold_rows_all), fold_partial_path, index=False)
            print(f"[SCREEN] {n_done:,}/{len(catalog):,} combinations complete; elapsed={(time.time()-t0)/60:.1f} min")

    screen = pd.DataFrame(result_rows)
    screen = _apply_robust_scores(
        screen,
        grid["screening"]["robust_score_weights"],
        int(grid["screening"]["minimum_closed_trades"]),
    )
    _atomic_to_csv(screen, out / "optimizer_results_all.csv", index=False)
    _atomic_to_csv(pd.DataFrame(fold_rows_all), out / "optimizer_fold_results.csv", index=False)
    _atomic_to_csv(screen.head(50), out / "optimizer_top50_screen.csv", index=False)

    finalists_n = int(grid["screening"]["exact_finalists"])
    finalists = _select_diverse_finalists(screen, finalists_n)
    if len(finalists) < 3:
        raise RuntimeError("Fewer than three eligible optimizer finalists")
    print(f"[EXACT] Rerunning {len(finalists)} screening finalists through the exact engine with stochastic rechecks.")

    cal_cache_exact: dict = {}
    forecast_cache_exact: dict = {}
    prefilters = _PrefilterStore(out / "optimizer_prefilter_parts", max_cache=96)
    exact_rows = []
    exact_fold_rows = []
    exact_entry_sets: dict[str, set[tuple[int, str]]] = {}
    finalist_root = out / "exact_finalists"
    finalist_root.mkdir(exist_ok=True)
    for j, (_, r) in enumerate(finalists.iterrows(), 1):
        combo = {k: r[k] for k in catalog.columns}
        eq, tr, ev, fc, unresolved = _exact_replay(
            combo, all_dates, prefilters, history, cfg, spec, last_signal,
            cal_cache_exact, forecast_cache_exact, collect_details=False,
        )
        if unresolved:
            raise RuntimeError(f"Exact finalist {combo['combo_id']} unresolved positions: {unresolved}")
        row, folds = _exact_metrics(combo, eq, tr)
        if len(tr) and {"permno", "entry_date"}.issubset(tr.columns):
            exact_entry_sets[str(combo["combo_id"])] = {
                (int(z["permno"]), str(pd.Timestamp(z["entry_date"]).date()))
                for z in tr[["permno", "entry_date"]].to_dict("records")
            }
        else:
            exact_entry_sets[str(combo["combo_id"])] = set()
        row["screen_rank"] = int(r["optimizer_rank"])
        row["screen_robust_score"] = float(r["robust_score"])
        exact_rows.append(row)
        exact_fold_rows.extend(folds)
        d = finalist_root / str(combo["combo_id"])
        d.mkdir(exist_ok=True)
        write_json(d / "combo.json", combo)
        write_json(d / "exact_summary.json", row)
        print(f"[EXACT] {j}/{len(finalists)} {combo['combo_id']}: CAGR={row['full_cagr']:.2%} Sharpe={row['full_sharpe']:.3f} MDD={row['full_max_drawdown']:.2%} PF$={row['portfolio_dollar_profit_factor']}")

    exact = pd.DataFrame(exact_rows)
    exact = _apply_robust_scores(
        exact,
        grid["screening"]["robust_score_weights"],
        int(grid["screening"]["minimum_closed_trades"]),
    )
    exact = exact.rename(columns={"optimizer_rank": "exact_rank", "robust_score": "exact_robust_score"})
    _atomic_to_csv(exact, out / "exact_finalist_results.csv", index=False)
    _atomic_to_csv(pd.DataFrame(exact_fold_rows), out / "exact_finalist_fold_results.csv", index=False)

    top_n = int(grid["final_selection"]["top_n"])
    overlap_cap = float(grid["final_selection"].get("maximum_entry_jaccard", 0.85))
    selected_indices = []
    selected_sets = []
    for idx, row in exact.iterrows():
        entries = exact_entry_sets.get(str(row["combo_id"]), set())
        if not entries: continue
        if all(len(entries & prev) / max(1, len(entries | prev)) <= overlap_cap for prev in selected_sets):
            selected_indices.append(idx)
            selected_sets.append(entries)
        if len(selected_indices) >= top_n: break
    top3 = exact.loc[selected_indices].copy()
    if len(top3) < top_n:
        print(f"[DIVERSITY] WARNING: only {len(top3)}/{top_n} exact finalists satisfy entry Jaccard <= {overlap_cap:.2f}; not fabricating three independent strategies.")
    _atomic_to_csv(top3, out / "top_3_candidates.csv", index=False)

    # Full output rerun for the final three. Shared caches make this much cheaper than
    # three independent stochastic backtests while preserving exact engine semantics.
    top3_root = out / "top3"
    if top3_root.exists():
        shutil.rmtree(top3_root)
    top3_root.mkdir()
    top_manifest = []
    for rank, (_, r) in enumerate(top3.iterrows(), 1):
        combo = {k: r[k] for k in catalog.columns}
        eq, tr, ev, fc, unresolved = _exact_replay(
            combo, all_dates, prefilters, history, cfg, spec, last_signal,
            cal_cache_exact, forecast_cache_exact, collect_details=True,
        )
        if unresolved:
            raise RuntimeError(f"Top3 {combo['combo_id']} unresolved positions: {unresolved}")
        d = top3_root / f"R{rank}_{combo['combo_id']}"
        d.mkdir()
        _atomic_to_csv(eq, d / "equity_curve.csv", index=False)
        _atomic_to_csv(tr, d / "trade_log.csv", index=False)
        _atomic_to_csv(ev, d / "event_log.csv", index=False)
        _atomic_to_csv(fc, d / "model_forecasts.csv", index=False)
        write_json(d / "combo.json", combo)
        exact_record = json.loads(pd.Series(r).to_json())
        write_json(d / "summary.json", exact_record)
        top_manifest.append({
            "rank": rank,
            "combo_id": combo["combo_id"],
            "directory": str(d.relative_to(out)),
            "config": combo,
            "exact_metrics": exact_record,
        })

    write_json(out / "top_3_sealed_manifest.json", {
        "patch": PATCH,
        "development_window": "2015-01-01 through 2024-12-31",
        "holdout_status": "LOCKED_NOT_USED_IN_OPTIMIZER",
        "selection_basis": "Exact-engine robust score subject to <=85% pairwise entry Jaccard overlap; fewer than three reported honestly if necessary",
        "maximum_entry_jaccard": overlap_cap,
        "requested_top_n": top_n,
        "returned_top_n": len(top3),
        "diversity_limited": len(top3) < top_n,
        "top3": top_manifest,
        "note": "Top-three candidates are development finalists for analysis. This file does not unlock or run the 2025+ evaluation reserve.",
    })
    write_json(out / "OPTIMIZER_COMPLETE.json", {
        "patch": PATCH,
        "status": "PASS",
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
        "combination_count": int(len(catalog)),
        "screened": int(len(screen)),
        "exact_finalists": int(len(exact)),
        "top3": [str(x) for x in top3["combo_id"].tolist()],
        "holdout_loaded": False,
        "max_feature_date": warm["feature_max_loaded"],
        "memory_mode": warm["data_mode"],
        "production_identity": identity,
        "runtime_minutes": (time.time() - t0) / 60.0,
        "optimizer_grid_sha256": sha256_file(HERE / "DEVELOPMENT_OPTIMIZER_GRID.json"),
    })
    print("=" * 120)
    print("OPTIMIZER COMPLETE - TOP 3 DEVELOPMENT CANDIDATES")
    print(top3[["exact_rank", "combo_id", "full_cagr", "full_sharpe", "full_max_drawdown", "portfolio_dollar_profit_factor", "closed_trades"]].to_string(index=False))
    print(f"Outputs: {out}")
    print("2025+ remains LOCKED and was not loaded by this optimizer.")
    print("=" * 120)


if __name__ == "__main__":
    main()
