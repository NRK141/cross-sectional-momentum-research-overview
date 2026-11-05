from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
LIVE_CONFIG = ROOT / "config.yaml"
SPEC = ROOT / "PRODUCTION_SPEC.json"
ENGINE = ROOT / "src" / "backtest" / "engine.py"
BASELINE_DIR = ROOT / "outputs" / "baselines" / "foundation2_2_production_v0_2_2"
LIVE_BACKTEST = ROOT / "outputs" / "backtest"
LIVE_SCREENER = ROOT / "outputs" / "screener"
RELEASE = "Foundation 2.2 / v0.2.2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spec() -> dict[str, Any]:
    if not SPEC.exists():
        raise FileNotFoundError(f"Missing production specification: {SPEC}")
    return json.loads(SPEC.read_text(encoding="utf-8"))


def load_config() -> dict[str, Any]:
    if not LIVE_CONFIG.exists():
        raise FileNotFoundError(LIVE_CONFIG)
    return yaml.safe_load(LIVE_CONFIG.read_text(encoding="utf-8-sig")) or {}


def _get(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def validate_active_identity() -> dict[str, Any]:
    try:
        spec = load_spec()
    except Exception as exc:
        return {"pass": False, "checks": {"production_spec": False}, "error": str(exc)}
    try:
        cfg = load_config()
    except Exception as exc:
        return {"pass": False, "checks": {"config_exists": False}, "error": str(exc)}

    expected_cfg_sha = str(spec.get("config_sha256", ""))
    expected_engine_sha = str(spec.get("engine_sha256", ""))
    checks: dict[str, bool] = {
        "production_spec": spec.get("status") == "FINAL_PRODUCTION",
        "release": spec.get("release") == RELEASE and _get(cfg, "project.release") == RELEASE,
        "config_sha256": bool(expected_cfg_sha) and sha256_file(LIVE_CONFIG) == expected_cfg_sha,
        "engine_sha256": ENGINE.exists() and bool(expected_engine_sha) and sha256_file(ENGINE) == expected_engine_sha,
    }
    for dotted, expected in (spec.get("frozen_parameter_checks") or {}).items():
        actual = _get(cfg, dotted)
        if isinstance(expected, float):
            try:
                ok = abs(float(actual) - expected) <= 1e-12
            except Exception:
                ok = False
        else:
            ok = actual == expected
        checks[f"parameter:{dotted}"] = ok
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "release": RELEASE,
        "config_sha256": sha256_file(LIVE_CONFIG),
        "engine_sha256": sha256_file(ENGINE) if ENGINE.exists() else None,
    }
