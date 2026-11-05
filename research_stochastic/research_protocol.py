from __future__ import annotations

from pathlib import Path
import hashlib
import json

from stochastic_core import sha256_file


PATCH = "2.9.3"
CRITICAL_CODE_FILES = [
    "run_stochastic_standalone.py",
    "stochastic_core.py",
    "reporting.py",
    "data_access.py",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_json_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def strategy_payload(spec: dict) -> dict:
    """Exact research decision/model configuration that must be frozen before holdout."""
    return {
        "patch": str(spec["patch"]),
        "random_seed": int(spec["random_seed"]),
        "model": spec["model"],
        "standalone": spec["standalone"],
        "boundary": spec["boundary"],
        "security_discontinuation": spec.get("security_discontinuation", {}),
        "development_window": spec["experiment_windows"]["development"],
        "holdout_window": spec["experiment_windows"]["holdout"],
    }


def strategy_fingerprint(spec: dict) -> str:
    return _canonical_json_hash(strategy_payload(spec))


def code_fingerprints(root: Path | None = None) -> dict[str, str]:
    root = root or _repo_root()
    rs = root / "research_stochastic"
    out: dict[str, str] = {}
    for name in CRITICAL_CODE_FILES:
        p = rs / name
        if not p.exists():
            raise FileNotFoundError(f"Critical research file missing: {p}")
        out[name] = sha256_file(p)
    return out


def freeze_path(root: Path, spec: dict) -> Path:
    return root / str(spec["outputs"]["freeze_manifest"])


def development_output_dir(root: Path, spec: dict) -> Path:
    return root / str(spec["experiment_windows"]["development"]["output_dir"])


def holdout_output_dir(root: Path, spec: dict) -> Path:
    return root / str(spec["experiment_windows"]["holdout"]["output_dir"])


def build_freeze_payload(root: Path, spec: dict) -> dict:
    dev = development_output_dir(root, spec)
    required = ["summary.json", "manifest.json", "RUN_COMPLETE.json"]
    missing = [name for name in required if not (dev / name).exists()]
    if missing:
        raise RuntimeError(
            "Cannot freeze development configuration because the completed development "
            f"bundle is missing: {missing}. Run 99_RUN_DEVELOPMENT_2015_2024.cmd first."
        )
    summary = json.loads((dev / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((dev / "manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((dev / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    run_ids = {summary.get("run_id"), manifest.get("run_id"), complete.get("run_id")}
    if None in run_ids or len(run_ids) != 1:
        raise RuntimeError(f"Development output run-id mismatch: {run_ids}")
    if summary.get("patch") != PATCH or manifest.get("spec_patch") != PATCH or complete.get("patch") != PATCH:
        raise RuntimeError("Development output was not produced by Patch 2.9.3.")
    if summary.get("research_mode") != "development":
        raise RuntimeError("Development summary does not declare research_mode=development.")
    window = summary.get("evaluation_window", {})
    if window.get("history_start") != "2014-01-01":
        raise RuntimeError(f"Development feature-history start is not 2014-01-01: {window}")
    if window.get("score_end") != "2024-12-31":
        raise RuntimeError(f"Development output score_end is not frozen at 2024-12-31: {window}")
    audit = summary.get("entry_order_audit", {})
    if int(audit.get("unresolved", 0)) != 0:
        raise RuntimeError(f"Development output has unresolved entry orders: {audit}")
    if manifest.get("max_loaded_date") and str(manifest["max_loaded_date"]) > "2024-12-31":
        raise RuntimeError("Development run loaded post-2024 rows; refuse to freeze.")
    return {
        "patch": PATCH,
        "status": "DEVELOPMENT_CONFIG_FROZEN_FOR_HOLDOUT",
        "development_run_id": next(iter(run_ids)),
        "strategy_fingerprint": strategy_fingerprint(spec),
        "strategy_payload": strategy_payload(spec),
        "code_fingerprints": code_fingerprints(root),
        "development_output_dir": str(dev),
        "development_score_start": window.get("score_start"),
        "development_score_end": window.get("score_end"),
        "holdout_score_start": spec["experiment_windows"]["holdout"]["score_start"],
        "holdout_policy": "NO_CONFIG_OR_CRITICAL_CODE_CHANGE_AFTER_FREEZE",
    }


def write_development_freeze(root: Path, spec: dict) -> Path:
    payload = build_freeze_payload(root, spec)
    p = freeze_path(root, spec)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def verify_holdout_freeze(root: Path, spec: dict) -> dict:
    p = freeze_path(root, spec)
    if not p.exists():
        raise RuntimeError(
            "HOLDOUT LOCKED. No frozen development configuration exists. "
            "Do not run 2025-present until tuning is complete; then run "
            "97_FREEZE_DEVELOPMENT_CONFIG.cmd."
        )
    freeze = json.loads(p.read_text(encoding="utf-8"))
    if freeze.get("patch") != PATCH:
        raise RuntimeError(f"Holdout freeze patch mismatch: {freeze.get('patch')} != {PATCH}")
    current_strategy = strategy_fingerprint(spec)
    if freeze.get("strategy_fingerprint") != current_strategy:
        raise RuntimeError(
            "HOLDOUT REFUSED: strategy/spec changed after the development configuration was frozen."
        )
    current_code = code_fingerprints(root)
    if freeze.get("code_fingerprints") != current_code:
        raise RuntimeError(
            "HOLDOUT REFUSED: critical research code changed after the development configuration was frozen."
        )
    return freeze
