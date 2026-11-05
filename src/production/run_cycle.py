from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .forward_data_readiness import build_report
from .forward_prefix_lock import commit_lock, verify_lock
from .identity import ROOT, RELEASE, validate_active_identity
from .production_state import capture_state, commit_state, restore_state

MANIFEST = ROOT / "data" / "manifests" / "f2_2_production_cycle.json"
LOG = ROOT / "logs" / "f2_2_production_cycle.log"


def _run(cmd: list[str], log, label: str) -> None:
    log.write("\n" + "=" * 78 + "\n" + label + "\n" + "=" * 78 + "\n")
    log.flush()
    p = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"{label} failed with code {p.returncode}: {cmd}")


def _cmd(name: str) -> list[str]:
    p = ROOT / name
    if not p.exists():
        raise FileNotFoundError(p)
    if os.name != "nt":
        raise RuntimeError("Windows .cmd launcher required for the WRDS data build")
    return ["cmd.exe", "/d", "/c", "call", str(p)]


def _write(payload: dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the sole Foundation 2.2 production strategy transactionally.")
    ap.add_argument("--target-start", default="2026-01-01")
    ap.add_argument("--skip-data-build", action="store_true")
    args = ap.parse_args()

    ident = validate_active_identity()
    if not ident.get("pass"):
        raise RuntimeError(f"F2.2 production identity failed: {ident}")
    prior = verify_lock()
    if prior.get("status") == "FAIL":
        raise RuntimeError(f"Existing F2.2 production prefix lock is broken: {prior}")

    before = build_report(args.target_start)
    if not before.get("flags", {}).get("raw_forward_available"):
        raise RuntimeError("No configured forward raw source covers the requested period")

    snapshot = capture_state("f2_2_production_cycle")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "STARTED",
        "release": RELEASE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_start": args.target_start,
        "skip_data_build": bool(args.skip_data_build),
        "readiness_before": before,
        "prefix_lock_before": prior,
    }
    _write(payload)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    try:
        with LOG.open("w", encoding="utf-8") as log:
            source_kind = before.get("flags", {}).get("raw_forward_source_kind")
            if not args.skip_data_build and source_kind == "wrds":
                _run(_cmd("01_build_data.cmd"), log, "F2.2 WRDS normalized/feature/benchmark build")
            elif not args.skip_data_build and source_kind == "alpaca":
                log.write("Alpaca forward source detected; WRDS-only 01_build_data.cmd intentionally skipped.\n")

            ready = build_report(args.target_start)
            payload["readiness_after_build"] = ready
            if not ready.get("flags", {}).get("normalized_features_benchmarks_forward_ready"):
                raise RuntimeError("Normalized/features/benchmarks are not forward-ready")

            _run([sys.executable, "-m", "src.screener.build_monthly_rankings"], log, "F2.2 production screener")
            _run([sys.executable, "-m", "src.backtest.run_backtest"], log, "F2.2 production backtest")
            _run([sys.executable, "-m", "src.production.canonicalize_outputs"], log, "Canonical output normalization")
            _run([sys.executable, "-m", "src.production.analyze_outputs"], log, "Hard integrity audit")

            locked = verify_lock()
            payload["prefix_lock_after_run_before_commit"] = locked
            if locked.get("status") == "FAIL":
                raise RuntimeError(f"Previously locked F2.2 evidence changed: {locked}")

            _run([sys.executable, "-m", "src.production.production_health"], log, "F2.2 production health")
            _run([sys.executable, "-m", "src.production.prospective_monitor"], log, "F2.2 prospective monitor")

            new_lock = commit_lock()
            payload["prefix_lock_committed"] = {"status": new_lock.get("status"), "cutoff": new_lock.get("cutoff")}
            _run([sys.executable, "-m", "src.production.verify"], log, "Final F2.2 production verification")

        payload["status"] = "PASS_COMMITTED"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        commit_state(snapshot)
        _write(payload)
    except Exception as exc:
        payload["status"] = "FAILED_LIVE_STATE_RESTORED"
        payload["error"] = str(exc)
        payload["failed_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            restored = restore_state(snapshot, verify=True, remove_after=True)
            payload["restore_verification"] = restored.get("restore_verification")
        except Exception as restore_exc:
            payload["status"] = "FAILED_RESTORE_ERROR"
            payload["restore_error"] = str(restore_exc)
        _write(payload)
        raise

    print("=" * 78)
    print("FOUNDATION 2.2 PRODUCTION CYCLE COMPLETE")
    print("=" * 78)
    print("Status: PASS_COMMITTED")
    print(f"Active production: {RELEASE}")
    print(f"Manifest: {MANIFEST}")
    print(f"Log: {LOG}")
    print("=" * 78)


if __name__ == "__main__":
    main()
