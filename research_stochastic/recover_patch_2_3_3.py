from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = "2.3.3"
OUT = ROOT / "outputs" / "research" / "stochastic_standalone_patch_2_3_3"
STAGING = OUT.parent / ".stochastic_standalone_patch_2_3_3_staging_2_3_3"
BACKUP = OUT.parent / ".stochastic_standalone_patch_2_3_3_previous"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_bundle(path: Path) -> tuple[bool, str | None]:
    required = ["summary.json", "manifest.json", "RUN_COMPLETE.json"]
    if any(not (path / n).exists() for n in required):
        return False, None
    try:
        summary = load_json(path / "summary.json")
        manifest = load_json(path / "manifest.json")
        complete = load_json(path / "RUN_COMPLETE.json")
    except Exception:
        return False, None
    ids = {summary.get("run_id"), manifest.get("run_id"), complete.get("run_id")}
    ok = (
        summary.get("patch") == PATCH
        and manifest.get("spec_patch") == PATCH
        and complete.get("patch") == PATCH
        and complete.get("status") == "PASS"
        and None not in ids
        and len(ids) == 1
    )
    return ok, next(iter(ids)) if ok else None


def promote(src: Path, dst: Path) -> str:
    ok, run_id = valid_bundle(src)
    if not ok or run_id is None:
        raise RuntimeError(f"Source is not a complete recoverable Patch {PATCH} bundle: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    # Delete an old completion marker first so partial promotion can never appear valid.
    complete = dst / "RUN_COMPLETE.json"
    if complete.exists():
        complete.unlink()
    files = [p for p in src.iterdir() if p.is_file() and p.name != "RUN_COMPLETE.json"]
    for p in sorted(files, key=lambda x: x.name):
        tmp = dst / (p.name + ".recover_tmp")
        shutil.copy2(p, tmp)
        os.replace(tmp, dst / p.name)
    tmp = dst / "RUN_COMPLETE.json.recover_tmp"
    shutil.copy2(src / "RUN_COMPLETE.json", tmp)
    os.replace(tmp, dst / "RUN_COMPLETE.json")
    ok2, run_id2 = valid_bundle(dst)
    if not ok2 or run_id2 != run_id:
        raise RuntimeError("Recovered output failed final consistency validation")
    return run_id


def main() -> None:
    print(f"[CHECK] Existing Patch {PATCH} output: {OUT}")
    ok, run_id = valid_bundle(OUT)
    if ok:
        print(f"[PASS] Existing Patch {PATCH} output is already complete; run_id={run_id}")
        return

    candidates = []
    if STAGING.exists():
        candidates.append(("staging", STAGING))
    if BACKUP.exists():
        candidates.append(("backup", BACKUP))

    if not candidates:
        raise SystemExit(
            "[NO RECOVERY] No Patch 2.3.3 staging/backup directory exists. "
            "The expensive calculation cannot be salvaged from disk; run Patch 2.3.4 instead."
        )

    for label, path in candidates:
        ok, rid = valid_bundle(path)
        print(f"[CHECK] {label}: {path} -> {'recoverable' if ok else 'not complete'}")
        if ok:
            run_id = promote(path, OUT)
            print(f"[PASS] Recovered Patch {PATCH} outputs without rerunning the model; run_id={run_id}")
            return

    raise SystemExit(
        "[NO RECOVERY] Staging/backup directories exist but none contains a complete Patch 2.3.3 bundle. "
        "Run Patch 2.3.4 for a clean execution."
    )


if __name__ == "__main__":
    main()
