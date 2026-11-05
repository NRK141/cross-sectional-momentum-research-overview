from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROLLBACK = ROOT / "data" / ".f2_2_production_rollback"
CONFIG = ROOT / "config.yaml"
OUTPUTS = ROOT / "outputs"
LOCK = ROOT / "data" / "manifests" / "foundation2_2_main_forward_prefix_lock.json"
LIVE_DIRS = ("screener", "backtest")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    if path.is_file():
        h.update(path.name.encode("utf-8"))
        h.update(_sha256_file(path).encode("ascii"))
        return h.hexdigest()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(p.relative_to(path).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(_sha256_file(p).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def capture_state(label: str) -> Path:
    if ROLLBACK.exists():
        raise RuntimeError(
            f"Unresolved production rollback snapshot exists: {ROLLBACK}. Run 18_recover_production_state.cmd before starting another production cycle."
        )
    ROLLBACK.mkdir(parents=True, exist_ok=False)
    shutil.copy2(CONFIG, ROLLBACK / "config.yaml")
    manifest: dict = {
        "version": 1,
        "label": label,
        "status": "CAPTURED",
        "config_sha256": _sha256_file(CONFIG),
        "prefix_lock": {"existed": LOCK.exists(), "sha256": _sha256_file(LOCK) if LOCK.exists() else None},
        "live_outputs": {},
    }
    if LOCK.exists():
        shutil.copy2(LOCK, ROLLBACK / LOCK.name)
    for name in LIVE_DIRS:
        src = OUTPUTS / name
        manifest["live_outputs"][name] = {"existed": src.exists(), "tree_sha256": tree_hash(src)}
        if src.exists():
            shutil.copytree(src, ROLLBACK / name)
    (ROLLBACK / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ROLLBACK


def restore_state(snapshot: Path = ROLLBACK, *, verify: bool = True, remove_after: bool = True) -> dict:
    snapshot = snapshot.resolve()
    mp = snapshot / "manifest.json"
    if not mp.exists():
        raise FileNotFoundError(f"Missing rollback manifest: {mp}")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    shutil.copy2(snapshot / "config.yaml", CONFIG)
    for name in LIVE_DIRS:
        dst = OUTPUTS / name
        if dst.exists():
            shutil.rmtree(dst)
        rec = manifest["live_outputs"].get(name, {})
        if rec.get("existed"):
            shutil.copytree(snapshot / name, dst)
    if LOCK.exists():
        LOCK.unlink()
    if manifest.get("prefix_lock", {}).get("existed"):
        shutil.copy2(snapshot / LOCK.name, LOCK)

    checks = {
        "config_match": _sha256_file(CONFIG) == manifest["config_sha256"],
        "prefix_lock_match": (
            (not manifest.get("prefix_lock", {}).get("existed") and not LOCK.exists())
            or (LOCK.exists() and _sha256_file(LOCK) == manifest.get("prefix_lock", {}).get("sha256"))
        ),
        "outputs": {},
    }
    for name in LIVE_DIRS:
        expected = manifest["live_outputs"].get(name, {}).get("tree_sha256")
        actual = tree_hash(OUTPUTS / name)
        checks["outputs"][name] = {"expected": expected, "actual": actual, "match": expected == actual}
    ok = checks["config_match"] and checks["prefix_lock_match"] and all(v["match"] for v in checks["outputs"].values())
    result = {**manifest, "status": "RESTORED" if ok else "RESTORE_MISMATCH", "restore_verification": checks}
    if verify and not ok:
        raise RuntimeError(f"Production state restoration failed: {checks}")
    if remove_after and ok:
        shutil.rmtree(snapshot, ignore_errors=True)
    return result


def commit_state(snapshot: Path = ROLLBACK) -> None:
    if snapshot.exists():
        shutil.rmtree(snapshot, ignore_errors=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Restore an interrupted F2.2 production cycle from its transactional rollback snapshot.")
    ap.add_argument("--keep-snapshot", action="store_true")
    args = ap.parse_args()
    if not ROLLBACK.exists():
        print("No unresolved F2.2 production rollback snapshot exists. Nothing to recover.")
        return
    result = restore_state(ROLLBACK, verify=True, remove_after=not args.keep_snapshot)
    print("=" * 78)
    print("FOUNDATION 2.2 PRODUCTION STATE RECOVERY")
    print("=" * 78)
    print("Status: RESTORED")
    print(json.dumps(result.get("restore_verification", {}), indent=2))
    print("=" * 78)


if __name__ == "__main__":
    main()
