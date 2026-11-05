from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
from run_development_optimizer import _find_legacy_checkpoint_source, _valid_panel_part


def main() -> None:
    spec = json.loads((HERE / "STOCHASTIC_RESEARCH_SPEC.json").read_text(encoding="utf-8"))
    grid = json.loads((HERE / "DEVELOPMENT_OPTIMIZER_GRID.json").read_text(encoding="utf-8"))
    out = Path(spec["outputs"]["optimizer"])
    root = Path(spec["storage"]["research_output_root"])
    print("=" * 96)
    print("PATCH 2.9.2 - STORAGE / LEGACY CHECKPOINT PREFLIGHT")
    print("=" * 96)
    print(f"[OUTPUT ROOT] {root}")
    print(f"[OUTPUT DIR]  {out}")
    if not root.exists():
        raise SystemExit(f"[FAIL] D: research output root does not exist: {root}")
    usage = shutil.disk_usage(root)
    print(f"[FREE]        {usage.free / 1024**3:.2f} GiB")
    if usage.free / 1024**3 < float(spec["storage"].get("minimum_free_gb_start", 20.0)):
        raise SystemExit("[FAIL] Insufficient free space for optimizer startup guard")

    source = _find_legacy_checkpoint_source(spec, out)
    if source is None:
        print("[RESUME] No moved Patch 2.9.1 checkpoint source detected. A fresh panel build will be used.")
        return
    print(f"[RESUME SOURCE] {source}")
    panel_dir = source / "optimizer_panel_parts"
    pre_dir = source / "optimizer_prefilter_parts"
    stops = [float(x) for x in grid["variables"]["daily_close_stop_pct"]]
    valid = 0
    invalid = 0
    for p in sorted(panel_dir.glob("*.csv")) if panel_dir.is_dir() else []:
        if _valid_panel_part(p, stops):
            valid += 1
        else:
            invalid += 1
    pre_count = len(list(pre_dir.glob("*.csv"))) if pre_dir.is_dir() else 0
    print(f"[RESUME] Structurally valid 2.9.1 panel date files: {valid}")
    print(f"[RESUME] Incomplete/invalid panel date files:       {invalid}")
    print(f"[RESUME] Legacy prefilter files present:            {pre_count}")
    print("[NOTE] Final import applies an additional exact top-N completeness check and may reject more files.")
    print("[PASS] Storage preflight completed.")


if __name__ == "__main__":
    main()
