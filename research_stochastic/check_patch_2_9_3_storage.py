from __future__ import annotations
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = json.loads((HERE / "STOCHASTIC_RESEARCH_SPEC.json").read_text(encoding="utf-8"))
output = Path(spec["storage"]["optimizer_output_dir"])
if str(output.drive).lower() != "d:":
    raise SystemExit("[FAIL] Optimizer output is not configured for D:")
probe = output
while not probe.exists() and probe != probe.parent:
    probe = probe.parent
if not probe.exists():
    raise SystemExit("[FAIL] D: drive/project folder not accessible")
free = shutil.disk_usage(probe).free / (1024**3)
print(f"[OUTPUT] {output}")
print(f"[FREE] {free:.2f} GiB")
print(f"[TEMP] {spec['storage']['temp_dir']}")
print("[REUSE] Older 2.9.1/2.9.2 panel and prefilter parts are INVALID for this price-basis/alpha/grid change")
if free < float(spec['storage'].get('minimum_free_gb_start',20)):
    raise SystemExit("[FAIL] Insufficient space for a fresh 2.9.3 optimizer run")
print("[PASS] D: storage available. Fresh 2.9.3 checkpoints will resume only within this version.")
