from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERNS = {
    "legacy_euler_or_truncation": re.compile(r"full[_ -]?trunc|euler|maximum\s*\(\s*v|np\.maximum\([^\n]*variance", re.I),
    "hard_drift_filter": re.compile(r"(drift|\bmu\b).{0,80}(threshold|cutoff|reject|filter)|(?:threshold|cutoff).{0,80}(drift|\bmu\b)", re.I),
    "signal_close_payoff": re.compile(r"signal[_ ]?close|entry[_ ]?price.{0,40}(close|s0)|stop.{0,80}(signal|s0)", re.I),
    "monte_carlo_entrypoint": re.compile(r"monte.?carlo|heston|merton|simulate.*path|forecast.*panel", re.I),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-root", default=".")
    ap.add_argument("--output", default="patch_2_8_0_integration_audit.txt")
    args = ap.parse_args()
    root = Path(args.research_root).resolve()
    src = root
    if not (root / "stochastic_core.py").exists():
        raise SystemExit(f"ERROR: stochastic_core.py not found under {root}")

    lines = []
    for file in sorted(src.rglob("*.py")):
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = file.relative_to(root)
        for no, line in enumerate(text.splitlines(), start=1):
            if "mc_v280" in str(rel) or str(rel) in {"validate_patch_2_8_0.py", "benchmark_mc_v280.py", "run_mc_v280_from_csv.py", "audit_legacy_mc_integration.py"}:
                continue
            for label, rx in PATTERNS.items():
                if rx.search(line):
                    lines.append(f"[{label}] {rel}:{no}: {line.strip()}")

    out = root / args.output
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print("=" * 78)
    print("PATCH 2.8.0 - LEGACY MONTE CARLO INTEGRATION AUDIT")
    print("=" * 78)
    print(f"Matches: {len(lines):,}")
    print(f"Report:  {out}")
    if lines:
        print("\nFirst matches:")
        for line in lines[:40]:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
