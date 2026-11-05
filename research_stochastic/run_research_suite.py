from __future__ import annotations

from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    cmd = [sys.executable, str(HERE / "run_stochastic_standalone.py")]
    print("\n" + "=" * 108)
    print("RUNNING STANDALONE HESTON-MERTON V4:", " ".join(cmd))
    print("=" * 108)
    subprocess.run(cmd, cwd=ROOT, check=True)
    print("\nPatch 2.3.5 standalone Heston-Merton research run completed successfully.")


if __name__ == "__main__":
    main()
