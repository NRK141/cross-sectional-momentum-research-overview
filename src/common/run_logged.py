from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .config import PROJECT_ROOT


def run_module(module: str, log_path: Path, append: bool = False, module_args: list[str] | None = None) -> int:
    """Run a project module with the current Python and tee combined output.

    This intentionally avoids nested CMD/PowerShell redirection. The child
    process writes stdout/stderr into one stream, which is mirrored to both
    the visible console and the requested log file. The child's exit code is
    returned unchanged.
    """
    log_path = log_path if log_path.is_absolute() else PROJECT_ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    cmd = [sys.executable, "-u", "-m", module, *(module_args or [])]

    try:
        with log_path.open(mode, encoding="utf-8", newline="") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                # Mirror to the interactive CMD window and the UTF-8 log.
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return int(proc.wait())
    except Exception as exc:
        msg = f"LOG RUNNER FAILED before/during {module}: {type(exc).__name__}: {exc}\n"
        sys.stderr.write(msg)
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(msg)
        except Exception:
            pass
        return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a Python module and mirror output to console + log.")
    ap.add_argument("--module", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--module-arg", action="append", default=[], help="Argument passed to the child module; repeat as needed.")
    args = ap.parse_args()
    raise SystemExit(run_module(args.module, Path(args.log), append=args.append, module_args=args.module_arg))


if __name__ == "__main__":
    main()
