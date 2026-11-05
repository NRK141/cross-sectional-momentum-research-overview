from __future__ import annotations
import csv
from pathlib import Path
from src.common.config import load_config, root_path
from src.common.schema import resolve_columns


def read_headers(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        return next(csv.reader(f))

def main():
    cfg=load_config(); raw=root_path(cfg,cfg["data"]["raw_crsp_csv"])
    if not raw.exists(): raise FileNotFoundError(f"Raw CRSP CSV not found: {raw}")
    headers=read_headers(raw); cols=resolve_columns(headers)
    print(f"Raw file: {raw}")
    print(f"Size GiB: {raw.stat().st_size/1024**3:.2f}")
    print(f"Columns detected: {cols}")

if __name__ == "__main__": main()
