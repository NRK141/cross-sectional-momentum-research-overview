from __future__ import annotations
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg

def root_path(cfg: dict, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return Path(cfg["_root"]) / p
