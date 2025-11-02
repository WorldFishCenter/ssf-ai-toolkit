from __future__ import annotations
from pathlib import Path
import joblib

def save_joblib(obj, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)

def load_joblib(path: str | Path):
    return joblib.load(Path(path))
