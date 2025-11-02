from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from pathlib import Path
from ..utils.io import save_joblib, load_joblib

@runtime_checkable
class ModelProtocol(Protocol):
    def fit(self, X, y) -> Any: ...
    def predict(self, X): ...

@dataclass
class BaseModel:
    artifact_path: str | None = None

    def save(self, path: str | Path) -> None:
        save_joblib(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        return load_joblib(path)
