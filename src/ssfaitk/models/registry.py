from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelCard:
    name: str
    version: str
    task: str
    dataset: str
    metrics: dict
    notes: str = ""
    license: str = "Apache-2.0"
    region: Optional[str] = None


# In the future: load/save registry as JSON/YAML; expose via CLI
