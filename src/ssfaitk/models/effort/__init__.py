from __future__ import annotations

from .effort_classifier import EffortClassifier
# from .effort_statistical import EffortStatistical
# Switched to v2 for shore distance filtering support
from .statistical_effort_v2 import StatisticalEffortClassifier # , predict_fishing_effort

__all__ = [
    "EffortClassifier",
    # "EffortStatistical",
    "StatisticalEffortClassifier",
    # "predict_fishing_effort",
]
