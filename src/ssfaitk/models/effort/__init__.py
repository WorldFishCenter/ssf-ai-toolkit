from __future__ import annotations

from .effort_classifier import EffortClassifier
# from .effort_statistical import EffortStatistical
from .statistical_effort_enhanced import StatisticalEffortClassifier # , predict_fishing_effort

__all__ = [
    "EffortClassifier",
    # "EffortStatistical",
    "StatisticalEffortClassifier",
    # "predict_fishing_effort",
]
