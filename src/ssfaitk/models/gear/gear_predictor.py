from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ...utils.logger import get_logger
from ..base import BaseModel

logger = get_logger(__name__)

FEATURES = ["duration_hours", "distance_nm", "mean_sog"]
LABEL_COL = "gear_label"


@dataclass
class GearPredictor(BaseModel):
    pipeline: Pipeline | None = None
    classes_: list[str] | None = None  # e.g., ["gillnet", "handline", "longline"]

    @staticmethod
    def _make_pipeline() -> Pipeline:
        pre = ColumnTransformer([("num", StandardScaler(), FEATURES)], remainder="drop")
        clf = LogisticRegression(max_iter=1000, multi_class="auto")
        return Pipeline([("pre", pre), ("clf", clf)])

    def fit_df(self, df: pd.DataFrame) -> GearPredictor:
        X = df[FEATURES]
        y = df[LABEL_COL].astype("category")
        self.pipeline = self._make_pipeline().fit(X, y)
        self.classes_ = list(y.cat.categories.astype(str))
        logger.info("GearPredictor trained on %d rows, classes=%s", len(df), self.classes_)
        return self

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained/loaded.")
        out = df.copy()
        out["gear_pred"] = self.pipeline.predict(df[FEATURES]).astype(str)
        return out

    @classmethod
    def load_default(cls) -> GearPredictor:
        model = cls()
        model.pipeline = cls._make_pipeline()
        logger.warning(
            "Loaded default untrained GearPredictor pipeline. Call .fit_df() or load trained artifact."
        )
        return model
