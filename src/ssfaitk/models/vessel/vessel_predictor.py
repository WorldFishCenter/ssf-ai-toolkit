from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ...utils.logging import get_logger
from ..base import BaseModel

logger = get_logger(__name__)

FEATURES = ["duration_hours", "distance_nm", "mean_sog"]
LABEL_COL = "vessel_type_label"  # "motorized"/"non-motorized"


@dataclass
class VesselTypePredictor(BaseModel):
    pipeline: Pipeline | None = None

    @staticmethod
    def _make_pipeline() -> Pipeline:
        pre = ColumnTransformer([("num", StandardScaler(), FEATURES)], remainder="drop")
        clf = SVC(kernel="rbf", probability=True)
        return Pipeline([("pre", pre), ("clf", clf)])

    def fit_df(self, df: pd.DataFrame) -> VesselTypePredictor:
        X = df[FEATURES]
        y = df[LABEL_COL].astype(str)
        self.pipeline = self._make_pipeline().fit(X, y)
        logger.info("VesselTypePredictor trained on %d rows", len(df))
        return self

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained/loaded.")
        out = df.copy()
        out["vessel_type_pred"] = self.pipeline.predict(df[FEATURES]).astype(str)
        return out

    @classmethod
    def load_default(cls) -> VesselTypePredictor:
        model = cls()
        model.pipeline = cls._make_pipeline()
        logger.warning(
            "Loaded default untrained VesselTypePredictor pipeline. Call .fit_df() or load trained artifact."
        )
        return model
