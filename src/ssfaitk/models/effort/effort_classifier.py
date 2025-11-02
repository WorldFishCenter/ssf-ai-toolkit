from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from ..base import BaseModel
from ...utils.logging import get_logger

logger = get_logger(__name__)

FEATURES = ["sog", "cog"]
LABEL_COL = "is_fishing"

@dataclass
class EffortClassifier(BaseModel):
    pipeline: Pipeline | None = None

    @staticmethod
    def _make_pipeline() -> Pipeline:
        pre = ColumnTransformer([("num", StandardScaler(), FEATURES)], remainder="drop")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        return Pipeline([("pre", pre), ("clf", clf)])

    def fit_df(self, df: pd.DataFrame) -> "EffortClassifier":
        X = df[FEATURES]
        y = df[LABEL_COL].astype(int)
        self.pipeline = self._make_pipeline().fit(X, y)
        logger.info("EffortClassifier trained on %d rows", len(df))
        return self

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained/loaded.")
        out = df.copy()
        out["effort_pred"] = self.pipeline.predict(df[FEATURES]).astype(int)
        return out

    @classmethod
    def load_default(cls) -> "EffortClassifier":
        # Placeholder: in production, load a packaged artifact shipped with the library
        model = cls()
        model.pipeline = cls._make_pipeline()
        logger.warning("Loaded default untrained EffortClassifier pipeline. Call .fit_df() or load trained artifact.")
        return model
