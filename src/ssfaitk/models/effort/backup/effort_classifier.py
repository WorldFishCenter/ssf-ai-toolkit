# src/ssfaitk/models/effort/effort_classifier.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ...utils.logger import get_logger
from ..base import BaseModel

logger = get_logger(__name__)


# ------------------------------
# Column helpers
# ------------------------------
_CANONICAL = dict(
    trip_id=["Trip_ID", "trip_id", "TRIP_ID"],
    time=["ltime", "timestamp", "time", "TIME"],
    lat=["Latitude", "lat", "LATITUDE"],
    lon=["Longitude", "lon", "LONGITUDE"],
    altitude=["altitude", "Altitude"],
    model=["model", "device_model", "Model"],
)


def _first_present(df: pd.DataFrame, options: Iterable[str]) -> None:
    for c in options:
        if c in df.columns:
            return c
    return None


def _require(colname: str | None, which: str) -> str:
    if not colname:
        raise ValueError(
            f"Required column for '{which}' not found. " f"Expected one of {_CANONICAL[which]}"
        )
    return colname


# ------------------------------
# Geometry & feature functions
# ------------------------------
def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    b = np.degrees(np.arctan2(x, y))
    return (b + 360) % 360


def _rolling_centered(g: pd.Series, win: int, fn) -> pd.Series:
    roll = g.rolling(win, min_periods=1, center=True)
    return roll.apply(fn) if callable(fn) else getattr(roll, fn)()


def _straightness_window(lat_win: np.ndarray, lon_win: np.ndarray) -> float:
    if len(lat_win) < 2:
        return 1.0
    chord = _haversine(lat_win[0], lon_win[0], lat_win[-1], lon_win[-1])
    path = np.nansum(_haversine(lat_win[:-1], lon_win[:-1], lat_win[1:], lon_win[1:]))
    if not np.isfinite(path) or path == 0:
        return 1.0
    return float(chord / path)


def _rog_window(lat_win: np.ndarray, lon_win: np.ndarray) -> float:
    if len(lat_win) == 0:
        return 0.0
    latc, lonc = np.nanmean(lat_win), np.nanmean(lon_win)
    d = _haversine(lat_win, lon_win, latc, lonc)
    return float(np.sqrt(np.nanmean(d**2)))


def _shape_windows_for_trip(g: pd.DataFrame, win: int) -> pd.DataFrame:
    lat = g["Latitude"].to_numpy()
    lon = g["Longitude"].to_numpy()
    n = len(g)
    h = win // 2
    straight = np.empty(n, dtype=float)
    rog = np.empty(n, dtype=float)
    for i in range(n):
        s = max(0, i - h)
        e = min(n, i + h + 1)
        lat_w = lat[s:e]
        lon_w = lon[s:e]
        straight[i] = _straightness_window(lat_w, lon_w)
        rog[i] = _rog_window(lat_w, lon_w)
    return pd.DataFrame({"straightness_w": straight, "rog_w": rog}, index=g.index)


def _add_features(
    df: pd.DataFrame,
    win: int = 5,
    speed_clip: float = 60.0,
    accel_clip: float = 30.0,
    colmap: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Builds per-point spatiotemporal features (speed/accel/turning/rolling stats,
    straightness, radius-of-gyration, trip context) with robust NaN handling.
    """
    df = df.copy()

    # Map/require columns
    trip_col = _require(
        colmap.get("trip_id") if colmap else _first_present(df, _CANONICAL["trip_id"]), "trip_id"
    )
    time_col = _require(
        colmap.get("time") if colmap else _first_present(df, _CANONICAL["time"]), "time"
    )
    lat_col = _require(
        colmap.get("lat") if colmap else _first_present(df, _CANONICAL["lat"]), "lat"
    )
    lon_col = _require(
        colmap.get("lon") if colmap else _first_present(df, _CANONICAL["lon"]), "lon"
    )
    alt_col = (
        colmap.get("altitude") if colmap else _first_present(df, _CANONICAL["altitude"])
    ) or "altitude"
    mdl_col = (
        colmap.get("model") if colmap else _first_present(df, _CANONICAL["model"])
    ) or "model"

    # Normalize columns expected by feature logic
    df = df.rename(
        columns={
            trip_col: "Trip_ID",
            time_col: "ltime",
            lat_col: "Latitude",
            lon_col: "Longitude",
            alt_col: "altitude",
            mdl_col: "model",
        }
    )

    # Types & basic NA handling
    df["ltime"] = pd.to_datetime(df["ltime"], errors="coerce")
    if "altitude" in df.columns:
        df["altitude"] = df["altitude"].fillna(df["altitude"].median())
    if "model" in df.columns:
        df["model"] = df["model"].fillna("Unknown")

    # Label-encode model (if present)
    try:
        from sklearn.preprocessing import LabelEncoder

        df["model_enc"] = LabelEncoder().fit_transform(df.get("model", "Unknown"))
    except Exception:
        df["model_enc"] = 0

    # Sort/group
    df = df.sort_values(["Trip_ID", "ltime"]).reset_index(drop=True)

    # Lags/leads
    for col in ["Latitude", "Longitude", "ltime"]:
        df[f"{col}_prev"] = df.groupby("Trip_ID")[col].shift(1)
        df[f"{col}_next"] = df.groupby("Trip_ID")[col].shift(-1)

    # Distances/time/speed/accel/jerk
    df["dist_km"] = _haversine(
        df["Latitude"], df["Longitude"], df["Latitude_prev"], df["Longitude_prev"]
    )
    dt = (
        pd.to_datetime(df["ltime"]) - pd.to_datetime(df["ltime_prev"])
    ).dt.total_seconds() / 3600.0
    dt[dt <= 0] = np.nan
    df["dt_hr"] = dt
    df["speed_kmh"] = df["dist_km"] / df["dt_hr"]

    df["speed_prev"] = df.groupby("Trip_ID")["speed_kmh"].shift(1)
    df["acceleration"] = (df["speed_kmh"] - df["speed_prev"]) / df["dt_hr"]
    df["accel_prev"] = df.groupby("Trip_ID")["acceleration"].shift(1)
    df["jerk"] = (df["acceleration"] - df["accel_prev"]) / df["dt_hr"]

    # Bearings/turning
    df["bearing"] = _bearing_deg(
        df["Latitude"], df["Longitude"], df["Latitude_next"], df["Longitude_next"]
    )
    df["bearing_prev"] = df.groupby("Trip_ID")["bearing"].shift(1)
    da = df["bearing"] - df["bearing_prev"]
    da = ((da + 180) % 360) - 180
    df["turn_angle"] = da.abs()

    # Rolling window stats (centered)
    for c in ["speed_kmh", "acceleration", "turn_angle"]:
        df[f"{c}_mean_w"] = df.groupby("Trip_ID")[c].transform(
            lambda x: x.rolling(win, min_periods=1, center=True).mean()
        )
        df[f"{c}_std_w"] = df.groupby("Trip_ID")[c].transform(
            lambda x: x.rolling(win, min_periods=1, center=True).std()
        )
        df[f"{c}_q25_w"] = df.groupby("Trip_ID")[c].transform(
            lambda x: x.rolling(win, min_periods=1, center=True).quantile(0.25)
        )
        df[f"{c}_q75_w"] = df.groupby("Trip_ID")[c].transform(
            lambda x: x.rolling(win, min_periods=1, center=True).quantile(0.75)
        )

    # Spatial spread
    df["lat_std_w"] = df.groupby("Trip_ID")["Latitude"].transform(
        lambda x: x.rolling(win, min_periods=1, center=True).std()
    )
    df["lon_std_w"] = df.groupby("Trip_ID")["Longitude"].transform(
        lambda x: x.rolling(win, min_periods=1, center=True).std()
    )

    # Trip context
    df["point_num"] = df.groupby("Trip_ID").cumcount()
    df["trip_size"] = df.groupby("Trip_ID")["Trip_ID"].transform("count")
    df["trip_pos_norm"] = df["point_num"] / df["trip_size"]
    df["t_curr"] = pd.to_datetime(df["ltime"])
    df["time_since_start_min"] = (
        df["t_curr"] - df.groupby("Trip_ID")["t_curr"].transform("min")
    ).dt.total_seconds() / 60.0
    df["hour"] = df["t_curr"].dt.hour
    df["dayofweek"] = df["t_curr"].dt.dayofweek

    # Trajectory shape in window (centered)
    shape = df.groupby("Trip_ID", group_keys=False).apply(lambda g: _shape_windows_for_trip(g, win))
    df[["straightness_w", "rog_w"]] = shape

    # Stationarity proxy
    df["speed_mean_w"] = df.groupby("Trip_ID")["speed_kmh"].transform(
        lambda x: x.rolling(win, min_periods=1, center=True).mean()
    )
    df["is_stationary"] = (df["speed_mean_w"] < 2).astype(int)

    # Distance to trip start
    df["start_lat"] = df.groupby("Trip_ID")["Latitude"].transform("first")
    df["start_lon"] = df.groupby("Trip_ID")["Longitude"].transform("first")
    df["dist_to_start_km"] = _haversine(
        df["Latitude"], df["Longitude"], df["start_lat"], df["start_lon"]
    )

    # Cleanup/clipping
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in ["speed_kmh", "acceleration", "jerk"]:
        df[c] = df[c].fillna(0.0)
    df = df.fillna(0.0)
    df["speed_kmh"] = df["speed_kmh"].clip(0, speed_clip)
    df["acceleration"] = df["acceleration"].clip(-accel_clip, accel_clip)
    df["jerk"] = df["jerk"].clip(-3 * accel_clip, 3 * accel_clip)

    return df


# ------------------------------
# EffortClassifier
# ------------------------------
# Feature set distilled from the RingNet notebook for the binary task
_FEAT_COLS: list[str] = [
    "Latitude",
    "Longitude",  # "altitude", "model_enc",
    "hour",
    "dayofweek",
    "trip_pos_norm",
    "time_since_start_min",
    "speed_kmh",
    "acceleration",
    "jerk",
    "speed_kmh_mean_w",
    "speed_kmh_std_w",
    "speed_kmh_q25_w",
    "speed_kmh_q75_w",
    "acceleration_mean_w",
    "acceleration_std_w",
    "acceleration_q25_w",
    "acceleration_q75_w",
    "turn_angle",
    "turn_angle_mean_w",
    "turn_angle_std_w",
    "turn_angle_q25_w",
    "turn_angle_q75_w",
    "lat_std_w",
    "lon_std_w",
    "straightness_w",
    "rog_w",
    "is_stationary",
    "dist_to_start_km",
]


# Label mapping: (Fishing|Searching)=1 vs (Sailing|Traveling|other)=0
def _activity_to_binary(s: pd.Series) -> pd.Series:
    return (
        s.replace({"Fishing": 1, "Searching": 1, "Sailing": 0, "Traveling": 0})
        .fillna(0)
        .astype(int)
    )


@dataclass
class EffortClassifier(BaseModel):
    """
    Binary fishing vs non-fishing classifier with engineered spatiotemporal features.
    - fit_df: trains on a labeled dataframe
    - predict_df: returns predictions (+ optional per-trip median smoothing)
    """

    pipeline: Pipeline | None = None
    smoothing_k: int = 5  # median filter window (per-trip). Set 0/1 to disable.

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.feat_cols = _FEAT_COLS

    @staticmethod
    def _make_pipeline() -> Pipeline:
        """Simplified pipeline (no preprocessing, model operates directly on feature matrix)."""
        clf = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42)
        return Pipeline([("clf", clf)])

    # Public API -----------------
    def fit_df(
        self,
        df: pd.DataFrame,
        label_col: str = "Activity",
        colmap: dict[str, str] | None = None,
        feature_kwargs: dict[str, any] | None = None,
    ) -> EffortClassifier:
        """Train on per-point labeled data."""
        feature_kwargs = feature_kwargs or {}
        df_feat = _add_features(df, **feature_kwargs, colmap=colmap)

        if label_col not in df_feat.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataframe.")
        y = _activity_to_binary(df_feat[label_col])

        X = df_feat[_FEAT_COLS].astype("float32")
        self.pipeline = self._make_pipeline().fit(X, y)
        logger.info(
            "EffortClassifier trained on %d rows (binary fishing vs non-fishing).", len(df_feat)
        )
        return self

    def predict_df(
        self,
        df: pd.DataFrame,
        feature_kwargs: dict[str, any] | None = None,
        return_proba: bool = True,
    ) -> pd.DataFrame:
        """Predict on raw points; returns dataframe with 'effort_pred' (0/1) and 'effort_prob'."""
        feature_kwargs = feature_kwargs or {}
        if self.pipeline is None:
            raise RuntimeError("Model is not trained/loaded.")

        if not set(self.feat_cols).issubset(df.columns):
            print("computing features..")
            df = _add_features(df, **feature_kwargs)

        X = df[self.feat_cols].astype("float32")
        pred_int = self.pipeline.predict(X).astype(int)
        if return_proba and hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            proba = self.pipeline.predict_proba(X)[:, 1]
        else:
            # Fall back to 0/1 as probability proxy if model lacks predict_proba
            proba = pred_int.astype(float)

        df["effort_pred"] = self.pipeline.predict(X)
        if return_proba:
            df["effort_prob"] = proba
        return df

    # Artifacts ------------------
    @classmethod
    def load_default(cls) -> EffortClassifier:
        """
        Returns an *untrained* pipeline by default (so it can be fit immediately).
        Replace this to load a shipped artifact when available.
        """
        model = cls()
        model.pipeline = cls._make_pipeline()
        logger.warning(
            "Loaded default untrained EffortClassifier pipeline. "
            "Call .fit_df() or .load(<artifact>) for real predictions."
        )
        return model

    @classmethod
    def load_trained(cls, model_name="rf"):
        path = (
            Path(__file__).resolve().parent / "artifacts" / f"effort_classifier_{model_name}.joblib"
        )
        if not path.exists():
            raise FileNotFoundError(f"No model found at {path}")

        artifact = joblib.load(path)
        model = cls()

        # Handle both dictionary and raw model formats
        if isinstance(artifact, dict):
            model.pipeline = Pipeline([("clf", artifact["model"])])
            model.feat_cols = artifact.get("feat_cols", _FEAT_COLS)
        else:
            # Legacy format: direct model object
            model.pipeline = Pipeline([("clf", artifact)])
            model.feat_cols = _FEAT_COLS

        logger.info(f"Loaded trained EffortClassifier with {len(model.feat_cols)} features")
        return model
