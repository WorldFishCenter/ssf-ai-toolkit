# src/ssfaitk/models/effort/effort_classifier.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ...utils.logger import get_logger
from ..base import BaseModel

logger = get_logger(__name__)

# Import column mapper for flexible column name handling
try:
    from ...utils.column_mapper import resolve_column_name
    COLUMN_MAPPER_AVAILABLE = True
except ImportError:
    COLUMN_MAPPER_AVAILABLE = False
    logger.warning("Column mapper not available - falling back to legacy column detection")


# ------------------------------
# Legacy column helpers (fallback)
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
    """Legacy: Find first column present from options list."""
    for c in options:
        if c in df.columns:
            return c
    return None


def _require(colname: str | None, which: str) -> str:
    """Legacy: Require column or raise error."""
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


def _resolve_columns(
    df: pd.DataFrame,
    trip_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    time_col: Optional[str] = None,
    altitude_col: Optional[str] = None,
    model_col: Optional[str] = None,
) -> dict[str, str]:
    """
    Resolve column names using column_mapper or fallback to legacy method.
    
    Returns:
        Dict mapping standard names to actual column names
    """
    if COLUMN_MAPPER_AVAILABLE:
        # Use smart column detection
        resolved = {
            'trip_id': resolve_column_name(df, 'trip_id', trip_col, required=True),
            'lat': resolve_column_name(df, 'latitude', lat_col, required=True),
            'lon': resolve_column_name(df, 'longitude', lon_col, required=True),
            'time': resolve_column_name(df, 'timestamp', time_col, required=True),
            'altitude': resolve_column_name(df, 'altitude', altitude_col, required=False),
            'model': resolve_column_name(df, 'device_model', model_col, required=False),
        }
        
        logger.debug(f"Resolved columns: {resolved}")
        return resolved
    else:
        # Fall back to legacy method
        trip = _require(
            trip_col if trip_col else _first_present(df, _CANONICAL["trip_id"]), 
            "trip_id"
        )
        time = _require(
            time_col if time_col else _first_present(df, _CANONICAL["time"]), 
            "time"
        )
        lat = _require(
            lat_col if lat_col else _first_present(df, _CANONICAL["lat"]), 
            "lat"
        )
        lon = _require(
            lon_col if lon_col else _first_present(df, _CANONICAL["lon"]), 
            "lon"
        )
        alt = altitude_col if altitude_col else _first_present(df, _CANONICAL["altitude"])
        mdl = model_col if model_col else _first_present(df, _CANONICAL["model"])
        
        return {
            'trip_id': trip,
            'lat': lat,
            'lon': lon,
            'time': time,
            'altitude': alt or 'altitude',
            'model': mdl or 'model',
        }


def _add_features(
    df: pd.DataFrame,
    win: int = 5,
    speed_clip: float = 60.0,
    accel_clip: float = 30.0,
    trip_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    time_col: Optional[str] = None,
    altitude_col: Optional[str] = None,
    model_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Builds per-point spatiotemporal features (speed/accel/turning/rolling stats,
    straightness, radius-of-gyration, trip context) with robust NaN handling.
    
    Column names are automatically detected if not provided. Supports common
    variations like 'latitude', 'Latitude', 'lat', 'LAT', etc.
    
    Args:
        df: DataFrame with GPS tracks
        win: Rolling window size for statistics
        speed_clip: Maximum speed in km/h
        accel_clip: Maximum acceleration clip value
        trip_col: Trip ID column name (auto-detected if None)
        lat_col: Latitude column name (auto-detected if None)
        lon_col: Longitude column name (auto-detected if None)
        time_col: Timestamp column name (auto-detected if None)
        altitude_col: Altitude column name (optional, auto-detected if None)
        model_col: Device model column name (optional, auto-detected if None)
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Resolve column names (auto-detect or use provided)
    colmap = _resolve_columns(
        df, 
        trip_col=trip_col,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col,
        altitude_col=altitude_col,
        model_col=model_col
    )

    # Normalize columns to expected names for feature logic
    df = df.rename(
        columns={
            colmap['trip_id']: "Trip_ID",
            colmap['time']: "ltime",
            colmap['lat']: "Latitude",
            colmap['lon']: "Longitude",
        }
    )
    
    # Handle optional columns
    if colmap['altitude'] and colmap['altitude'] in df.columns:
        df = df.rename(columns={colmap['altitude']: "altitude"})
    if colmap['model'] and colmap['model'] in df.columns:
        df = df.rename(columns={colmap['model']: "model"})

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
    
    Automatically detects column names if not provided. Supports common variations:
    - Trip ID: 'trip_id', 'Trip_ID', 'TRIP_ID', 'voyage_id'
    - Latitude: 'latitude', 'Latitude', 'lat', 'LAT'
    - Longitude: 'longitude', 'Longitude', 'lon', 'LON'
    - Timestamp: 'timestamp', 'time', 'ltime', 'datetime'
    
    Methods:
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
        trip_col: Optional[str] = None,
        lat_col: Optional[str] = None,
        lon_col: Optional[str] = None,
        time_col: Optional[str] = None,
        feature_kwargs: Optional[dict] = None,
    ) -> EffortClassifier:
        """
        Train on per-point labeled data.
        
        Column names are automatically detected if not provided.
        
        Args:
            df: DataFrame with GPS tracks and labels
            label_col: Column with activity labels ('Fishing', 'Searching', etc.)
            trip_col: Trip ID column (auto-detected if None)
            lat_col: Latitude column (auto-detected if None)
            lon_col: Longitude column (auto-detected if None)
            time_col: Timestamp column (auto-detected if None)
            feature_kwargs: Additional arguments for feature engineering
        
        Returns:
            Self (trained classifier)
        
        Examples:
            # Auto-detect all columns
            >>> clf = EffortClassifier()
            >>> clf.fit_df(training_data)
            
            # Specify custom columns
            >>> clf.fit_df(
            ...     training_data,
            ...     lat_col='my_lat',
            ...     lon_col='my_lon'
            ... )
        """
        feature_kwargs = feature_kwargs or {}
        
        # Add column parameters to feature_kwargs
        feature_kwargs.update({
            'trip_col': trip_col,
            'lat_col': lat_col,
            'lon_col': lon_col,
            'time_col': time_col,
        })
        
        df_feat = _add_features(df, **feature_kwargs)

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
        trip_col: Optional[str] = None,
        lat_col: Optional[str] = None,
        lon_col: Optional[str] = None,
        time_col: Optional[str] = None,
        feature_kwargs: Optional[dict] = None,
        return_proba: bool = True,
    ) -> pd.DataFrame:
        """
        Predict on raw points; returns dataframe with 'effort_pred' (0/1) and 'effort_prob'.
        
        Column names are automatically detected if not provided. Supports common
        variations like 'latitude', 'Latitude', 'lat', 'LAT', etc.
        
        Args:
            df: DataFrame with GPS tracks
            trip_col: Trip ID column (auto-detected if None)
            lat_col: Latitude column (auto-detected if None)
            lon_col: Longitude column (auto-detected if None)
            time_col: Timestamp column (auto-detected if None)
            feature_kwargs: Additional arguments for feature engineering
            return_proba: Whether to return probability scores
        
        Returns:
            DataFrame with predictions:
            - effort_pred: Binary classification (1=fishing, 0=non-fishing)
            - effort_prob: Fishing probability (0-1) if return_proba=True
        
        Examples:
            # Auto-detect all columns
            >>> clf = EffortClassifier.load_trained("rf")
            >>> predictions = clf.predict_df(test_data)
            
            # Specify custom columns
            >>> predictions = clf.predict_df(
            ...     test_data,
            ...     lat_col='my_latitude',
            ...     lon_col='my_longitude'
            ... )
            
            # Works with any naming convention
            >>> df1 = pd.DataFrame({'Latitude': ..., 'Longitude': ...})
            >>> df2 = pd.DataFrame({'lat': ..., 'lon': ...})
            >>> df3 = pd.DataFrame({'LAT': ..., 'LON': ...})
            >>> # All work:
            >>> clf.predict_df(df1)
            >>> clf.predict_df(df2)
            >>> clf.predict_df(df3)
        """
        feature_kwargs = feature_kwargs or {}
        if self.pipeline is None:
            raise RuntimeError("Model is not trained/loaded.")

        # Add column parameters to feature_kwargs
        feature_kwargs.update({
            'trip_col': trip_col,
            'lat_col': lat_col,
            'lon_col': lon_col,
            'time_col': time_col,
        })

        if not set(self.feat_cols).issubset(df.columns):
            logger.info("Computing features...")
            df = _add_features(df, **feature_kwargs)

        X = df[self.feat_cols].astype("float32")
        pred_int = self.pipeline.predict(X).astype(int)
        if return_proba and hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            proba = self.pipeline.predict_proba(X)[:, 1]
        else:
            # Fall back to 0/1 as probability proxy if model lacks predict_proba
            proba = pred_int.astype(float)

        df["effort_pred"] = pred_int
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
        """
        Load a pre-trained model from artifacts directory.
        
        Args:
            model_name: Name of the model to load (default: 'rf')
        
        Returns:
            EffortClassifier with loaded model
        
        Example:
            >>> clf = EffortClassifier.load_trained("rf")
            >>> predictions = clf.predict_df(data)
        """
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
