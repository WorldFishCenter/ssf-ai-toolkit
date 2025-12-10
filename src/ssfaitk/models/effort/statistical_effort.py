# src/ssfaitk/models/effort/statistical_effort_classifier.py
"""
Statistical Fishing Effort Classifier

A rule-based classifier that uses behavioral and kinematic features to identify
fishing activity from GPS tracks. No ML training required - uses empirically
derived thresholds based on fishing vessel behavior patterns.

Key behavioral indicators:
- Low speed with high turning (searching/setting gear)
- Stationary periods (hauling/setting)
- Directional changes and course variability
- Speed variability and acceleration patterns
- Spatial clustering and path tortuosity
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Import column mapper for flexible column name handling
try:
    from ...utils.column_mapper import resolve_column_name
    COLUMN_MAPPER_AVAILABLE = True
except ImportError:
    COLUMN_MAPPER_AVAILABLE = False
    logging.warning("Column mapper not available - falling back to strict column names")

logger = logging.getLogger(__name__)


# ===============================================================
# Geometry & Kinematic Helpers
# ===============================================================


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Calculate great circle distance in kilometers."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def calculate_bearing(lat1, lon1, lat2, lon2) -> float:
    """Calculate bearing in degrees (0-360)."""
    dLon = np.radians(lon2 - lon1)
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def turning_angle(bearing1, bearing2) -> float:
    """Calculate absolute turning angle between two bearings."""
    angle = np.abs(bearing2 - bearing1)
    return np.minimum(angle, 360 - angle)


# ===============================================================
# Feature Engineering Functions
# ===============================================================


def compute_kinematic_features(df: pd.DataFrame, trip_col: str = "trip_id") -> pd.DataFrame:
    """
    Compute kinematic features: speed, acceleration, jerk, bearings, turns.

    Features computed:
    - speed_kmh: Instantaneous speed
    - acceleration: Rate of speed change
    - jerk: Rate of acceleration change
    - bearing: Course direction
    - turn_angle: Angular change in direction
    """
    df = df.sort_values([trip_col, "timestamp"]).copy()

    # Shift for previous values
    for col in ["latitude", "longitude", "timestamp"]:
        df[f"{col}_prev"] = df.groupby(trip_col)[col].shift(1)

    # Distance and time deltas
    df["distance_km"] = haversine_distance(
        df["latitude"], df["longitude"], df["latitude_prev"], df["longitude_prev"]
    )

    df["dt_hours"] = (
        pd.to_datetime(df["timestamp"]) - pd.to_datetime(df["timestamp_prev"])
    ).dt.total_seconds() / 3600.0

    # Speed (km/h)
    df.loc[df["dt_hours"] > 0, "speed_kmh"] = (
        df.loc[df["dt_hours"] > 0, "distance_km"] / df.loc[df["dt_hours"] > 0, "dt_hours"]
    )
    df["speed_kmh"] = df["speed_kmh"].fillna(0).clip(0, 50)  # Cap at reasonable vessel speed

    # Acceleration (km/h²)
    df["speed_prev"] = df.groupby(trip_col)["speed_kmh"].shift(1)
    df["acceleration"] = (df["speed_kmh"] - df["speed_prev"]) / df["dt_hours"]
    df["acceleration"] = df["acceleration"].fillna(0).clip(-30, 30)

    # Jerk (km/h³)
    df["accel_prev"] = df.groupby(trip_col)["acceleration"].shift(1)
    df["jerk"] = (df["acceleration"] - df["accel_prev"]) / df["dt_hours"]
    df["jerk"] = df["jerk"].fillna(0).clip(-100, 100)

    # Bearing and turns
    df["longitude_next"] = df.groupby(trip_col)["longitude"].shift(-1)
    df["latitude_next"] = df.groupby(trip_col)["latitude"].shift(-1)

    df["bearing"] = calculate_bearing(
        df["latitude"], df["longitude"], df["latitude_next"], df["longitude_next"]
    )
    df["bearing_prev"] = df.groupby(trip_col)["bearing"].shift(1)
    df["turn_angle"] = turning_angle(df["bearing_prev"], df["bearing"])
    df["turn_angle"] = df["turn_angle"].fillna(0)

    return df


def compute_local_statistics(
    df: pd.DataFrame, trip_col: str = "trip_id", window_minutes: float = 10.0
) -> pd.DataFrame:
    """
    Compute rolling statistics over a time window.

    Features:
    - Speed statistics (mean, std, cv)
    - Acceleration statistics
    - Turn angle statistics
    - Directional variability
    """
    df = df.sort_values([trip_col, "timestamp"]).copy()

    # Estimate window size based on median sampling interval
    def estimate_window(g):
        times = pd.to_datetime(g["timestamp"])
        if len(times) < 2:
            return 5
        dt_seconds = np.diff(times.values.astype("datetime64[s]").astype(float))
        median_dt = np.median(dt_seconds[dt_seconds > 0]) if len(dt_seconds) > 0 else 60
        return max(3, min(25, int((window_minutes * 60) / median_dt)))

    for trip_id, trip_data in df.groupby(trip_col):
        idx = trip_data.index
        win_size = estimate_window(trip_data)

        # Rolling window (centered)
        for col in ["speed_kmh", "acceleration", "turn_angle"]:
            rolling = trip_data[col].rolling(win_size, min_periods=1, center=True)
            df.loc[idx, f"{col}_mean"] = rolling.mean()
            df.loc[idx, f"{col}_std"] = rolling.std().fillna(0)
            df.loc[idx, f"{col}_max"] = rolling.max()
            df.loc[idx, f"{col}_min"] = rolling.min()

        # Coefficient of variation (CV) - indicator of erratic behavior
        df.loc[idx, "speed_cv"] = df.loc[idx, "speed_kmh_std"] / (
            df.loc[idx, "speed_kmh_mean"] + 0.1
        )
        df.loc[idx, "accel_cv"] = df.loc[idx, "acceleration_std"] / (
            abs(df.loc[idx, "acceleration_mean"]) + 0.1
        )

    return df


def compute_spatial_features(
    df: pd.DataFrame, trip_col: str = "trip_id", window_points: int = 15
) -> pd.DataFrame:
    """
    Compute spatial behavior features.

    Features:
    - Straightness index (path efficiency)
    - Radius of gyration (spatial spread)
    - Sinuosity (path tortuosity)
    - Distance to start point
    """
    df = df.sort_values([trip_col, "timestamp"]).copy()

    for trip_id, trip_data in df.groupby(trip_col):
        idx = trip_data.index
        lats = trip_data["latitude"].values
        lons = trip_data["longitude"].values
        n = len(lats)

        # Distance to trip start
        start_lat, start_lon = lats[0], lons[0]
        df.loc[idx, "dist_to_start_km"] = haversine_distance(lats, lons, start_lat, start_lon)

        # Windowed spatial metrics
        half_win = window_points // 2
        straightness = np.zeros(n)
        radius_gyration = np.zeros(n)
        sinuosity = np.zeros(n)

        for i in range(n):
            start_idx = max(0, i - half_win)
            end_idx = min(n, i + half_win + 1)

            win_lats = lats[start_idx:end_idx]
            win_lons = lons[start_idx:end_idx]

            if len(win_lats) < 2:
                straightness[i] = 1.0
                radius_gyration[i] = 0.0
                sinuosity[i] = 1.0
                continue

            # Straightness: chord/path ratio
            chord = haversine_distance(win_lats[0], win_lons[0], win_lats[-1], win_lons[-1])
            path = np.sum(
                [
                    haversine_distance(win_lats[j], win_lons[j], win_lats[j + 1], win_lons[j + 1])
                    for j in range(len(win_lats) - 1)
                ]
            )
            straightness[i] = chord / path if path > 0 else 1.0

            # Radius of gyration: spatial dispersion
            centroid_lat, centroid_lon = np.mean(win_lats), np.mean(win_lons)
            distances = haversine_distance(win_lats, win_lons, centroid_lat, centroid_lon)
            radius_gyration[i] = np.sqrt(np.mean(distances**2))

            # Sinuosity: path length / straight line distance
            sinuosity[i] = path / chord if chord > 0.001 else 1.0

        df.loc[idx, "straightness"] = straightness
        df.loc[idx, "radius_gyration_km"] = radius_gyration
        df.loc[idx, "sinuosity"] = np.clip(sinuosity, 1.0, 10.0)

    return df


def compute_temporal_features(df: pd.DataFrame, trip_col: str = "trip_id") -> pd.DataFrame:
    """
    Compute temporal features: time of day, position in trip.

    Features:
    - hour: Hour of day (0-23)
    - is_daytime: Binary daytime indicator
    - trip_position: Normalized position within trip (0-1)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)

    # Position within trip
    df["point_number"] = df.groupby(trip_col).cumcount()
    df["total_points"] = df.groupby(trip_col)["point_number"].transform("max") + 1
    df["trip_position"] = df["point_number"] / df["total_points"]

    return df


# ===============================================================
# Fishing Behavior Rules
# ===============================================================


class FishingBehaviorRules:
    """Rule-based fishing behavior detection using empirical thresholds."""

    def __init__(self, config: dict | None = None):
        """Initialize with default or custom configuration."""
        self.config = self._default_config()
        if config:
            self.config.update(config)

    @staticmethod
    def _default_config() -> dict:
        """Default behavioral thresholds calibrated for small-scale fisheries."""
        return {
            # Speed thresholds (km/h)
            "min_fishing_speed": 0.5,
            "max_fishing_speed": 8.0,
            "min_transit_speed": 12.0,
            # Turning behavior
            "high_turn_threshold": 45.0,  # degrees
            # Path characteristics
            "low_straightness_threshold": 0.4,
            "high_sinuosity_threshold": 1.5,
            # Spatial clustering
            "clustering_radius_km": 0.5,
            # Speed variability
            "high_speed_cv_threshold": 0.6,
            # Weights for combined score
            "weight_speed": 3.0,
            "weight_turning": 2.0,
            "weight_straightness": 2.0,
            "weight_sinuosity": 1.5,
            "weight_clustering": 2.0,
            "weight_speed_variability": 1.5,
            # Classification threshold
            "fishing_score_threshold": 0.5,
        }

    def compute_fishing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute binary fishing behavior indicators."""
        cfg = self.config

        # 1. Speed-based indicators
        df["is_fishing_speed"] = (
            (df["speed_kmh_mean"] >= cfg["min_fishing_speed"])
            & (df["speed_kmh_mean"] <= cfg["max_fishing_speed"])
        ).astype(int)

        df["is_transit_speed"] = (df["speed_kmh_mean"] >= cfg["min_transit_speed"]).astype(int)

        # 2. Turning behavior
        df["is_high_turning"] = (df["turn_angle_mean"] >= cfg["high_turn_threshold"]).astype(int)

        # 3. Path characteristics
        df["is_low_straightness"] = (df["straightness"] <= cfg["low_straightness_threshold"]).astype(
            int
        )

        df["is_high_sinuosity"] = (df["sinuosity"] >= cfg["high_sinuosity_threshold"]).astype(int)

        df["is_clustered"] = (df["radius_gyration_km"] <= cfg["clustering_radius_km"]).astype(int)

        # 4. Speed variability
        df["is_variable_speed"] = (df["speed_cv"] >= cfg["high_speed_cv_threshold"]).astype(int)

        return df

    def compute_fishing_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute weighted fishing likelihood score (0-1).

        Combines multiple indicators with empirical weights.
        """
        cfg = self.config

        # Weighted sum of indicators
        fishing_score = (
            cfg["weight_speed"] * df["is_fishing_speed"]
            + cfg["weight_turning"] * df["is_high_turning"]
            + cfg["weight_straightness"] * df["is_low_straightness"]
            + cfg["weight_sinuosity"] * df["is_high_sinuosity"]
            + cfg["weight_clustering"] * df["is_clustered"]
            + cfg["weight_speed_variability"] * df["is_variable_speed"]
        )

        # Penalty for transit speed (strong negative indicator)
        fishing_score -= 5.0 * df["is_transit_speed"]

        # Normalize to 0-1
        total_weight = (
            cfg["weight_speed"]
            + cfg["weight_turning"]
            + cfg["weight_straightness"]
            + cfg["weight_sinuosity"]
            + cfg["weight_clustering"]
            + cfg["weight_speed_variability"]
        )

        df["fishing_score"] = (fishing_score / total_weight).clip(0, 1)

        return df

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Binary classification: fishing (1) vs non-fishing (0).
        """
        df = self.compute_fishing_indicators(df)
        df = self.compute_fishing_score(df)

        df["is_fishing"] = (df["fishing_score"] >= self.config["fishing_score_threshold"]).astype(
            int
        )

        return df


# ===============================================================
# Main Statistical Classifier
# ===============================================================


class StatisticalEffortClassifier:
    """
    Statistical fishing effort classifier.

    Uses rule-based methods to identify fishing activity from GPS tracks
    based on kinematic and behavioral patterns.

    No machine learning training required.
    
    Column names are automatically detected if not provided. Supports
    common variations like 'latitude', 'Latitude', 'lat', 'LAT', etc.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize classifier.

        Args:
            config: Optional configuration dict for behavioral thresholds
        """
        self.rules = FishingBehaviorRules(config)
        self.config = self.rules.config

    def predict(
        self,
        df: pd.DataFrame,
        trip_col: Optional[str] = None,
        lat_col: Optional[str] = None,
        lon_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Predict fishing effort for GPS tracks.
        
        Column names are automatically detected if not provided. Searches for
        common variations:
        - Latitude: 'latitude', 'Latitude', 'lat', 'Lat', 'LAT', etc.
        - Longitude: 'longitude', 'Longitude', 'lon', 'Lon', 'LON', etc.
        - Timestamp: 'timestamp', 'time', 'datetime', 'ltime', etc.
        - Trip ID: 'trip_id', 'Trip_ID', 'TRIP_ID', 'voyage_id', etc.

        Args:
            df: DataFrame with GPS tracks
            trip_col: Column name for trip identifier (auto-detected if None)
            lat_col: Column name for latitude (auto-detected if None)
            lon_col: Column name for longitude (auto-detected if None)
            time_col: Column name for timestamp (auto-detected if None)

        Returns:
            DataFrame with predictions and features added:
            - is_fishing: Binary classification (1=fishing, 0=non-fishing)
            - fishing_score: Continuous likelihood score (0-1)
            - All engineered features

        Examples:
            # Auto-detect all columns (works with any naming)
            >>> clf = StatisticalEffortClassifier()
            >>> predictions = clf.predict(df)
            
            # Specify custom column names
            >>> predictions = clf.predict(
            ...     df,
            ...     lat_col='my_latitude',
            ...     lon_col='my_longitude'
            ... )
            
            # Works with various column namings
            >>> df1 = pd.DataFrame({'Latitude': ..., 'Longitude': ...})
            >>> df2 = pd.DataFrame({'lat': ..., 'lon': ...})
            >>> df3 = pd.DataFrame({'LAT': ..., 'LON': ...})
            >>> # All work with the same call:
            >>> clf.predict(df1)
            >>> clf.predict(df2)
            >>> clf.predict(df3)
        
        Raises:
            ValueError: If required columns cannot be found in DataFrame
        """
        df = df.copy()
        
        # Resolve column names (auto-detect or use user-provided)
        if COLUMN_MAPPER_AVAILABLE:
            # Use smart column detection
            lat_col_resolved = resolve_column_name(df, 'latitude', lat_col, required=True)
            lon_col_resolved = resolve_column_name(df, 'longitude', lon_col, required=True)
            time_col_resolved = resolve_column_name(df, 'timestamp', time_col, required=True)
            
            # Trip column is optional
            if trip_col:
                # User specified trip column
                if trip_col not in df.columns:
                    raise ValueError(
                        f"Specified trip column '{trip_col}' not found. "
                        f"Available: {df.columns.tolist()}"
                    )
                trip_col_resolved = trip_col
            else:
                # Try to auto-detect
                trip_col_resolved = resolve_column_name(df, 'trip_id', None, required=False)
            
            logger.info(f"Resolved columns: lat={lat_col_resolved}, lon={lon_col_resolved}, "
                       f"time={time_col_resolved}, trip={trip_col_resolved}")
        else:
            # Fall back to strict names or user-provided
            lat_col_resolved = lat_col or 'latitude'
            lon_col_resolved = lon_col or 'longitude'
            time_col_resolved = time_col or 'timestamp'
            trip_col_resolved = trip_col
            
            # Validate required columns exist
            required_cols = [lat_col_resolved, lon_col_resolved, time_col_resolved]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing required columns: {missing}. "
                    f"Available columns: {df.columns.tolist()}"
                )

        # Standardize column names for internal processing
        rename_dict = {
            lat_col_resolved: 'latitude',
            lon_col_resolved: 'longitude',
            time_col_resolved: 'timestamp'
        }
        
        df = df.rename(columns=rename_dict)

        # Handle trip_id
        if trip_col_resolved and trip_col_resolved in df.columns:
            df = df.rename(columns={trip_col_resolved: 'trip_id'})
        elif 'trip_id' not in df.columns:
            logger.info("No trip_id provided - treating all data as single trip")
            df['trip_id'] = 1

        logger.info(f"Processing {len(df)} points across {df['trip_id'].nunique()} trips")

        # Feature engineering pipeline
        logger.info("Computing kinematic features...")
        df = compute_kinematic_features(df, "trip_id")

        logger.info("Computing local statistics...")
        df = compute_local_statistics(df, "trip_id")

        logger.info("Computing spatial features...")
        df = compute_spatial_features(df, "trip_id")

        logger.info("Computing temporal features...")
        df = compute_temporal_features(df, "trip_id")

        # Apply classification rules
        logger.info("Applying statistical classification rules...")
        df = self.rules.classify(df)

        # Summary statistics
        fishing_pct = df["is_fishing"].mean() * 100
        logger.info(f"Classification complete: {fishing_pct:.1f}% classified as fishing")

        return df

    def predict_trips(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Alias for predict() - more intuitive name."""
        return self.predict(df, **kwargs)

    def get_trip_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate predictions to trip level.

        Returns:
            DataFrame with one row per trip
        """
        if "trip_id" not in df.columns or "is_fishing" not in df.columns:
            raise ValueError("Must run predict() before getting trip summary")

        summary = (
            df.groupby("trip_id")
            .agg(
                {
                    "is_fishing": ["count", "sum", "mean"],
                    "fishing_score": ["mean", "std"],
                    "speed_kmh": ["mean", "max"],
                    "timestamp": ["min", "max"],
                }
            )
            .round(4)
        )

        summary.columns = [
            "total_points",
            "fishing_points",
            "fishing_ratio",
            "avg_fishing_score",
            "std_fishing_score",
            "avg_speed_kmh",
            "max_speed_kmh",
            "start_time",
            "end_time",
        ]

        summary["duration_hours"] = (
            pd.to_datetime(summary["end_time"]) - pd.to_datetime(summary["start_time"])
        ).dt.total_seconds() / 3600.0

        return summary.reset_index()

    def save_config(self, filepath: str):
        """Save configuration thresholds to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load_config(cls, filepath: str) -> StatisticalEffortClassifier:
        """Load classifier with custom configuration."""
        with open(filepath) as f:
            config = json.load(f)
        return cls(config=config)


# ===============================================================
# Convenience function
# ===============================================================


def predict_fishing_effort(
    df: pd.DataFrame,
    trip_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    time_col: Optional[str] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Convenience function for one-shot prediction.
    
    Column names are automatically detected if not provided.
    
    Args:
        df: DataFrame with GPS tracks
        trip_col: Trip identifier column (auto-detected if None)
        lat_col: Latitude column (auto-detected if None)
        lon_col: Longitude column (auto-detected if None)
        time_col: Timestamp column (auto-detected if None)
        config: Optional custom configuration dict
    
    Returns:
        DataFrame with predictions added
    
    Examples:
        # Auto-detect all columns
        >>> df = pd.read_csv('tracks.csv')
        >>> predictions = predict_fishing_effort(df)
        >>> print(predictions['is_fishing'].value_counts())
        
        # Specify custom columns
        >>> predictions = predict_fishing_effort(
        ...     df,
        ...     lat_col='my_lat',
        ...     lon_col='my_lon'
        ... )
    """
    clf = StatisticalEffortClassifier(config=config)
    
    return clf.predict(
        df,
        trip_col=trip_col,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col
    )
