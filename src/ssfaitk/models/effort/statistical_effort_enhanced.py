# src/ssfaitk/models/effort/statistical_effort_classifier.py
"""
Statistical Fishing Effort Classifier with Trip Phase Detection

Enhanced version that distinguishes:
1. Fishing - actual fishing activity
2. Sailing - transit/traveling
3. Starting trip - departure activities (near start point/time)
4. Ending trip - arrival activities (near end point/time)

Key enhancement: Prevents misclassifying start/end activities as fishing,
even when they show fishing-like behavior (slow speed, turning, stopping).
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
# Trip Phase Detection (NEW FEATURE)
# ===============================================================

def detect_trip_phases(
    df: pd.DataFrame,
    trip_col: str = 'trip_id',
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    time_col: str = 'timestamp',
    start_distance_km: float = 2.0,
    end_distance_km: float = 2.0,
    start_time_fraction: float = 0.15,
    end_time_fraction: float = 0.15,
) -> pd.DataFrame:
    """
    Detect trip phases: starting_trip, ending_trip, in_progress.
    
    Identifies departure and arrival phases based on:
    1. Distance from start/end points
    2. Time proximity to trip start/end
    
    Args:
        df: DataFrame with trip data
        trip_col: Trip identifier column
        lat_col: Latitude column
        lon_col: Longitude column
        time_col: Timestamp column
        start_distance_km: Distance threshold for start zone (default: 2 km)
        end_distance_km: Distance threshold for end zone (default: 2 km)
        start_time_fraction: Fraction of trip duration for start phase (default: 0.15 = 15%)
        end_time_fraction: Fraction of trip duration for end phase (default: 0.15 = 15%)
    
    Returns:
        DataFrame with added columns:
        - dist_to_start_km: Distance to trip start point
        - dist_to_end_km: Distance to trip end point
        - trip_progress: Normalized trip progress (0 to 1)
        - is_start_phase: Binary flag for start phase
        - is_end_phase: Binary flag for end phase
        - trip_phase: 'starting', 'ending', or 'in_progress'
    """
    df = df.copy()
    df = df.sort_values([trip_col, time_col])
    
    logger.info("Detecting trip phases...")
    logger.info(f"  Start zone: {start_distance_km} km or first {start_time_fraction*100:.0f}% of trip")
    logger.info(f"  End zone: {end_distance_km} km or last {end_time_fraction*100:.0f}% of trip")
    
    # Initialize columns
    df['dist_to_start_km'] = 0.0
    df['dist_to_end_km'] = 0.0
    df['trip_progress'] = 0.0
    df['is_start_phase'] = 0
    df['is_end_phase'] = 0
    df['trip_phase'] = 'in_progress'
    
    # Process each trip
    for trip_id, trip_data in df.groupby(trip_col):
        idx = trip_data.index
        
        if len(trip_data) < 3:
            # Too short to have phases
            continue
        
        # Get start and end points
        start_lat = trip_data[lat_col].iloc[0]
        start_lon = trip_data[lon_col].iloc[0]
        end_lat = trip_data[lat_col].iloc[-1]
        end_lon = trip_data[lon_col].iloc[-1]
        
        # Calculate distances to start/end points
        dist_to_start = haversine_distance(
            trip_data[lat_col].values,
            trip_data[lon_col].values,
            start_lat,
            start_lon
        )
        
        dist_to_end = haversine_distance(
            trip_data[lat_col].values,
            trip_data[lon_col].values,
            end_lat,
            end_lon
        )
        
        df.loc[idx, 'dist_to_start_km'] = dist_to_start
        df.loc[idx, 'dist_to_end_km'] = dist_to_end
        
        # Calculate trip progress (0 to 1)
        trip_progress = np.arange(len(trip_data)) / (len(trip_data) - 1)
        df.loc[idx, 'trip_progress'] = trip_progress
        
        # Detect start phase (either within distance OR within time threshold)
        start_phase_distance = dist_to_start <= start_distance_km
        start_phase_time = trip_progress <= start_time_fraction
        is_start = start_phase_distance | start_phase_time
        
        # Detect end phase (either within distance OR within time threshold)
        end_phase_distance = dist_to_end <= end_distance_km
        end_phase_time = trip_progress >= (1 - end_time_fraction)
        is_end = end_phase_distance | end_phase_time
        
        df.loc[idx, 'is_start_phase'] = is_start.astype(int)
        df.loc[idx, 'is_end_phase'] = is_end.astype(int)
        
        # Assign phase (end takes priority if overlap)
        df.loc[idx[is_start], 'trip_phase'] = 'starting'
        df.loc[idx[is_end], 'trip_phase'] = 'ending'
    
    # Log statistics
    n_starting = (df['trip_phase'] == 'starting').sum()
    n_ending = (df['trip_phase'] == 'ending').sum()
    n_progress = (df['trip_phase'] == 'in_progress').sum()
    
    logger.info(f"  Starting phase: {n_starting:,} points ({100*n_starting/len(df):.1f}%)")
    logger.info(f"  Ending phase: {n_ending:,} points ({100*n_ending/len(df):.1f}%)")
    logger.info(f"  In progress: {n_progress:,} points ({100*n_progress/len(df):.1f}%)")
    
    return df


def assign_activity_types(
    df: pd.DataFrame,
    fishing_col: str = 'is_fishing',
) -> pd.DataFrame:
    """
    Assign explicit activity types based on fishing classification and trip phase.
    
    Logic:
    - If in starting phase → 'starting_trip' (even if shows fishing behavior)
    - If in ending phase → 'ending_trip' (even if shows fishing behavior)
    - If classified as fishing and in progress → 'fishing'
    - Otherwise → 'sailing'
    
    Args:
        df: DataFrame with fishing classification and trip phase
        fishing_col: Column with binary fishing classification
    
    Returns:
        DataFrame with 'activity_type' column added
    """
    df = df.copy()
    
    # Initialize with default
    df['activity_type'] = 'sailing'
    
    # Assign based on priority (start/end phases override fishing classification)
    df.loc[df['trip_phase'] == 'starting', 'activity_type'] = 'starting_trip'
    df.loc[df['trip_phase'] == 'ending', 'activity_type'] = 'ending_trip'
    
    # Only mark as fishing if in progress phase (not start/end)
    fishing_in_progress = (df[fishing_col] == 1) & (df['trip_phase'] == 'in_progress')
    df.loc[fishing_in_progress, 'activity_type'] = 'fishing'
    
    # Log statistics
    logger.info("Activity type distribution:")
    for activity, count in df['activity_type'].value_counts().items():
        pct = 100 * count / len(df)
        logger.info(f"  {activity}: {count:,} ({pct:.1f}%)")
    
    return df


# ===============================================================
# Feature Engineering Functions (from original file)
# ===============================================================

def compute_kinematic_features(df: pd.DataFrame, trip_col: str = "trip_id") -> pd.DataFrame:
    """
    Compute kinematic features: speed, acceleration, jerk, bearings, turns.
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
    df["speed_kmh"] = df["speed_kmh"].fillna(0)

    # Acceleration (km/h²)
    df["speed_prev"] = df.groupby(trip_col)["speed_kmh"].shift(1)
    df.loc[df["dt_hours"] > 0, "acceleration"] = (
        (df.loc[df["dt_hours"] > 0, "speed_kmh"] - df.loc[df["dt_hours"] > 0, "speed_prev"])
        / df.loc[df["dt_hours"] > 0, "dt_hours"]
    )
    df["acceleration"] = df["acceleration"].fillna(0)

    # Jerk (rate of acceleration change)
    df["accel_prev"] = df.groupby(trip_col)["acceleration"].shift(1)
    df.loc[df["dt_hours"] > 0, "jerk"] = (
        (df.loc[df["dt_hours"] > 0, "acceleration"] - df.loc[df["dt_hours"] > 0, "accel_prev"])
        / df.loc[df["dt_hours"] > 0, "dt_hours"]
    )
    df["jerk"] = df["jerk"].fillna(0)

    # Bearing and turn angles
    df["longitude_next"] = df.groupby(trip_col)["longitude"].shift(-1)
    df["latitude_next"] = df.groupby(trip_col)["latitude"].shift(-1)

    df["bearing"] = calculate_bearing(
        df["latitude"], df["longitude"], df["latitude_next"], df["longitude_next"]
    )
    df["bearing_prev"] = df.groupby(trip_col)["bearing"].shift(1)
    df["turn_angle"] = turning_angle(df["bearing"], df["bearing_prev"])
    df["turn_angle"] = df["turn_angle"].fillna(0)

    return df


def compute_local_statistics(
    df: pd.DataFrame, trip_col: str = "trip_id", window: int = 10
) -> pd.DataFrame:
    """
    Compute rolling statistics in local windows.
    """
    df = df.sort_values([trip_col, "timestamp"]).copy()

    for col in ["speed_kmh", "acceleration", "turn_angle"]:
        df[f"{col}_mean"] = (
            df.groupby(trip_col)[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"{col}_std"] = (
            df.groupby(trip_col)[col].transform(lambda x: x.rolling(window, min_periods=1).std())
        )
        df[f"{col}_max"] = (
            df.groupby(trip_col)[col].transform(lambda x: x.rolling(window, min_periods=1).max())
        )
        df[f"{col}_min"] = (
            df.groupby(trip_col)[col].transform(lambda x: x.rolling(window, min_periods=1).min())
        )

    # Coefficient of variation (normalized variability)
    df["speed_cv"] = df["speed_kmh_std"] / (df["speed_kmh_mean"] + 1e-6)
    df["accel_cv"] = df["acceleration_std"] / (df["acceleration_mean"].abs() + 1e-6)

    return df


def compute_spatial_features(df: pd.DataFrame, trip_col: str = "trip_id") -> pd.DataFrame:
    """
    Compute spatial features related to trip geometry.
    """
    df = df.sort_values([trip_col, "timestamp"]).copy()

    for trip_id, trip_data in df.groupby(trip_col):
        idx = trip_data.index

        # Distance to start point
        start_lat, start_lon = trip_data["latitude"].iloc[0], trip_data["longitude"].iloc[0]
        df.loc[idx, "dist_to_start_km"] = haversine_distance(
            trip_data["latitude"].values,
            trip_data["longitude"].values,
            start_lat,
            start_lon,
        )

        # Straightness: ratio of straight-line to actual distance
        cumulative_dist = trip_data["distance_km"].cumsum()
        straight_dist = df.loc[idx, "dist_to_start_km"]
        df.loc[idx, "straightness"] = straight_dist / (cumulative_dist + 1e-6)

        # Radius of gyration
        mean_lat, mean_lon = trip_data["latitude"].mean(), trip_data["longitude"].mean()
        df.loc[idx, "radius_gyration_km"] = haversine_distance(
            trip_data["latitude"].values,
            trip_data["longitude"].values,
            mean_lat,
            mean_lon,
        ).mean()

        # Sinuosity: actual path / straight-line distance
        total_dist = cumulative_dist.iloc[-1] if len(cumulative_dist) > 0 else 0
        straight_total = haversine_distance(start_lat, start_lon, trip_data["latitude"].iloc[-1], trip_data["longitude"].iloc[-1])
        df.loc[idx, "sinuosity"] = total_dist / (straight_total + 1e-6)

    return df


def compute_temporal_features(df: pd.DataFrame, trip_col: str = "trip_id") -> pd.DataFrame:
    """
    Compute time-based features.
    """
    df = df.sort_values([trip_col, "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Hour of day
    df["hour"] = df["timestamp"].dt.hour

    # Daytime fishing (6am-6pm)
    df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] < 18)).astype(int)

    # Point position within trip
    df["point_number"] = df.groupby(trip_col).cumcount() + 1
    df["total_points"] = df.groupby(trip_col)["point_number"].transform("max")
    df["trip_position"] = df["point_number"] / df["total_points"]

    return df


# ===============================================================
# Behavioral Classification Rules
# ===============================================================

class FishingBehaviorRules:
    """
    Rule-based fishing behavior detection.
    """

    def __init__(self, config: dict | None = None):
        """Initialize with optional custom thresholds."""
        self.config = config or self._default_config()

    def _default_config(self) -> dict:
        """Default thresholds for fishing behavior."""
        return {
            "speed": {"fishing_min": 0.5, "fishing_max": 5.0, "transit_min": 8.0},
            "turning": {"high_threshold": 45.0},
            "straightness": {"low_threshold": 0.3},
            "sinuosity": {"high_threshold": 1.5},
            "speed_variability": {"cv_threshold": 0.5},
        }

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply classification rules.
        """
        df = df.copy()

        # Individual behavior indicators
        # df["is_fishing_speed"] = (df["speed_kmh"] < self.config["speed"]["fishing_max"]).astype(int)
        df["is_fishing_speed"] = (
                (df["speed_kmh_mean"] >= self.config["speed"]["fishing_min"]) &
                (df["speed_kmh_mean"] <= self.config["speed"]["fishing_max"])
        ).astype(int)
        df["is_transit_speed"] = (df["speed_kmh_mean"] > self.config["speed"]["transit_min"]).astype(int)
        df["is_high_turning"] = (df["turn_angle"] > self.config["turning"]["high_threshold"]).astype(int)
        df["is_low_straightness"] = (df["straightness"] < self.config["straightness"]["low_threshold"]).astype(int)
        df["is_high_sinuosity"] = (df["sinuosity"] > self.config["sinuosity"]["high_threshold"]).astype(int)
        df["is_clustered"] = (df["radius_gyration_km"] < 0.5).astype(int)
        df["is_variable_speed"] = (df["speed_cv"] > self.config["speed_variability"]["cv_threshold"]).astype(int)

        # Fishing score (weighted combination)
        df["fishing_score"] = (
            0.25 * df["is_fishing_speed"]
            + 0.20 * df["is_high_turning"]
            + 0.15 * df["is_low_straightness"]
            + 0.15 * df["is_high_sinuosity"]
            + 0.10 * df["is_clustered"]
            + 0.15 * df["is_variable_speed"]
        )

        # Binary classification
        df["is_fishing"] = (df["fishing_score"] >= 0.5).astype(int)

        # Override: definitely not fishing if high transit speed
        df.loc[df["is_transit_speed"] == 1, "is_fishing"] = 0

        return df


# ===============================================================
# Main Classifier (ENHANCED)
# ===============================================================

class StatisticalEffortClassifier:
    """
    Statistical fishing effort classifier with trip phase detection.
    
    ENHANCED VERSION:
    - Detects trip phases (starting, in_progress, ending)
    - Assigns explicit activity types (fishing, sailing, starting_trip, ending_trip)
    - Prevents misclassifying departure/arrival as fishing
    """

    def __init__(
        self,
        config: dict | None = None,
        start_distance_km: float = 1.0,
        end_distance_km: float = 1.0,
        start_time_fraction: float = 0.05,
        end_time_fraction: float = 0.05,
    ):
        """
        Initialize classifier with trip phase detection parameters.

        Args:
            config: Optional configuration dict for behavioral thresholds
            start_distance_km: Distance threshold for start zone (default: 2 km)
            end_distance_km: Distance threshold for end zone (default: 2 km)
            start_time_fraction: Fraction of trip for start phase (default: 0.15 = 15%)
            end_time_fraction: Fraction of trip for end phase (default: 0.15 = 15%)
        """
        self.rules = FishingBehaviorRules(config)
        self.config = self.rules.config
        
        # Trip phase detection parameters
        self.start_distance_km = start_distance_km
        self.end_distance_km = end_distance_km
        self.start_time_fraction = start_time_fraction
        self.end_time_fraction = end_time_fraction

    def predict(
        self,
        df: pd.DataFrame,
        trip_col: Optional[str] = None,
        lat_col: Optional[str] = None,
        lon_col: Optional[str] = None,
        time_col: Optional[str] = None,
        detect_trip_phase: bool = True,
    ) -> pd.DataFrame:
        """
        Predict fishing effort with trip phase detection.
        
        Returns DataFrame with:
        - is_fishing: Binary classification (1=fishing, 0=non-fishing)
        - activity_type: Explicit activity ('fishing', 'sailing', 'starting_trip', 'ending_trip')
        - fishing_score: Continuous likelihood score (0-1)
        - trip_phase: Trip phase ('starting', 'in_progress', 'ending')
        - All engineered features
        
        Args:
            df: DataFrame with GPS tracks
            trip_col: Trip identifier column (auto-detected if None)
            lat_col: Latitude column (auto-detected if None)
            lon_col: Longitude column (auto-detected if None)
            time_col: Timestamp column (auto-detected if None)
            detect_trip_phase: Enable trip phase detection (default: True)
        
        Examples:
            >>> clf = StatisticalEffortClassifier()
            >>> predictions = clf.predict(df)
            >>> 
            >>> # Check activity distribution
            >>> print(predictions['activity_type'].value_counts())
            fishing         125,432
            sailing          98,234
            starting_trip    15,678
            ending_trip      14,892
        """
        df = df.copy()
        
        # Resolve column names (auto-detect or use user-provided)
        if COLUMN_MAPPER_AVAILABLE:
            lat_col_resolved = resolve_column_name(df, 'latitude', lat_col, required=True)
            lon_col_resolved = resolve_column_name(df, 'longitude', lon_col, required=True)
            time_col_resolved = resolve_column_name(df, 'timestamp', time_col, required=True)
            
            if trip_col:
                if trip_col not in df.columns:
                    raise ValueError(f"Specified trip column '{trip_col}' not found")
                trip_col_resolved = trip_col
            else:
                trip_col_resolved = resolve_column_name(df, 'trip_id', None, required=False)
            
            logger.info(f"Resolved columns: lat={lat_col_resolved}, lon={lon_col_resolved}, "
                       f"time={time_col_resolved}, trip={trip_col_resolved}")
        else:
            lat_col_resolved = lat_col or 'latitude'
            lon_col_resolved = lon_col or 'longitude'
            time_col_resolved = time_col or 'timestamp'
            trip_col_resolved = trip_col
            
            required_cols = [lat_col_resolved, lon_col_resolved, time_col_resolved]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        # Standardize column names
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

        logger.info(f"Processing {len(df):,} points across {df['trip_id'].nunique()} trips")

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

        # Trip phase detection (NEW FEATURE)
        if detect_trip_phase:
            df = detect_trip_phases(
                df,
                trip_col='trip_id',
                lat_col='latitude',
                lon_col='longitude',
                time_col='timestamp',
                start_distance_km=self.start_distance_km,
                end_distance_km=self.end_distance_km,
                start_time_fraction=self.start_time_fraction,
                end_time_fraction=self.end_time_fraction,
            )
            
            # Assign activity types
            df = assign_activity_types(df, fishing_col='is_fishing')
        else:
            # No trip phase detection - simple activity types
            df['activity_type'] = 'sailing'
            df.loc[df['is_fishing'] == 1, 'activity_type'] = 'fishing'

        # Summary statistics
        fishing_pct = df["is_fishing"].mean() * 100
        logger.info(f"✓ Classification complete: {fishing_pct:.1f}% classified as fishing")

        return df

    def predict_trips(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Alias for predict() - more intuitive name."""
        return self.predict(df, **kwargs)

    def save_config(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return cls(config=config)
