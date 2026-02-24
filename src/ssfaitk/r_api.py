"""
R-friendly API wrapper for SSF AI Toolkit.

This module provides simplified function interfaces for R users via reticulate.
All functions accept R data.frames (automatically converted to pandas DataFrames)
and return pandas DataFrames (automatically converted back to R data.frames).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .models.effort import EffortClassifier, StatisticalEffortClassifier
from .models.gear import GearPredictor
from .models.vessel import VesselTypePredictor
from .utils.logger import get_logger
from .utils.hexaGrid_pipeline import run_hex_aggregation

logger = get_logger(__name__)


# ============================================================================
# Effort Prediction Functions
# ============================================================================

def effort_predict(
    df: pd.DataFrame,
    model_path: Optional[str] = None,
    return_proba: bool = True,
    colmap: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Predict fishing effort on GPS tracks using a trained model.

    Args:
        df: DataFrame with GPS tracks (columns: trip_id, timestamp, latitude, longitude)
        model_path: Path to trained model file (.joblib). If None, tries to load default.
        return_proba: Whether to return probability scores
        colmap: Optional dict mapping standard names to actual column names.
                Example: {'trip_id': 'TripNumber', 'time': 'GPS_Time',
                         'lat': 'Lat', 'lon': 'Lon'}

    Returns:
        DataFrame with original data plus:
        - effort_pred: Binary prediction (0 = non-fishing, 1 = fishing)
        - effort_prob: Fishing probability (0-1) if return_proba=True

    Example:
        >>> import pandas as pd
        >>> tracks = pd.DataFrame({
        ...     'trip_id': [1, 1, 2, 2],
        ...     'timestamp': ['2023-01-01 10:00', '2023-01-01 10:05', ...],
        ...     'latitude': [-6.0, -6.01, -5.9, -5.91],
        ...     'longitude': [39.0, 39.01, 39.1, 39.11]
        ... })
        >>> predictions = effort_predict(tracks)
    """
    # Extract column mappings from colmap if provided
    trip_col = None
    lat_col = None
    lon_col = None
    time_col = None

    if colmap:
        trip_col = colmap.get('trip_id')
        time_col = colmap.get('time') or colmap.get('timestamp')
        lat_col = colmap.get('lat') or colmap.get('latitude')
        lon_col = colmap.get('lon') or colmap.get('longitude')

    # Load model
    if model_path:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        logger.info(f"Loading model from {model_path}")
        model = EffortClassifier.load(model_path)
    else:
        # Try to load default trained model
        try:
            model = EffortClassifier.load_trained("rf")
            logger.info("Loaded default trained model")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load default model: {e}")
            raise RuntimeError(
                "No trained model found. Options:\n"
                "1. Use statistical prediction (no model needed): effort_predict_statistical()\n"
                "2. Train your own model: effort_fit() then use model_path\n"
                "3. Provide a trained model file via model_path parameter"
            ) from e

    # Make predictions
    return model.predict_df(
        df=df,
        trip_col=trip_col,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col,
        return_proba=return_proba,
    )


def effort_predict_statistical(
    df: pd.DataFrame,
    apply_filter: bool = False,
    trip_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    time_col: Optional[str] = None,
    colmap: Optional[dict] = None,
    use_parallel: bool = True,
    n_jobs: int = -1,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Predict fishing effort using statistical (rule-based) classifier.
    No training required - uses behavioral rules.

    Args:
        df: DataFrame with GPS tracks
        apply_filter: Enable shore distance filtering to remove on-land and near-shore
                      points. Bundled WIO coastline is used by default. Default: False
        trip_col: Trip ID column name (auto-detected if None)
        lat_col: Latitude column name (auto-detected if None)
        lon_col: Longitude column name (auto-detected if None)
        time_col: Timestamp column name (auto-detected if None)
        colmap: Optional dict mapping standard names to actual column names.
                Example: {'trip_id': 'TripNumber', 'time': 'GPS_Time',
                         'lat': 'Lat', 'lon': 'Lon'}
        use_parallel: Enable parallel processing across trips (default: True).
                      Recommended for datasets with >10 trips.
        n_jobs: Number of parallel workers. -1 = use all available CPUs (default: -1)
        config: Optional dict to override behavioral thresholds. Available keys:
                - min_fishing_speed (float): Min speed to classify as fishing, km/h (default: 0.5)
                - max_fishing_speed (float): Max speed to classify as fishing, km/h (default: 8.0)
                - min_transit_speed (float): Speed threshold for transit, km/h (default: 12.0)
                - high_turn_threshold (float): Turning angle for fishing behavior, degrees (default: 45.0)
                - low_straightness_threshold (float): Path straightness below = fishing (default: 0.4)
                - high_sinuosity_threshold (float): Sinuosity above = fishing (default: 1.5)
                - clustering_radius_km (float): Spatial clustering radius, km (default: 0.5)
                - time_windows (list): Rolling window sizes in minutes (default: [10.0])
                - spatial_window_km (float): Distance-based feature window, km (default: 1.0)
                - min_state_duration (int): Min consecutive points for a state (default: 3)
                - fishing_score_threshold (float): Score above = fishing, 0-1 (default: 0.5)
                - weight_speed (float): Weight for speed indicator (default: 3.0)
                - weight_turning (float): Weight for turning indicator (default: 2.0)
                - weight_straightness (float): Weight for straightness indicator (default: 2.0)
                - weight_sinuosity (float): Weight for sinuosity indicator (default: 1.5)
                - weight_clustering (float): Weight for clustering indicator (default: 2.0)
                - weight_speed_variability (float): Weight for speed variability (default: 1.5)

    Returns:
        DataFrame with original data plus:
        - is_fishing: Binary prediction (0 or 1)
        - fishing_score: Continuous fishing likelihood score (0-1)
        - activity_type: Activity category (fishing, sailing, starting_trip, ending_trip)
        - trip_phase: Trip phase (starting, in_progress, ending)
        - Additional feature columns (speed, acceleration, turning behavior, etc.)

    Example:
        >>> predictions = effort_predict_statistical(tracks)
        >>> predictions = effort_predict_statistical(tracks, apply_filter=True)
        >>> predictions = effort_predict_statistical(
        ...     tracks,
        ...     config={'max_fishing_speed': 6.0, 'fishing_score_threshold': 0.6}
        ... )
    """
    # Extract column mappings from colmap if provided
    if colmap:
        trip_col = trip_col or colmap.get('trip_id')
        time_col = time_col or colmap.get('time') or colmap.get('timestamp')
        lat_col = lat_col or colmap.get('lat') or colmap.get('latitude')
        lon_col = lon_col or colmap.get('lon') or colmap.get('longitude')

    # Create classifier
    classifier = StatisticalEffortClassifier(config=config)

    # Make predictions
    return classifier.predict(
        df=df,
        filter=apply_filter,
        trip_col=trip_col,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col,
        use_parallel=use_parallel,
        n_jobs=n_jobs,
    )


def effort_fit(
    df: pd.DataFrame,
    label_col: str = "Activity",
    save_path: Optional[str] = None,
    trip_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    time_col: Optional[str] = None,
    colmap: Optional[dict] = None,
) -> EffortClassifier:
    """
    Train a custom effort prediction model on labeled data.

    Args:
        df: DataFrame with GPS tracks and labels
        label_col: Column name with activity labels ('Fishing', 'Searching', 'Sailing', 'Traveling')
        save_path: Optional path to save trained model (.joblib)
        trip_col: Trip ID column name (auto-detected if None)
        lat_col: Latitude column name (auto-detected if None)
        lon_col: Longitude column name (auto-detected if None)
        time_col: Timestamp column name (auto-detected if None)
        colmap: Optional dict mapping standard names to actual column names

    Returns:
        Trained EffortClassifier

    Example:
        >>> model = effort_fit(training_data, label_col='Activity',
        ...                     save_path='my_model.joblib')
    """
    # Extract column mappings from colmap if provided
    if colmap:
        trip_col = trip_col or colmap.get('trip_id')
        time_col = time_col or colmap.get('time') or colmap.get('timestamp')
        lat_col = lat_col or colmap.get('lat') or colmap.get('latitude')
        lon_col = lon_col or colmap.get('lon') or colmap.get('longitude')

    # Create and train model
    model = EffortClassifier()
    model.fit_df(
        df=df,
        label_col=label_col,
        trip_col=trip_col,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col,
    )

    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    return model


# ============================================================================
# Gear Prediction Functions
# ============================================================================

def gear_predict(
    df: pd.DataFrame,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict fishing gear type from trip-level features.

    Args:
        df: DataFrame with trip-level features
            Required columns: duration_hours, distance_nm, mean_sog
        model_path: Path to trained model file (.joblib). Required unless using a default.

    Returns:
        DataFrame with original data plus:
        - gear_pred: Predicted gear type (e.g., "gillnet", "handline", "longline")

    Example:
        >>> import pandas as pd
        >>> trips = pd.DataFrame({
        ...     'trip_id': [1, 2, 3],
        ...     'duration_hours': [4.5, 3.2, 5.8],
        ...     'distance_nm': [12.3, 8.5, 15.2],
        ...     'mean_sog': [2.8, 2.6, 3.1]
        ... })
        >>> predictions = gear_predict(trips, model_path='gear_model.joblib')
    """
    # Load model
    if model_path:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        logger.info(f"Loading gear model from {model_path}")
        model = GearPredictor.load(model_path)
    else:
        # Try to load default trained model
        try:
            model = GearPredictor.load_trained("default")
            logger.info("Loaded default trained gear model")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load default gear model: {e}")
            raise RuntimeError(
                "No trained gear model found. Options:\n"
                "1. Train your own model: gear_fit() then use model_path\n"
                "2. Provide a trained model file via model_path parameter\n\n"
                "Note: Gear prediction requires trip-level data with columns:\n"
                "  - duration_hours\n"
                "  - distance_nm\n"
                "  - mean_sog"
            ) from e

    # Make predictions
    return model.predict_df(df)


def gear_fit(
    df: pd.DataFrame,
    label_col: str = "gear_label",
    save_path: Optional[str] = None,
) -> GearPredictor:
    """
    Train a custom gear prediction model on labeled trip data.

    Args:
        df: DataFrame with trip-level features and labels
            Required columns: duration_hours, distance_nm, mean_sog, gear_label
        label_col: Column name with gear type labels (default: 'gear_label')
        save_path: Optional path to save trained model (.joblib)

    Returns:
        Trained GearPredictor

    Example:
        >>> trips = pd.DataFrame({
        ...     'trip_id': [1, 2, 3, 4],
        ...     'duration_hours': [4.5, 3.2, 5.8, 4.1],
        ...     'distance_nm': [12.3, 8.5, 15.2, 10.1],
        ...     'mean_sog': [2.8, 2.6, 3.1, 2.5],
        ...     'gear_label': ['gillnet', 'handline', 'gillnet', 'longline']
        ... })
        >>> model = gear_fit(trips, save_path='my_gear_model.joblib')
    """
    # Validate required columns
    required_cols = ['duration_hours', 'distance_nm', 'mean_sog']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}\n"
            f"Gear prediction requires: {required_cols}"
        )

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    # Create and train model
    model = GearPredictor()
    # Temporarily set the label column in the dataframe with expected name
    df_train = df.copy()
    if label_col != 'gear_label':
        df_train['gear_label'] = df_train[label_col]

    model.fit_df(df_train)

    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        logger.info(f"Gear model saved to {save_path}")

    return model


# ============================================================================
# Vessel Type Prediction Functions
# ============================================================================

def vessel_predict(
    df: pd.DataFrame,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict vessel type (motorized vs. non-motorized) from trip-level features.

    Args:
        df: DataFrame with trip-level features
            Required columns: duration_hours, distance_nm, mean_sog
        model_path: Path to trained model file (.joblib). Required unless using a default.

    Returns:
        DataFrame with original data plus:
        - vessel_type_pred: Predicted vessel type (e.g., "motorized", "non-motorized")

    Example:
        >>> import pandas as pd
        >>> trips = pd.DataFrame({
        ...     'trip_id': [1, 2, 3],
        ...     'duration_hours': [4.5, 3.2, 5.8],
        ...     'distance_nm': [12.3, 8.5, 15.2],
        ...     'mean_sog': [2.8, 2.6, 3.1]
        ... })
        >>> predictions = vessel_predict(trips, model_path='vessel_model.joblib')
    """
    # Load model
    if model_path:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        logger.info(f"Loading vessel model from {model_path}")
        model = VesselTypePredictor.load(model_path)
    else:
        # Try to load default trained model
        try:
            model = VesselTypePredictor.load_trained("default")
            logger.info("Loaded default trained vessel model")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load default vessel model: {e}")
            raise RuntimeError(
                "No trained vessel model found. Options:\n"
                "1. Train your own model: vessel_fit() then use model_path\n"
                "2. Provide a trained model file via model_path parameter\n\n"
                "Note: Vessel type prediction requires trip-level data with columns:\n"
                "  - duration_hours\n"
                "  - distance_nm\n"
                "  - mean_sog"
            ) from e

    # Make predictions
    return model.predict_df(df)


def vessel_fit(
    df: pd.DataFrame,
    label_col: str = "vessel_type_label",
    save_path: Optional[str] = None,
) -> VesselTypePredictor:
    """
    Train a custom vessel type prediction model on labeled trip data.

    Args:
        df: DataFrame with trip-level features and labels
            Required columns: duration_hours, distance_nm, mean_sog, vessel_type_label
        label_col: Column name with vessel type labels (default: 'vessel_type_label')
        save_path: Optional path to save trained model (.joblib)

    Returns:
        Trained VesselTypePredictor

    Example:
        >>> trips = pd.DataFrame({
        ...     'trip_id': [1, 2, 3, 4],
        ...     'duration_hours': [4.5, 3.2, 5.8, 4.1],
        ...     'distance_nm': [12.3, 8.5, 15.2, 10.1],
        ...     'mean_sog': [2.8, 2.6, 3.1, 2.5],
        ...     'vessel_type_label': ['motorized', 'non-motorized', 'motorized', 'motorized']
        ... })
        >>> model = vessel_fit(trips, save_path='my_vessel_model.joblib')
    """
    # Validate required columns
    required_cols = ['duration_hours', 'distance_nm', 'mean_sog']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}\n"
            f"Vessel type prediction requires: {required_cols}"
        )

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    # Create and train model
    model = VesselTypePredictor()
    # Temporarily set the label column in the dataframe with expected name
    df_train = df.copy()
    if label_col != 'vessel_type_label':
        df_train['vessel_type_label'] = df_train[label_col]

    model.fit_df(df_train)

    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        logger.info(f"Vessel model saved to {save_path}")

    return model


# ============================================================================
# Hexagonal Grid Aggregation
# ============================================================================

def hexagrid_aggregate(
    df: pd.DataFrame,
    resolutions: Optional[list] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    time_col: Optional[str] = None,
    trip_col: Optional[str] = None,
    fishing_col: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_parquet: bool = True,
    save_csv: bool = False,
    min_hours: float = 0.1,
    min_trips: int = 1,
    min_days: int = 1,
) -> dict:
    """
    Aggregate GPS fishing tracks into H3 hexagonal grid cells.

    Computes time-weighted fishing effort metrics per hexagon across multiple
    temporal dimensions (overall, year, season, month, day/night). Designed
    for visualization and dashboard generation.

    Args:
        df: DataFrame with GPS fishing track data (with effort/fishing predictions)
        resolutions: List of H3 resolutions to compute (default: [7, 8])
        lat_col: Latitude column name (auto-detected if None)
        lon_col: Longitude column name (auto-detected if None)
        time_col: Timestamp column name (auto-detected if None)
        trip_col: Trip ID column name (auto-detected if None)
        fishing_col: Column with fishing classification 0/1 (auto-detected if None)
        output_dir: Directory to save results. If None, results are only returned.
        save_parquet: Save results as parquet files (default: True)
        save_csv: Also save results as CSV files (default: False)
        min_hours: Minimum fishing hours per hexagon to include (default: 0.1)
        min_trips: Minimum number of trips per hexagon to include (default: 1)
        min_days: Minimum number of days per hexagon to include (default: 1)

    Returns:
        Dictionary with aggregation results per resolution, keyed by resolution
        (e.g., 'res7', 'res8'). Each value contains temporal breakdowns:
        'overall', 'year', 'season', 'month', 'daynight'.

    Example:
        >>> # After running effort prediction
        >>> predictions <- effort_predict_statistical(tracks)
        >>> hex_results <- hexagrid_aggregate(predictions, resolutions = c(7L, 8L))
    """
    return run_hex_aggregation(
        df=df,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col,
        trip_col=trip_col,
        fishing_col=fishing_col,
        resolutions=resolutions,
        min_hours=min_hours,
        min_trips=min_trips,
        min_days=min_days,
        output_dir=output_dir,
        save_parquet=save_parquet,
        save_csv=save_csv,
    )
