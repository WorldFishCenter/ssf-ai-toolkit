# src/ssfaitk/r_api.py
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
from .utils.logging import get_logger

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
        import joblib
        
        model = EffortClassifier()
        artifact = joblib.load(model_path)
        
        # Handle different artifact formats
        if isinstance(artifact, dict):
            from sklearn.pipeline import Pipeline
            model.pipeline = Pipeline([("clf", artifact["model"])])
            model.feat_cols = artifact.get("feat_cols", model.feat_cols)
        elif hasattr(artifact, 'predict_df'):
            # Already an EffortClassifier instance
            model = artifact
        elif hasattr(artifact, 'predict'):
            from sklearn.pipeline import Pipeline
            model.pipeline = Pipeline([("clf", artifact)])
        else:
            raise ValueError(f"Unknown model format in {model_path}")
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
            )
    
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
    trip_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    time_col: Optional[str] = None,
    colmap: Optional[dict] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Predict fishing effort using statistical (rule-based) classifier.
    No training required - uses behavioral rules.
    
    Args:
        df: DataFrame with GPS tracks
        trip_col: Trip ID column name (auto-detected if None)
        lat_col: Latitude column name (auto-detected if None)
        lon_col: Longitude column name (auto-detected if None)
        time_col: Timestamp column name (auto-detected if None)
        colmap: Optional dict mapping standard names to actual column names.
                Example: {'trip_id': 'TripNumber', 'time': 'GPS_Time', 
                         'lat': 'Lat', 'lon': 'Lon'}
        config: Optional configuration dict for behavioral thresholds
    
    Returns:
        DataFrame with original data plus:
        - is_fishing: Binary prediction (0 or 1)
        - fishing_score: Continuous fishing likelihood score (0-1)
        - activity_type: Activity type ('fishing', 'sailing', 'starting_trip', 'ending_trip')
        - trip_phase: Trip phase ('starting', 'in_progress', 'ending')
        - Additional feature columns (speed, acceleration, turning behavior, etc.)
    
    Example:
        >>> predictions = effort_predict_statistical(tracks)
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
        trip_col=trip_col,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col,
        detect_trip_phase=True,
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
        import joblib
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': model.pipeline.named_steps['clf'],
            'feat_cols': model.feat_cols,
        }, save_path)
        logger.info(f"Model saved to {save_path}")
    
    return model
