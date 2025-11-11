from __future__ import annotations

# src/ssfaitk/config.py

default_config = {
    # --- Speed thresholds (km/h) ---
    "fishing_speed_min": 0.8,
    "fishing_speed_max": 7.5,
    "stationary_speed": 1.5,  # Below this = stationary
    "transit_speed_min": 10.0,  # Above this = likely transit
    # --- Turning thresholds (degrees) ---
    "high_turn_threshold": 40.0,
    "moderate_turn_threshold": 15.0,
    # --- Spatial thresholds ---
    "low_straightness_threshold": 0.75,
    "high_sinuosity_threshold": 1.4,
    "clustering_radius_km": 1.5,
    # --- Speed variability ---
    "high_speed_cv_threshold": 0.7,
    # --- Feature weights ---
    "weight_speed": 3.0,
    "weight_turning": 2.0,
    "weight_straightness": 1.5,
    "weight_sinuosity": 1.0,
    "weight_clustering": 2.0,
    "weight_speed_variability": 1.0,
    # --- Classification threshold ---
    "fishing_score_threshold": 0.55,  # Higher = more conservative
}
