# ssfaitk/models/effort/effort_statistical.py

import numpy as np
import pandas as pd

# from utils import haversine, bearing_deg  # reuse from your lib
from .utils import bearing_deg, haversine


class EffortStatistical:
    def __init__(self, config=None):
        from ssfaitk.utils.config import default_config

        self.config = config or default_config

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["Trip_ID", "ltime"]).copy()
        # Ensure datetime
        df["ltime"] = pd.to_datetime(df["ltime"], errors="coerce")

        # Shifted coordinates
        for col in ["Latitude", "Longitude", "ltime"]:
            df[f"{col}_prev"] = df.groupby("Trip_ID")[col].shift(1)
            df[f"{col}_next"] = df.groupby("Trip_ID")[col].shift(-1)

        # Distance and time difference
        df["dist_km"] = haversine(
            df["Latitude"], df["Longitude"], df["Latitude_prev"], df["Longitude_prev"]
        )
        df["dt_hr"] = (df["ltime"] - df["ltime_prev"]).dt.total_seconds() / 3600.0
        df.loc[df["dt_hr"] <= 0, "dt_hr"] = np.nan

        # Kinematic features
        df["speed_kmh"] = df["dist_km"] / df["dt_hr"]
        df["speed_prev"] = df.groupby("Trip_ID")["speed_kmh"].shift(1)
        df["acceleration"] = (df["speed_kmh"] - df["speed_prev"]) / df["dt_hr"]
        df["jerk"] = df.groupby("Trip_ID")["acceleration"].diff()

        # Turning
        df["bearing"] = bearing_deg(
            df["Latitude"], df["Longitude"], df["Latitude_next"], df["Longitude_next"]
        )
        df["bearing_prev"] = df.groupby("Trip_ID")["bearing"].shift(1)
        da = ((df["bearing"] - df["bearing_prev"] + 180) % 360) - 180
        df["turn_angle"] = da.abs()

        # Spatial and temporal context
        df["hour"] = df["ltime"].dt.hour
        df["dayofweek"] = df["ltime"].dt.dayofweek
        df["trip_pos_norm"] = df.groupby("Trip_ID").cumcount() / df.groupby("Trip_ID")[
            "Trip_ID"
        ].transform("count")

        # Stationary proxy
        df["is_stationary"] = (df["speed_kmh"] < self.config["stationary_speed"]).astype(int)

        return df.fillna(0)

    def _compute_score(self, df: pd.DataFrame) -> np.ndarray:
        c = self.config
        # Speed score: ideal within fishing range
        s_speed = np.clip(
            1
            - np.abs(df["speed_kmh"] - (c["fishing_speed_min"] + c["fishing_speed_max"]) / 2)
            / ((c["fishing_speed_max"] - c["fishing_speed_min"]) / 2),
            0,
            1,
        )
        # Turning score
        s_turn = np.clip(df["turn_angle"] / c["high_turn_threshold"], 0, 1)
        # Stationarity (high stationary ratio → high fishing)
        s_stat = df["is_stationary"] * 0.8
        # Straightness proxy (inverse if available)
        s_area = 1 - df.get("straightness_w", 0.8)
        # Combined weighted score
        score = (
            c["weight_speed"] * s_speed
            + c["weight_turning"] * s_turn
            + c["weight_straightness"] * s_area
            + c["weight_speed_variability"] * np.abs(df["acceleration"])
        ) / (
            c["weight_speed"]
            + c["weight_turning"]
            + c["weight_straightness"]
            + c["weight_speed_variability"]
        )
        return score

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Trip_ID" not in df.columns:
            df["Trip_ID"] = 1
        df_feat = self._compute_features(df)
        df_feat["fishing_score"] = self._compute_score(df_feat)
        df_feat["effort_pred"] = (
            df_feat["fishing_score"] >= self.config["fishing_score_threshold"]
        ).astype(int)
        return df_feat
