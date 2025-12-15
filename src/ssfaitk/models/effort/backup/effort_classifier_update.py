# src/ssfaitk/models/effort/effort_classifier.py
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# ===============================================================
# Logger
# ===============================================================
logger = logging.getLogger(__name__)

# ===============================================================
# 1. Dynamic feature generator
# ===============================================================


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    b = np.degrees(np.arctan2(x, y))
    return (b + 360) % 360


def _median_cadence_seconds(g: pd.DataFrame, time_col="ltime") -> float:
    t = pd.to_datetime(g[time_col]).values
    if len(t) < 2:
        return 60.0
    dt = np.diff(t.astype("datetime64[s]")).astype(float)
    dt = dt[dt > 0]
    return float(np.median(dt)) if len(dt) else 60.0


def choose_adaptive_windows(
    df: pd.DataFrame,
    trip_col="Trip_ID",
    time_col="ltime",
    local_span_min=0.2,
    shape_span_min=1.0,
    min_win=3,
    max_win=25,
):
    rows = []
    for tid, g in df.groupby(trip_col):
        cad_s = _median_cadence_seconds(g, time_col)
        win_local = int(np.clip(round((local_span_min * 60.0) / cad_s), min_win, max_win))
        win_shape = int(np.clip(round((shape_span_min * 60.0) / cad_s), min_win, max_win))
        rows.append(
            {"Trip_ID": tid, "cadence_s": cad_s, "win_local": win_local, "win_shape": win_shape}
        )
    return pd.DataFrame(rows)


def add_features_dynamic(df: pd.DataFrame, speed_clip=60, accel_clip=30):
    import numpy as np
    import pandas as pd

    df["ltime"] = pd.to_datetime(df["ltime"], errors="coerce")
    df = df.sort_values(["Trip_ID", "ltime"]).copy()

    # ======================================================
    # Determine adaptive windows for each trip
    # ======================================================
    def _median_cadence_seconds(g: pd.DataFrame, time_col: str = "ltime") -> float:
        t = pd.to_datetime(g[time_col]).values
        if len(t) < 2:
            return 60.0
        dt = np.diff(t.astype("datetime64[s]")).astype(float)
        dt = dt[dt > 0]
        return float(np.median(dt)) if len(dt) else 60.0

    def choose_adaptive_windows(
        df: pd.DataFrame,
        trip_col: str = "Trip_ID",
        time_col: str = "ltime",
        local_span_min: float = 0.2,  # ~6 minutes
        shape_span_min: float = 1.0,  # ~18 minutes
        min_win: int = 3,
        max_win: int = 25,
    ) -> pd.DataFrame:
        rows = []
        for tid, g in df.groupby(trip_col):
            cad_s = _median_cadence_seconds(g, time_col)
            win_local = int(np.clip(round((local_span_min * 60.0) / cad_s), min_win, max_win))
            win_shape = int(np.clip(round((shape_span_min * 60.0) / cad_s), min_win, max_win))
            rows.append(
                {
                    trip_col: tid,
                    "cadence_s": cad_s,
                    "win_local": win_local,
                    "win_shape": win_shape,
                }
            )
        return pd.DataFrame(rows)

    wtbl = choose_adaptive_windows(df, trip_col="Trip_ID", time_col="ltime")
    df = df.merge(wtbl, on="Trip_ID", how="left")

    # ======================================================
    # Geometry helpers
    # ======================================================
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    def bearing_deg(lat1, lon1, lat2, lon2):
        dLon = np.radians(lon2 - lon1)
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)
        x = np.sin(dLon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
        b = np.degrees(np.arctan2(x, y))
        return (b + 360) % 360

    # ======================================================
    # Lags / leads
    # ======================================================
    for col in ["Latitude", "Longitude", "ltime"]:
        df[f"{col}_prev"] = df.groupby("Trip_ID")[col].shift(1)
        df[f"{col}_next"] = df.groupby("Trip_ID")[col].shift(-1)

    # Distances, time deltas, speed, acceleration, jerk
    df["dist_km"] = haversine(
        df["Latitude"], df["Longitude"], df["Latitude_prev"], df["Longitude_prev"]
    )
    df["dt_hr"] = (
        pd.to_datetime(df["ltime"]) - pd.to_datetime(df["ltime_prev"])
    ).dt.total_seconds() / 3600.0
    df.loc[df["dt_hr"] <= 0, "dt_hr"] = np.nan
    df["speed_kmh"] = df["dist_km"] / df["dt_hr"]

    df["speed_prev"] = df.groupby("Trip_ID")["speed_kmh"].shift(1)
    df["acceleration"] = (df["speed_kmh"] - df["speed_prev"]) / df["dt_hr"]
    df["accel_prev"] = df.groupby("Trip_ID")["acceleration"].shift(1)
    df["jerk"] = (df["acceleration"] - df["accel_prev"]) / df["dt_hr"]

    # Bearings / turning angles
    df["bearing"] = bearing_deg(
        df["Latitude"], df["Longitude"], df["Latitude_next"], df["Longitude_next"]
    )
    df["bearing_prev"] = df.groupby("Trip_ID")["bearing"].shift(1)
    da = df["bearing"] - df["bearing_prev"]
    da = ((da + 180) % 360) - 180
    df["turn_angle"] = da.abs()

    # ======================================================
    # Rolling stats & shape metrics (dynamic per trip)
    # ======================================================
    def _compute_shape_windows(g, win):
        lat = g["Latitude"].to_numpy()
        lon = g["Longitude"].to_numpy()
        n = len(g)
        h = win // 2
        straight, rog = np.empty(n), np.empty(n)
        for i in range(n):
            s, e = max(0, i - h), min(n, i + h + 1)
            lat_win, lon_win = lat[s:e], lon[s:e]
            if len(lat_win) < 2:
                straight[i], rog[i] = 1.0, 0.0
                continue
            chord = haversine(lat_win[0], lon_win[0], lat_win[-1], lon_win[-1])
            path = np.nansum(haversine(lat_win[:-1], lon_win[:-1], lat_win[1:], lon_win[1:]))
            straight[i] = float(chord / path) if path > 0 else 1.0
            latc, lonc = np.nanmean(lat_win), np.nanmean(lon_win)
            d = haversine(lat_win, lon_win, latc, lonc)
            rog[i] = float(np.sqrt(np.nanmean(d**2)))
        return pd.DataFrame({"straightness_w": straight, "rog_w": rog}, index=g.index)

    shape_frames = []
    for tid, g in df.groupby("Trip_ID"):
        wl = int(g["win_local"].iloc[0])
        ws = int(g["win_shape"].iloc[0])
        roll_local = lambda s: s.rolling(wl, min_periods=1, center=True)
        for c in ["speed_kmh", "acceleration", "turn_angle"]:
            df.loc[g.index, f"{c}_mean_w"] = roll_local(g[c]).mean().values
            df.loc[g.index, f"{c}_std_w"] = roll_local(g[c]).std().values
            df.loc[g.index, f"{c}_q25_w"] = roll_local(g[c]).quantile(0.25).values
            df.loc[g.index, f"{c}_q75_w"] = roll_local(g[c]).quantile(0.75).values
        df.loc[g.index, "lat_std_w"] = roll_local(g["Latitude"]).std().values
        df.loc[g.index, "lon_std_w"] = roll_local(g["Longitude"]).std().values
        shape_frames.append(_compute_shape_windows(g, ws))

    shape_df = pd.concat(shape_frames)
    df[["straightness_w", "rog_w"]] = shape_df[["straightness_w", "rog_w"]]

    # ======================================================
    # Trip-level context
    # ======================================================
    df["point_num"] = df.groupby("Trip_ID").cumcount()
    df["trip_size"] = df.groupby("Trip_ID")["Trip_ID"].transform("count")
    df["trip_pos_norm"] = df["point_num"] / df["trip_size"]
    df["t_curr"] = pd.to_datetime(df["ltime"])
    df["time_since_start_min"] = (
        df["t_curr"] - df.groupby("Trip_ID")["t_curr"].transform("min")
    ).dt.total_seconds() / 60.0
    df["hour"] = df["t_curr"].dt.hour
    df["dayofweek"] = df["t_curr"].dt.dayofweek

    # ======================================================
    # Stationarity proxy and distance to start
    # ======================================================
    df["is_stationary"] = (df["speed_kmh_mean_w"] < 2).astype(int)
    df["start_lat"] = df.groupby("Trip_ID")["Latitude"].transform("first")
    df["start_lon"] = df.groupby("Trip_ID")["Longitude"].transform("first")
    df["dist_to_start_km"] = haversine(
        df["Latitude"], df["Longitude"], df["start_lat"], df["start_lon"]
    )

    # ======================================================
    # Cleanup
    # ======================================================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[["speed_kmh", "acceleration", "jerk"]] = df[["speed_kmh", "acceleration", "jerk"]].fillna(
        0.0
    )
    df.fillna(0.0, inplace=True)
    df["speed_kmh"] = df["speed_kmh"].clip(0, speed_clip)
    df["acceleration"] = df["acceleration"].clip(-accel_clip, accel_clip)
    df["jerk"] = df["jerk"].clip(-3 * accel_clip, 3 * accel_clip)

    return df


# ===============================================================
# 2. EffortClassifier class
# ===============================================================
_FEAT_COLS = [
    "Latitude",
    "Longitude",
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


class EffortClassifier:
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.feat_cols = None

    def fit_df(self, df: pd.DataFrame):
        df_feat = add_features_dynamic(df)
        df_feat["y"] = (
            df_feat["Activity"]
            .replace({"Fishing": 1, "Searching": 1, "Sailing": 0, "Traveling": 0})
            .fillna(0)
        )
        X = df_feat[_FEAT_COLS].astype("float32")
        y = df_feat["y"].astype(int)
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X, y)
        self.pipeline = Pipeline([("clf", clf)])
        logger.info(f"Trained on {len(df_feat)} rows")
        return self

    def predict_df(self, df: pd.DataFrame):
        # Compute features only if not already provided
        if not set(self.feat_cols).issubset(df.columns):
            print("computing features..")
            df = add_features_dynamic(df)

        X = df[self.feat_cols].astype("float32")
        df["effort_pred"] = self.pipeline.predict(X)
        return df

    def save(self, model_name="rf"):
        path = (
            Path(__file__).resolve().parent / "artifacts" / f"effort_classifier_{model_name}.joblib"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {"model": self.pipeline["clf"], "feat_cols": self.feat_cols}
        joblib.dump(artifact, path)
        logger.info(f"Saved trained model to {path}")

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
