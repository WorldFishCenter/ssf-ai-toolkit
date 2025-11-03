"""
ssf-ai-toolkit | RingNet Effort Model Pipeline
Exact replication of the original Kenya RingNet feature engineering and model evaluation notebook.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 1. Data loading and preprocessing
# ============================================================
def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset and preprocess 'model'/'altitude' columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    data = pd.read_csv(csv_path)
    for col in ["Date", "ltime", "time_EAT"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")

    if "altitude" in data.columns:
        data["altitude"] = data["altitude"].fillna(data["altitude"].median())

    if "model" in data.columns:
        data["model"] = data["model"].fillna("Unknown")
        le_model = LabelEncoder()
        data["model_enc"] = le_model.fit_transform(data["model"])
    else:
        data["model_enc"] = 0

    return data


# ============================================================
# 2. Feature Engineering
# ============================================================
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


def add_features(
    df: pd.DataFrame, win: int = 5, speed_clip: float = 60, accel_clip: float = 30
) -> pd.DataFrame:
    """Feature engineering identical to the original notebook."""
    df = df.sort_values(["Trip_ID", "ltime"]).copy()

    # Lags/leads
    for col in ["Latitude", "Longitude", "ltime"]:
        df[f"{col}_prev"] = df.groupby("Trip_ID")[col].shift(1)
        df[f"{col}_next"] = df.groupby("Trip_ID")[col].shift(-1)

    # Distances, times, speed, acceleration, jerk
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

    # Bearings/turning
    df["bearing"] = bearing_deg(
        df["Latitude"], df["Longitude"], df["Latitude_next"], df["Longitude_next"]
    )
    df["bearing_prev"] = df.groupby("Trip_ID")["bearing"].shift(1)
    da = df["bearing"] - df["bearing_prev"]
    da = ((da + 180) % 360) - 180
    df["turn_angle"] = da.abs()

    # Rolling window stats (centered)
    roll = lambda s: s.rolling(win, min_periods=1, center=True)
    for c in ["speed_kmh", "acceleration", "turn_angle"]:
        df[f"{c}_mean_w"] = df.groupby("Trip_ID")[c].transform(lambda x: roll(x).mean())
        df[f"{c}_std_w"] = df.groupby("Trip_ID")[c].transform(lambda x: roll(x).std())
        df[f"{c}_q25_w"] = df.groupby("Trip_ID")[c].transform(lambda x: roll(x).quantile(0.25))
        df[f"{c}_q75_w"] = df.groupby("Trip_ID")[c].transform(lambda x: roll(x).quantile(0.75))

    # Spatial spread
    df["lat_std_w"] = df.groupby("Trip_ID")["Latitude"].transform(lambda x: roll(x).std())
    df["lon_std_w"] = df.groupby("Trip_ID")["Longitude"].transform(lambda x: roll(x).std())

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

    # Trajectory shape metrics (straightness, radius of gyration)
    def _straightness_win(lat_win, lon_win):
        if len(lat_win) < 2:
            return 1.0
        chord = haversine(lat_win[0], lon_win[0], lat_win[-1], lon_win[-1])
        path = np.nansum(haversine(lat_win[:-1], lon_win[:-1], lat_win[1:], lon_win[1:]))
        if not np.isfinite(path) or path == 0:
            return 1.0
        return float(chord / path)

    def _rog_win(lat_win, lon_win):
        if len(lat_win) == 0:
            return 0.0
        latc, lonc = np.nanmean(lat_win), np.nanmean(lon_win)
        d = haversine(lat_win, lon_win, latc, lonc)
        return float(np.sqrt(np.nanmean(d**2)))

    def _compute_shape_windows(g, win):
        lat = g["Latitude"].to_numpy()
        lon = g["Longitude"].to_numpy()
        n = len(g)
        h = win // 2
        straight = np.empty(n, dtype=float)
        rog = np.empty(n, dtype=float)
        for i in range(n):
            s = max(0, i - h)
            e = min(n, i + h + 1)
            straight[i] = _straightness_win(lat[s:e], lon[s:e])
            rog[i] = _rog_win(lat[s:e], lon[s:e])
        return pd.DataFrame({"straightness_w": straight, "rog_w": rog}, index=g.index)

    shape_df = df.groupby("Trip_ID", group_keys=False).apply(
        lambda g: _compute_shape_windows(g, win)
    )
    # df = df.fillna({col: 0.0 for col in df.select_dtypes(exclude=["datetime"]).columns})

    df[["straightness_w", "rog_w"]] = shape_df

    # Local context
    df["speed_mean_w"] = df.groupby("Trip_ID")["speed_kmh"].transform(lambda x: roll(x).mean())
    df["speed_std_w"] = df.groupby("Trip_ID")["speed_kmh"].transform(lambda x: roll(x).std())
    df["accel_mean_w"] = df.groupby("Trip_ID")["acceleration"].transform(lambda x: roll(x).mean())
    df["accel_std_w"] = df.groupby("Trip_ID")["acceleration"].transform(lambda x: roll(x).std())
    df["turn_mean_w"] = df.groupby("Trip_ID")["turn_angle"].transform(lambda x: roll(x).mean())
    df["turn_std_w"] = df.groupby("Trip_ID")["turn_angle"].transform(lambda x: roll(x).std())

    # Stationarity & distance
    df["is_stationary"] = (df["speed_mean_w"] < 2).astype(int)
    df["start_lat"] = df.groupby("Trip_ID")["Latitude"].transform("first")
    df["start_lon"] = df.groupby("Trip_ID")["Longitude"].transform("first")
    df["dist_to_start_km"] = haversine(
        df["Latitude"], df["Longitude"], df["start_lat"], df["start_lon"]
    )

    # Cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[["speed_kmh", "acceleration", "jerk"]] = df[["speed_kmh", "acceleration", "jerk"]].fillna(
        0.0
    )
    df.fillna(0.0, inplace=True)
    df["speed_kmh"] = df["speed_kmh"].clip(0, speed_clip)
    df["acceleration"] = df["acceleration"].clip(-accel_clip, accel_clip)
    df["jerk"] = df["jerk"].clip(-3 * accel_clip, 3 * accel_clip)
    return df


FEAT_COLS = [
    "Latitude",
    "Longitude",  #  "altitude", "model_enc",
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


# ============================================================
# 3. Model training / evaluation
# ============================================================
def grouped_split(X, y, groups, test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    return tr_idx, te_idx


def get_models():
    return {
        "LogReg": LogisticRegression(max_iter=1000, n_jobs=-1),
        "RF": RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42),
        "GBDT": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_jobs=-1),
        "XGB": xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),
        "LGBM": lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=8, random_seed=42, verbose=0
        ),
    }


def run_stage1_pipeline(csv_path: str):
    """End-to-end replication of the old notebook pipeline."""
    data = load_data(csv_path)
    data_feat = add_features(data, win=5)

    # Binary label
    data_feat["Activity_bin"] = (
        data_feat["Activity"]
        .replace(
            {
                "Fishing": "fishing_search",
                "Searching": "fishing_search",
                "Sailing": "sailing_travel",
                "Traveling": "sailing_travel",
            }
        )
        .fillna("sailing_travel")
    )

    X1 = data_feat[FEAT_COLS].astype("float32")
    y1 = data_feat["Activity_bin"]
    groups = data_feat["Trip_ID"]

    tr_idx, te_idx = grouped_split(X1, y1, groups, test_size=0.2, seed=42)
    X1_tr, X1_te = X1.iloc[tr_idx], X1.iloc[te_idx]
    y1_tr, y1_te = y1.iloc[tr_idx], y1.iloc[te_idx]

    le1 = LabelEncoder()
    y1_tr_enc = le1.fit_transform(y1_tr)
    y1_te_enc = le1.transform(y1_te)

    models = get_models()
    results = []
    for name, model in models.items():
        model.fit(X1_tr, y1_tr_enc)
        p = model.predict(X1_te)
        acc = accuracy_score(y1_te_enc, p)
        f1w = f1_score(y1_te_enc, p, average="weighted")
        results.append((name, acc, f1w))
        print(
            f"\n[Stage-1] {name}\n", classification_report(y1_te_enc, p, target_names=le1.classes_)
        )

    stage1_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1_weighted"]).sort_values(
        "F1_weighted", ascending=False
    )
    print("\nStage-1 Model Comparison:\n", stage1_df)
    return stage1_df


# ============================================================
# 4. Script entry point
# ============================================================
if __name__ == "__main__":
    csv_path = "../../../../data/dfKenyaAll_Tracks_df_cleaned.csv"
    # stage1_df = run_stage1_pipeline(csv_path)
    from ssfaitk.models import EffortClassifier

    # ---------------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------------
    df = pd.read_csv(csv_path)

    print("Unique activity labels:", df["Activity"].unique())

    # ---------------------------------------------------------------------
    # 2. Train model (Fishing + Searching = 1, Sailing = 0)
    # ---------------------------------------------------------------------
    clf = EffortClassifier.load_default()
    clf = clf.fit_df(df, label_col="Activity")

    # ---------------------------------------------------------------------
    # 3. Predict on the same dataset (or new data)
    # ---------------------------------------------------------------------
    preds = clf.predict_df(df)
    print(preds[["Activity", "effort_pred", "effort_prob"]].head())

    # ---------------------------------------------------------------------
    # 4. Save the trained model
    # ---------------------------------------------------------------------
    clf.save("model_artifacts/effort_rf_kenya.joblib")

    print("✅ Training complete — model saved to model_artifacts/effort_rf_kenya.joblib")

    from ssfaitk.eval.metrics import basic_classification_metrics

    # Convert your labels to binary (Fishing+Searching=1, Sailing=0)
    y_true = df["Activity"].replace({"Fishing": 1, "Searching": 1, "Sailing": 0})
    y_pred = preds["effort_pred"]

    metrics = basic_classification_metrics(y_true, y_pred)

    print("📊 Accuracy:", metrics["accuracy"])
    print("📊 F1-macro:", metrics["f1_macro"])
    print("Detailed classification report:")
    for label, vals in metrics["report"].items():
        print(label, vals)
