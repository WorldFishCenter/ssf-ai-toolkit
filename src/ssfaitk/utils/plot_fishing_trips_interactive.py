from __future__ import annotations

import folium
import pandas as pd
from folium.plugins import FeatureGroupSubGroup


def plot_fishing_trips_interactive(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    trip_col: str = "trip_id",
    pred_col: str = "is_fishing",
    map_center: tuple | None = None,
    zoom_start: int = 9,
):
    """
    Interactive map of trip routes, colored by fishing vs non-fishing.

    Parameters
    ----------
    df : pd.DataFrame
        Must include latitude, longitude, and predicted label columns.
    lat_col, lon_col : str
        Column names for coordinates.
    trip_col : str
        Column identifying trips (optional if all one trip).
    pred_col : str
        Binary column (1 = fishing, 0 = non-fishing).
    map_center : tuple
        (lat, lon) to center map. Defaults to mean of all points.
    zoom_start : int
        Initial zoom level.

    Returns
    -------
    folium.Map
        Interactive map object.
    """

    if map_center is None:
        map_center = (df[lat_col].mean(), df[lon_col].mean())

    fmap = folium.Map(location=map_center, zoom_start=zoom_start, tiles="CartoDB positron")

    # Create main layer groups
    fishing_layer = folium.FeatureGroup(name="Fishing", show=True)
    nonfishing_layer = folium.FeatureGroup(name="Non-Fishing", show=True)
    fmap.add_child(fishing_layer)
    fmap.add_child(nonfishing_layer)

    # Add trip subgroups
    for tid, g in df.groupby(trip_col if trip_col in df.columns else [0]):
        fishing_points = g[g[pred_col] == 1]
        nonfishing_points = g[g[pred_col] == 0]

        sub_fish = FeatureGroupSubGroup(fishing_layer, name=f"Trip {tid} - Fishing")
        sub_nonfish = FeatureGroupSubGroup(nonfishing_layer, name=f"Trip {tid} - Non-Fishing")

        # Fishing points
        for _, r in fishing_points.iterrows():
            folium.CircleMarker(
                location=(r[lat_col], r[lon_col]),
                radius=3,
                color="#1f77b4",
                fill=True,
                fill_opacity=0.8,
                popup=f"Trip {tid} - Fishing",
            ).add_to(sub_fish)

        # Non-fishing points
        for _, r in nonfishing_points.iterrows():
            folium.CircleMarker(
                location=(r[lat_col], r[lon_col]),
                radius=3,
                color="#ff7f0e",
                fill=True,
                fill_opacity=0.8,
                popup=f"Trip {tid} - Non-Fishing",
            ).add_to(sub_nonfish)

        fmap.add_child(sub_fish)
        fmap.add_child(sub_nonfish)

        # Draw line for full route
        folium.PolyLine(g[[lat_col, lon_col]].values, color="gray", weight=1.5, opacity=0.4).add_to(
            fmap
        )

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap
