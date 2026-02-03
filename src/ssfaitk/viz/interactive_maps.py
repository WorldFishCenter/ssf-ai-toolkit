# src/ssfaitk/viz/interactive_maps.py
"""Interactive HTML maps for fishing effort visualization using folium."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Import column mapper for flexible column name handling
try:
    from ..utils.column_mapper import resolve_column_name
    COLUMN_MAPPER_AVAILABLE = True
except ImportError:
    COLUMN_MAPPER_AVAILABLE = False
    logger.warning("Column mapper not available - will use provided column names only")

# Optional folium import
try:
    import folium
    from folium import plugins

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logger.warning(
        "folium not installed. Interactive maps unavailable. "
        "Install with: pip install folium"
    )


def _check_folium() -> None:
    """Raise error if folium is not available."""
    if not FOLIUM_AVAILABLE:
        raise ImportError(
            "folium is required for interactive maps. Install with: pip install folium"
        )


def create_interactive_effort_map(
    df: pd.DataFrame,
    output_path: str | Path,
    effort_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    trip_col: Optional[str] = None,
    zoom_start: int = 10,
    tiles: str = "OpenStreetMap",
) -> Path:
    """
    Create an interactive HTML map showing fishing effort tracks.
    
    Column names are automatically detected if not provided. Searches for
    common variations like 'latitude', 'Latitude', 'lat', 'LAT', etc.
    
    Args:
        df: DataFrame with track points and effort predictions
        output_path: Where to save the HTML file
        effort_col: Effort column name (auto-detected if None)
        lat_col: Latitude column name (auto-detected if None)
        lon_col: Longitude column name (auto-detected if None)
        trip_col: Trip identifier column name (auto-detected if None)
        zoom_start: Initial zoom level (default: 10)
        tiles: Base map tiles ('OpenStreetMap', 'Stamen Terrain', 'CartoDB positron')
    
    Returns:
        Path to saved HTML file
    
    Examples:
        # Auto-detect all columns
        >>> create_interactive_effort_map(df, 'map.html')
        
        # Specify custom columns
        >>> create_interactive_effort_map(
        ...     df,
        ...     'map.html',
        ...     effort_col='fishing_activity',
        ...     lat_col='lat',
        ...     lon_col='lon'
        ... )
        
        # Works with any naming convention
        >>> df1 = pd.DataFrame({'Latitude': ..., 'effort_pred': ...})
        >>> df2 = pd.DataFrame({'lat': ..., 'is_fishing': ...})
        >>> df3 = pd.DataFrame({'LAT': ..., 'Fishing': ...})
        >>> # All work:
        >>> create_interactive_effort_map(df1, 'map1.html')
        >>> create_interactive_effort_map(df2, 'map2.html')
        >>> create_interactive_effort_map(df3, 'map3.html')
    """
    _check_folium()
    
    # Resolve column names (auto-detect or use provided)
    if COLUMN_MAPPER_AVAILABLE:
        effort_col = resolve_column_name(df, 'effort', effort_col, required=True)
        lat_col = resolve_column_name(df, 'latitude', lat_col, required=True)
        lon_col = resolve_column_name(df, 'longitude', lon_col, required=True)
        trip_col = resolve_column_name(df, 'trip_id', trip_col, required=True)
        logger.debug(f"Resolved columns: effort={effort_col}, lat={lat_col}, "
                    f"lon={lon_col}, trip={trip_col}")
    else:
        # Fallback: require explicit column names or use defaults
        effort_col = effort_col or 'is_fishing'
        lat_col = lat_col or 'latitude'
        lon_col = lon_col or 'longitude'
        trip_col = trip_col or 'trip_id'
        
        # Validate columns exist
        missing = [c for c in [effort_col, lat_col, lon_col, trip_col] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns not found: {missing}. "
                f"Available columns: {df.columns.tolist()}"
            )
    
    # Calculate map center
    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=tiles,
        control_scale=True,
    )
    
    # Add layer control
    fishing_group = folium.FeatureGroup(name="Fishing Activity", show=True)
    non_fishing_group = folium.FeatureGroup(name="Non-Fishing Activity", show=True)
    
    # Process each trip
    for trip_id in df[trip_col].unique():
        trip_df = df[df[trip_col] == trip_id].sort_values(by=lat_col)
        
        if len(trip_df) < 2:
            continue
        
        # Separate fishing and non-fishing segments
        fishing_points = trip_df[trip_df[effort_col] == 1]
        non_fishing_points = trip_df[trip_df[effort_col] == 0]
        
        # Add fishing segments (red)
        if len(fishing_points) > 1:
            coords = fishing_points[[lat_col, lon_col]].values.tolist()
            folium.PolyLine(
                coords,
                color="red",
                weight=3,
                opacity=0.7,
                popup=f"Trip: {trip_id} (Fishing)",
                tooltip="Fishing Activity",
            ).add_to(fishing_group)
        
        # Add non-fishing segments (blue)
        if len(non_fishing_points) > 1:
            coords = non_fishing_points[[lat_col, lon_col]].values.tolist()
            folium.PolyLine(
                coords,
                color="blue",
                weight=2,
                opacity=0.5,
                popup=f"Trip: {trip_id} (Non-fishing)",
                tooltip="Non-Fishing Activity",
            ).add_to(non_fishing_group)
    
    # Add groups to map
    fishing_group.add_to(m)
    non_fishing_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add statistics
    total_points = len(df)
    fishing_points = (df[effort_col] == 1).sum()
    fishing_pct = 100 * fishing_points / total_points if total_points > 0 else 0
    
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
        <p><b>Fishing Effort Statistics</b></p>
        <p>Total Points: {total_points:,}</p>
        <p>Fishing: {fishing_points:,} ({fishing_pct:.1f}%)</p>
        <p>Trips: {df[trip_col].nunique()}</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    
    logger.info(f"Saved interactive map: {output_path}")
    return output_path


def create_heatmap_html(
    df: pd.DataFrame,
    output_path: str | Path,
    effort_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    radius: int = 15,
    blur: int = 25,
    max_zoom: int = 13,
    tiles: str = "CartoDB positron",
) -> Path:
    """
    Create an interactive HTML heatmap of fishing effort.
    
    Column names are automatically detected if not provided. Searches for
    common variations like 'latitude', 'Latitude', 'lat', 'LAT', etc.
    
    Args:
        df: DataFrame with track points and effort predictions
        output_path: Where to save the HTML file
        effort_col: Effort column name (auto-detected if None)
        lat_col: Latitude column name (auto-detected if None)
        lon_col: Longitude column name (auto-detected if None)
        radius: Heatmap point radius (default: 15)
        blur: Heatmap blur amount (default: 25)
        max_zoom: Maximum zoom level (default: 13)
        tiles: Base map tiles (default: 'CartoDB positron')
    
    Returns:
        Path to saved HTML file, or None if no fishing effort detected
    
    Examples:
        # Auto-detect all columns
        >>> create_heatmap_html(df, 'heatmap.html')
        
        # Specify custom columns
        >>> create_heatmap_html(
        ...     df,
        ...     'heatmap.html',
        ...     effort_col='fishing_activity',
        ...     lat_col='lat',
        ...     lon_col='lon'
        ... )
        
        # Adjust heatmap appearance
        >>> create_heatmap_html(
        ...     df,
        ...     'heatmap.html',
        ...     radius=20,
        ...     blur=30,
        ...     max_zoom=15
        ... )
    """
    _check_folium()
    
    # Resolve column names (auto-detect or use provided)
    if COLUMN_MAPPER_AVAILABLE:
        effort_col = resolve_column_name(df, 'effort', effort_col, required=True)
        lat_col = resolve_column_name(df, 'latitude', lat_col, required=True)
        lon_col = resolve_column_name(df, 'longitude', lon_col, required=True)
        logger.info(f"Resolved columns: effort={effort_col}, lat={lat_col}, lon={lon_col}")
    else:
        logger.info(f"Unable to Resolved columns, COLUMN_MAPPER_AVAILABLE is {COLUMN_MAPPER_AVAILABLE}")

        # Fallback
        effort_col = effort_col or 'is_fishing'
        lat_col = lat_col or 'latitude'
        lon_col = lon_col or 'longitude'
        
        missing = [c for c in [effort_col, lat_col, lon_col] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns not found: {missing}. "
                f"Available columns: {df.columns.tolist()}"
            )
    
    fishing_df = df[df[effort_col] == 1].copy()
    
    if len(fishing_df) == 0:
        logger.warning("No fishing effort detected")
        return None
    
    # Calculate map center
    center_lat = fishing_df[lat_col].mean()
    center_lon = fishing_df[lon_col].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=tiles,
        control_scale=True,
    )
    
    # Prepare heatmap data
    heat_data = fishing_df[[lat_col, lon_col]].values.tolist()
    
    # Add heatmap layer
    plugins.HeatMap(
        heat_data,
        radius=radius,
        blur=blur,
        max_zoom=max_zoom,
        gradient={
            0.0: "blue",
            0.4: "lime",
            0.6: "yellow",
            0.8: "orange",
            1.0: "red",
        },
    ).add_to(m)
    
    # Add title
    title_html = """
    <div style="position: fixed; 
                top: 10px; left: 50%; transform: translateX(-50%); 
                width: 400px; height: 50px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:16px; text-align: center; padding: 10px">
        <b>Fishing Effort Heatmap</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add statistics
    stats_html = f"""
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 80px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
        <p><b>Statistics</b></p>
        <p>Fishing Points: {len(fishing_df):,}</p>
        <p>Coverage: {fishing_df[lat_col].nunique() * fishing_df[lon_col].nunique()} cells</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # Save map
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    
    logger.info(f"Saved interactive heatmap: {output_path}")
    return output_path


def create_clustered_effort_map(
    df: pd.DataFrame,
    output_path: str | Path,
    effort_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    trip_col: Optional[str] = None,
    tiles: str = "OpenStreetMap",
) -> Path:
    """
    Create an interactive map with clustered fishing effort points.
    
    Column names are automatically detected if not provided. Searches for
    common variations like 'latitude', 'Latitude', 'lat', 'LAT', etc.
    
    Args:
        df: DataFrame with track points and effort predictions
        output_path: Where to save the HTML file
        effort_col: Effort column name (auto-detected if None)
        lat_col: Latitude column name (auto-detected if None)
        lon_col: Longitude column name (auto-detected if None)
        trip_col: Trip identifier column name (auto-detected if None)
        tiles: Base map tiles (default: 'OpenStreetMap')
    
    Returns:
        Path to saved HTML file, or None if no fishing effort detected
    
    Examples:
        # Auto-detect all columns
        >>> create_clustered_effort_map(df, 'clustered.html')
        
        # Specify custom columns
        >>> create_clustered_effort_map(
        ...     df,
        ...     'clustered.html',
        ...     effort_col='fishing_activity',
        ...     lat_col='lat',
        ...     lon_col='lon',
        ...     trip_col='voyage_id'
        ... )
        
        # Use different base tiles
        >>> create_clustered_effort_map(
        ...     df,
        ...     'clustered.html',
        ...     tiles='CartoDB positron'
        ... )
    """
    _check_folium()
    
    # Resolve column names (auto-detect or use provided)
    if COLUMN_MAPPER_AVAILABLE:
        effort_col = resolve_column_name(df, 'effort', effort_col, required=True)
        lat_col = resolve_column_name(df, 'latitude', lat_col, required=True)
        lon_col = resolve_column_name(df, 'longitude', lon_col, required=True)
        trip_col = resolve_column_name(df, 'trip_id', trip_col, required=True)
        logger.debug(f"Resolved columns: effort={effort_col}, lat={lat_col}, "
                    f"lon={lon_col}, trip={trip_col}")
    else:
        # Fallback
        effort_col = effort_col or 'is_fishing'
        lat_col = lat_col or 'latitude'
        lon_col = lon_col or 'longitude'
        trip_col = trip_col or 'trip_id'
        
        missing = [c for c in [effort_col, lat_col, lon_col, trip_col] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns not found: {missing}. "
                f"Available columns: {df.columns.tolist()}"
            )
    
    # Filter to fishing points only
    fishing_df = df[df[effort_col] == 1].copy()
    
    if len(fishing_df) == 0:
        logger.warning("No fishing effort detected")
        return None
    
    # Calculate map center
    center_lat = fishing_df[lat_col].mean()
    center_lon = fishing_df[lon_col].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=tiles,
        control_scale=True,
    )
    
    # Create marker cluster
    marker_cluster = plugins.MarkerCluster(name="Fishing Points").add_to(m)
    
    # Add fishing points to cluster
    for idx, row in fishing_df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=3,
            popup=f"Trip: {row[trip_col]}",
            color="red",
            fill=True,
            fill_opacity=0.6,
        ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    
    logger.info(f"Saved clustered map: {output_path}")
    return output_path
