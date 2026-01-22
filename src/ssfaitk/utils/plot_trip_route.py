#!/usr/bin/env python3
"""
Professional Interactive Trip Route Visualization

Creates a detailed, interactive HTML map showing a fishing vessel's complete trip
with fishing activity, speed, direction, and metadata.

Usage:
    python plot_trip_route_.py data.parquet trip_12345 output_map.html
    
    Or in Python:
    from plot_trip_route import plot_trip_route
    plot_trip_route(df, trip_id='12345', output_path='map.html')
"""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from ssfaitk.utils.column_mapper import resolve_column_name

# Try to import folium
try:
    import folium
    from folium import plugins
except ImportError:
    print("ERROR: folium not installed. Install with: pip install folium")
    sys.exit(1)


def calculate_trip_statistics(trip_df: pd.DataFrame, effort_col: str) -> dict:
    """Calculate comprehensive trip statistics."""
    
    stats = {}
    
    # Basic counts
    stats['total_points'] = len(trip_df)
    stats['fishing_points'] = (trip_df[effort_col] == 1).sum()
    stats['non_fishing_points'] = (trip_df[effort_col] == 0).sum()
    stats['fishing_percentage'] = 100 * stats['fishing_points'] / stats['total_points']
    
    # Time information
    if 'timestamp' in trip_df.columns:
        trip_df['timestamp'] = pd.to_datetime(trip_df['timestamp'])
        stats['start_time'] = trip_df['timestamp'].min()
        stats['end_time'] = trip_df['timestamp'].max()
        stats['duration'] = stats['end_time'] - stats['start_time']
        stats['duration_hours'] = stats['duration'].total_seconds() / 3600
    
    # Speed statistics (if available)
    if 'speed' in trip_df.columns:
        stats['avg_speed'] = trip_df['speed'].mean()
        stats['max_speed'] = trip_df['speed'].max()
        stats['avg_fishing_speed'] = trip_df[trip_df[effort_col] == 1]['speed'].mean()
        stats['avg_transit_speed'] = trip_df[trip_df[effort_col] == 0]['speed'].mean()
    
    # Distance calculations
    if 'latitude' in trip_df.columns and 'longitude' in trip_df.columns:
        from math import radians, sin, cos, sqrt, atan2
        
        def haversine(lat1, lon1, lat2, lon2):
            """Calculate distance between two points in km."""
            R = 6371  # Earth's radius in km
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c
        
        # Calculate cumulative distance
        distances = []
        for i in range(1, len(trip_df)):
            dist = haversine(
                trip_df.iloc[i-1]['latitude'],
                trip_df.iloc[i-1]['longitude'],
                trip_df.iloc[i]['latitude'],
                trip_df.iloc[i]['longitude']
            )
            distances.append(dist)
        
        stats['total_distance_km'] = sum(distances) if distances else 0
        
        # Distance from start to end (straight line)
        stats['displacement_km'] = haversine(
            trip_df.iloc[0]['latitude'],
            trip_df.iloc[0]['longitude'],
            trip_df.iloc[-1]['latitude'],
            trip_df.iloc[-1]['longitude']
        )
    
    return stats


def create_direction_arrows(trip_df: pd.DataFrame, lat_col: str, lon_col: str, 
                            interval: int = 10) -> list:
    """Create arrow markers to show direction of travel."""
    arrows = []
    
    for i in range(0, len(trip_df) - 1, interval):
        if i + 1 >= len(trip_df):
            break
            
        lat1, lon1 = trip_df.iloc[i][lat_col], trip_df.iloc[i][lon_col]
        lat2, lon2 = trip_df.iloc[i+1][lat_col], trip_df.iloc[i+1][lon_col]
        
        # Calculate bearing (direction)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        bearing = np.degrees(np.arctan2(dlon, dlat))
        
        # Create arrow marker
        arrows.append({
            'lat': lat1,
            'lon': lon1,
            'bearing': bearing
        })
    
    return arrows


def plot_trip_route(
    df: pd.DataFrame,
    trip_id: str,
    output_path: str = 'trip_route.html',
    effort_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    trip_col: Optional[str] = None,
    speed_col: Optional[str] = None,
    time_col: Optional[str] = None,
    show_direction_arrows: bool = True,
    arrow_interval: int = 15,
    tiles: str = 'OpenStreetMap',
) -> Path:
    """
    Create a professional interactive map showing a single trip route.
    
    Args:
        df: DataFrame with GPS track data
        trip_id: ID of the trip to visualize
        output_path: Where to save the HTML map
        effort_col: Column with fishing effort (0/1) - auto-detected if None
        lat_col: Latitude column - auto-detected if None
        lon_col: Longitude column - auto-detected if None
        trip_col: Trip ID column - auto-detected if None
        speed_col: Speed column (optional) - auto-detected if None
        time_col: Timestamp column - auto-detected if None
        show_direction_arrows: Whether to show direction arrows
        arrow_interval: Points between direction arrows
        tiles: Base map tiles ('OpenStreetMap', 'Stamen Terrain', 'CartoDB positron')
    
    Returns:
        Path to the saved HTML file
    
    Example:
        >>> df = pd.read_parquet('tracks.parquet')
        >>> plot_trip_route(df, trip_id='12345', output_path='trip_12345.html')
    """

    # Auto-detect columns (simple version - you can integrate column_mapper here)
    effort_col = resolve_column_name(df, 'effort', effort_col, required=False)
    lat_col = resolve_column_name(df, 'latitude', lat_col, required=True)
    lon_col = resolve_column_name(df, 'longitude', lon_col, required=True)
    trip_col = resolve_column_name(df, 'trip_id', trip_col, required=True)
    speed_col = resolve_column_name(df, 'speed', speed_col, required=True)
    time_col = resolve_column_name(df, 'timestamp', time_col, required=True)
    
    # Validate required columns
    if not all([lat_col, lon_col, trip_col]):
        raise ValueError(
            f"Could not find required columns. Found: {df.columns.tolist()}\n"
            f"Need: latitude, longitude, trip_id"
        )
    
    # Filter to specific trip
    trip_df = df[df[trip_col] == trip_id].copy()
    
    if len(trip_df) == 0:
        raise ValueError(f"No data found for trip ID: {trip_id}")
    
    # Sort by time if available to ensure consecutive points are actually consecutive
    if time_col and time_col in trip_df.columns:
        trip_df['timestamp'] = pd.to_datetime(trip_df[time_col])
        trip_df = trip_df.sort_values('timestamp').reset_index(drop=True)
    else:
        # If no time column, at least reset index to ensure proper ordering
        trip_df = trip_df.reset_index(drop=True)
    
    print(f"Plotting trip {trip_id} with {len(trip_df)} points...")
    
    # Calculate statistics
    
    # Calculate statistics
    stats = calculate_trip_statistics(trip_df, effort_col) if effort_col else {}
    
    # Calculate map center
    center_lat = trip_df[lat_col].mean()
    center_lon = trip_df[lon_col].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=tiles,
        control_scale=True,
    )
    
    # Add different tile layer options
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Create feature groups for layers
    route_group = folium.FeatureGroup(name='Complete Route', show=True)
    fishing_group = folium.FeatureGroup(name='Fishing Activity', show=True)
    non_fishing_group = folium.FeatureGroup(name='Non-Fishing Activity', show=True)
    points_group = folium.FeatureGroup(name='Individual Points', show=False)
    
    # Plot the complete route as a thin gray line first
    all_coords = trip_df[[lat_col, lon_col]].values.tolist()
    folium.PolyLine(
        all_coords,
        color='gray',
        weight=2,
        opacity=0.3,
        popup=f'Trip {trip_id} - Complete Route',
    ).add_to(route_group)
    
    # Plot fishing and non-fishing segments separately if effort column exists
    if effort_col and effort_col in trip_df.columns:
        # Identify consecutive segments of same activity
        # This prevents connecting fishing points that have non-fishing in between
        
        # Track activity changes
        current_activity = None
        current_segment = []
        
        for idx, row in trip_df.iterrows():
            activity = row[effort_col]
            point = [row[lat_col], row[lon_col]]
            
            # If activity changed or first point
            if activity != current_activity:
                # Save previous segment if it exists
                if len(current_segment) > 1:
                    if current_activity == 1:  # Fishing
                        folium.PolyLine(
                            current_segment,
                            color='#d62728',  # Red
                            weight=4,
                            opacity=0.8,
                            popup='Fishing Activity',
                        ).add_to(fishing_group)
                    else:  # Non-fishing
                        folium.PolyLine(
                            current_segment,
                            color='#1f77b4',  # Blue
                            weight=3,
                            opacity=0.6,
                            popup='Transit/Non-Fishing',
                        ).add_to(non_fishing_group)
                
                # Start new segment
                current_activity = activity
                current_segment = [point]
            else:
                # Continue current segment
                current_segment.append(point)
        
        # Don't forget the last segment!
        if len(current_segment) > 1:
            if current_activity == 1:  # Fishing
                folium.PolyLine(
                    current_segment,
                    color='#d62728',  # Red
                    weight=4,
                    opacity=0.8,
                    popup='Fishing Activity',
                ).add_to(fishing_group)
            else:  # Non-fishing
                folium.PolyLine(
                    current_segment,
                    color='#1f77b4',  # Blue
                    weight=3,
                    opacity=0.6,
                    popup='Transit/Non-Fishing',
                ).add_to(non_fishing_group)
    
    # Add individual points with detailed popups
    for idx, row in trip_df.iterrows():
        # Prepare popup information
        popup_lines = [f"<b>Point {idx}</b>"]
        
        if time_col and time_col in row:
            popup_lines.append(f"Time: {pd.to_datetime(row[time_col]).strftime('%Y-%m-%d %H:%M:%S')}")
        
        popup_lines.append(f"Position: {row[lat_col]:.5f}, {row[lon_col]:.5f}")
        
        if effort_col and effort_col in row:
            activity = "Fishing" if row[effort_col] == 1 else "Non-Fishing"
            popup_lines.append(f"Activity: {activity}")
        
        if speed_col and speed_col in row and pd.notna(row[speed_col]):
            popup_lines.append(f"Speed: {row[speed_col]:.1f} km/h")
        
        popup_html = "<br>".join(popup_lines)
        
        # Determine marker color
        if effort_col and effort_col in row:
            color = 'red' if row[effort_col] == 1 else 'blue'
        else:
            color = 'gray'
        
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=3,
            popup=folium.Popup(popup_html, max_width=250),
            color=color,
            fill=True,
            fillOpacity=0.6,
            weight=1,
        ).add_to(points_group)
    
    # Add START marker
    start_row = trip_df.iloc[0]
    folium.Marker(
        location=[start_row[lat_col], start_row[lon_col]],
        popup=folium.Popup(
            f"<b>START</b><br>"
            f"{pd.to_datetime(start_row[time_col]).strftime('%Y-%m-%d %H:%M') if time_col and time_col in start_row else 'Start of trip'}<br>"
            f"Position: {start_row[lat_col]:.5f}, {start_row[lon_col]:.5f}",
            max_width=250
        ),
        tooltip="Trip Start",
        icon=folium.Icon(color='green', icon='play', prefix='fa'),
    ).add_to(m)
    
    # Add END marker
    end_row = trip_df.iloc[-1]
    folium.Marker(
        location=[end_row[lat_col], end_row[lon_col]],
        popup=folium.Popup(
            f"<b>END</b><br>"
            f"{pd.to_datetime(end_row[time_col]).strftime('%Y-%m-%d %H:%M') if time_col and time_col in end_row else 'End of trip'}<br>"
            f"Position: {end_row[lat_col]:.5f}, {end_row[lon_col]:.5f}",
            max_width=250
        ),
        tooltip="Trip End",
        icon=folium.Icon(color='red', icon='stop', prefix='fa'),
    ).add_to(m)
    
    # Add direction arrows
    if show_direction_arrows:
        arrows = create_direction_arrows(trip_df, lat_col, lon_col, arrow_interval)
        for arrow in arrows:
            # Create a small rotated triangle icon using DivIcon
            icon_html = f"""
            <div style="transform: rotate({arrow['bearing']}deg); font-size: 16px; color: #333;">
                ▲
            </div>
            """
            folium.Marker(
                location=[arrow['lat'], arrow['lon']],
                icon=folium.DivIcon(html=icon_html),
            ).add_to(route_group)
    
    # Add all groups to map
    route_group.add_to(m)
    if effort_col and effort_col in trip_df.columns:
        fishing_group.add_to(m)
        non_fishing_group.add_to(m)
    points_group.add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    # Add title
    title_html = f"""
    <div style="position: fixed; 
                top: 10px; left: 50%; transform: translateX(-50%); 
                width: auto; min-width: 400px; height: auto; 
                background-color: white; border:3px solid #2171b5; z-index:9999; 
                border-radius: 5px;
                font-size:16px; text-align: center; padding: 15px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3)">
        <b style="font-size: 18px; color: #2171b5;">Trip Route Visualization</b><br>
        <span style="font-size: 14px;">Trip ID: {trip_id}</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add comprehensive statistics panel
    if stats:
        stats_lines = ["<p style='margin: 5px 0;'><b>Trip Statistics</b></p>"]
        stats_lines.append(f"<p style='margin: 3px 0;'>Total Points: {stats['total_points']:,}</p>")
        
        if 'fishing_points' in stats:
            stats_lines.append(
                f"<p style='margin: 3px 0;'>Fishing: "
                f"<span style='color: #d62728; font-weight: bold;'>{stats['fishing_points']:,}</span> "
                f"({stats['fishing_percentage']:.1f}%)</p>"
            )
            stats_lines.append(
                f"<p style='margin: 3px 0;'>Non-Fishing: "
                f"<span style='color: #1f77b4; font-weight: bold;'>{stats['non_fishing_points']:,}</span> "
                f"({100 - stats['fishing_percentage']:.1f}%)</p>"
            )
        
        if 'duration_hours' in stats:
            duration = stats['duration']
            days = duration.days
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            stats_lines.append(
                f"<p style='margin: 3px 0;'>Duration: {days}d {hours}h {minutes}m</p>"
            )
        
        if 'start_time' in stats:
            stats_lines.append(
                f"<p style='margin: 3px 0;'>Start: {stats['start_time'].strftime('%Y-%m-%d %H:%M')}</p>"
            )
            stats_lines.append(
                f"<p style='margin: 3px 0;'>End: {stats['end_time'].strftime('%Y-%m-%d %H:%M')}</p>"
            )
        
        if 'total_distance_km' in stats:
            stats_lines.append(f"<p style='margin: 3px 0;'>Distance: {stats['total_distance_km']:.1f} km</p>")
            stats_lines.append(f"<p style='margin: 3px 0;'>Displacement: {stats['displacement_km']:.1f} km</p>")
        
        if 'avg_speed' in stats and pd.notna(stats['avg_speed']):
            stats_lines.append(f"<p style='margin: 3px 0;'>Avg Speed: {stats['avg_speed']:.1f} km/h</p>")
            if 'avg_fishing_speed' in stats and pd.notna(stats['avg_fishing_speed']):
                stats_lines.append(f"<p style='margin: 3px 0; margin-left: 10px;'>Fishing: {stats['avg_fishing_speed']:.1f} km/h</p>")
            if 'avg_transit_speed' in stats and pd.notna(stats['avg_transit_speed']):
                stats_lines.append(f"<p style='margin: 3px 0; margin-left: 10px;'>Transit: {stats['avg_transit_speed']:.1f} km/h</p>")
        
        stats_html = f"""
        <div style="position: fixed; 
                    bottom: 20px; left: 20px; width: 240px; 
                    background-color: white; border:2px solid #2171b5; z-index:9999; 
                    border-radius: 5px;
                    font-size:12px; padding: 12px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3)">
            {''.join(stats_lines)}
        </div>
        """
        m.get_root().html.add_child(folium.Element(stats_html))
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 20px; right: 20px; width: 200px; 
                background-color: white; border:2px solid #2171b5; z-index:9999; 
                border-radius: 5px;
                font-size:12px; padding: 12px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3)">
        <p style="margin: 5px 0;"><b>Legend</b></p>
        <p style="margin: 5px 0;">
            <span style="color: #d62728; font-size: 20px; font-weight: bold;">━━</span> 
            Fishing Activity
        </p>
        <p style="margin: 5px 0;">
            <span style="color: #1f77b4; font-size: 20px; font-weight: bold;">━━</span> 
            Non-Fishing
        </p>
        <p style="margin: 5px 0;">
            <span style="color: gray; font-size: 20px;">━━</span> 
            Complete Route
        </p>
        <p style="margin: 5px 0;">
            <span style="font-size: 16px;">▲</span> 
            Direction Arrows
        </p>
        <p style="margin: 5px 0;">
            <i class="fa fa-play" style="color: green;"></i> Start Point
        </p>
        <p style="margin: 5px 0;">
            <i class="fa fa-stop" style="color: red;"></i> End Point
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add fullscreen button
    plugins.Fullscreen(
        position='topright',
        title='Fullscreen',
        title_cancel='Exit Fullscreen',
        force_separate_button=True,
    ).add_to(m)
    
    # Add measure control
    plugins.MeasureControl(
        position='topleft',
        primary_length_unit='kilometers',
        secondary_length_unit='miles',
        primary_area_unit='sqkilometers',
        secondary_area_unit='acres',
    ).add_to(m)
    
    # Save map
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    
    print(f"✓ Map saved to: {output_path}")
    print(f"  Total points: {len(trip_df)}")
    if stats and 'fishing_points' in stats:
        print(f"  Fishing: {stats['fishing_points']} ({stats['fishing_percentage']:.1f}%)")
    if stats and 'duration_hours' in stats:
        print(f"  Duration: {stats['duration_hours']:.1f} hours")
    if stats and 'total_distance_km' in stats:
        print(f"  Distance: {stats['total_distance_km']:.1f} km")
    
    return output_path


def main():
    """Command-line interface."""
    if len(sys.argv) < 3:
        print("Usage: python plot_trip_route_.py <data_file> <trip_id> [output_file]")
        print("\nExample:")
        print("  python plot_trip_route_.py tracks.parquet 12345 trip_12345.html")
        sys.exit(1)
    
    data_file = sys.argv[1]
    trip_id = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else f'trip_{trip_id}_route.html'
    
    print(f"Loading data from: {data_file}")
    
    # Load data
    if data_file.endswith('.parquet'):
        df = pd.read_parquet(data_file)
    elif data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    else:
        print(f"Unsupported file format: {data_file}")
        print("Supported formats: .parquet, .csv")
        sys.exit(1)
    
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create map
    plot_trip_route(df, trip_id=trip_id, output_path=output_file)
    
    print(f"\n✓ Success! Open the map in your browser:")
    print(f"  {Path(output_file).absolute()}")


if __name__ == '__main__':
    main()
