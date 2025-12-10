#!/usr/bin/env python3
"""
Interactive Trip Phase Visualizer

Creates interactive visualizations of fishing trips with color-coded activity phases.

Features:
- Interactive map with trip routes colored by activity
- Speed/progress charts
- Individual trip viewer
- Multi-trip comparison
- Export to HTML
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins
from pathlib import Path
from typing import Optional, List


# ===============================================================
# Color Schemes
# ===============================================================

ACTIVITY_COLORS = {
    'fishing': '#e74c3c',          # Red
    'sailing': '#3498db',          # Blue
    'starting_trip': '#2ecc71',    # Green
    'ending_trip': '#f39c12'       # Orange
}

PHASE_COLORS = {
    'starting': '#2ecc71',    # Green
    'in_progress': '#3498db', # Blue
    'ending': '#f39c12'       # Orange
}


# ===============================================================
# Folium Map Visualization
# ===============================================================

def create_trip_map(
    df: pd.DataFrame,
    trip_id: Optional[int] = None,
    color_by: str = 'activity_type',
    save_path: str = 'trip_map.html',
    zoom_start: int = 10,
) -> folium.Map:
    """
    Create interactive map with trip route colored by activity/phase.
    
    Args:
        df: DataFrame with predictions (must have trip phases)
        trip_id: Specific trip to plot (if None, plots all)
        color_by: 'activity_type' or 'trip_phase'
        save_path: Output HTML file path
        zoom_start: Initial zoom level
    
    Returns:
        Folium map object
    
    Examples:
        >>> # Single trip
        >>> map = create_trip_map(predictions, trip_id=12067898)
        >>> 
        >>> # All trips
        >>> map = create_trip_map(predictions)
        >>> 
        >>> # Color by phase instead of activity
        >>> map = create_trip_map(predictions, color_by='trip_phase')
    """
    # Filter to specific trip if requested
    if trip_id is not None:
        df = df[df['trip_id'] == trip_id].copy()
        if len(df) == 0:
            raise ValueError(f"Trip {trip_id} not found")
    
    # Validate color_by column
    if color_by not in df.columns:
        raise ValueError(f"Column '{color_by}' not found. Available: {df.columns.tolist()}")
    
    # Get colors
    if color_by == 'activity_type':
        colors = ACTIVITY_COLORS
    elif color_by == 'trip_phase':
        colors = PHASE_COLORS
    else:
        # Generic colors for other columns
        unique_values = df[color_by].unique()
        colors = {val: f'#{hash(val) % 0xFFFFFF:06x}' for val in unique_values}
    
    # Create map centered on data
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # Add tile layers
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    
    # Process each trip
    for trip_id_val, trip_data in df.groupby('trip_id'):
        trip_data = trip_data.sort_values('timestamp')
        
        # Create feature group for this trip
        fg = folium.FeatureGroup(name=f'Trip {trip_id_val}', show=True)
        
        # Plot points colored by activity/phase
        for category, color in colors.items():
            category_data = trip_data[trip_data[color_by] == category]
            
            if len(category_data) == 0:
                continue
            
            for idx, row in category_data.iterrows():
                # Create popup with details
                popup_html = f"""
                <div style="font-family: Arial; font-size: 12px;">
                    <b>Trip:</b> {trip_id_val}<br>
                    <b>Activity:</b> {row.get('activity_type', 'N/A')}<br>
                    <b>Phase:</b> {row.get('trip_phase', 'N/A')}<br>
                    <b>Speed:</b> {row.get('speed_kmh', 0):.1f} km/h<br>
                    <b>Fishing:</b> {'Yes' if row.get('is_fishing', 0) == 1 else 'No'}<br>
                    <b>Time:</b> {row['timestamp']}<br>
                    <b>Dist to start:</b> {row.get('dist_to_start_km', 0):.2f} km<br>
                    <b>Dist to end:</b> {row.get('dist_to_end_km', 0):.2f} km<br>
                    <b>Progress:</b> {row.get('trip_progress', 0):.1%}
                </div>
                """
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=4,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{category} - {row.get('speed_kmh', 0):.1f} km/h"
                ).add_to(fg)
        
        # Add start/end markers
        start = trip_data.iloc[0]
        end = trip_data.iloc[-1]
        
        folium.Marker(
            location=[start['latitude'], start['longitude']],
            popup=f"START - Trip {trip_id_val}",
            icon=folium.Icon(color='green', icon='play', prefix='fa'),
            tooltip='Trip Start'
        ).add_to(fg)
        
        folium.Marker(
            location=[end['latitude'], end['longitude']],
            popup=f"END - Trip {trip_id_val}",
            icon=folium.Icon(color='red', icon='stop', prefix='fa'),
            tooltip='Trip End'
        ).add_to(fg)
        
        fg.add_to(m)
    
    # Add legend
    legend_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 180px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
        <p style="margin:0; font-weight:bold;">Color: {color_by.replace('_', ' ').title()}</p>
    """
    
    for category, color in colors.items():
        legend_html += f"""
        <p style="margin:5px 0;">
            <span style="background-color:{color}; 
                        width:15px; height:15px; 
                        display:inline-block; margin-right:5px;"></span>
            {category.replace('_', ' ').title()}
        </p>
        """
    
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Save
    m.save(save_path)
    print(f"✓ Map saved: {save_path}")
    
    return m


# ===============================================================
# Plotly Chart Visualization
# ===============================================================

def create_trip_charts(
    df: pd.DataFrame,
    trip_id: Optional[int] = None,
    save_path: str = 'trip_charts.html',
) -> go.Figure:
    """
    Create interactive charts showing trip phases.
    
    Creates 4 subplots:
    1. Map view (lat/lon scatter)
    2. Speed over trip progress
    3. Distance to start/end
    4. Activity distribution
    
    Args:
        df: DataFrame with predictions
        trip_id: Specific trip to plot (if None, uses first trip)
        save_path: Output HTML file path
    
    Returns:
        Plotly figure object
    """
    # Get trip data
    if trip_id is not None:
        trip_data = df[df['trip_id'] == trip_id].copy()
        if len(trip_data) == 0:
            raise ValueError(f"Trip {trip_id} not found")
    else:
        # Use first trip
        trip_id = df['trip_id'].iloc[0]
        trip_data = df[df['trip_id'] == trip_id].copy()
    
    trip_data = trip_data.sort_values('timestamp')
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Trip Route',
            'Speed over Trip Progress',
            'Distance to Start/End',
            'Activity Distribution'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Subplot 1: Map view
    for activity, color in ACTIVITY_COLORS.items():
        activity_data = trip_data[trip_data['activity_type'] == activity]
        
        if len(activity_data) == 0:
            continue
        
        fig.add_trace(
            go.Scatter(
                x=activity_data['longitude'],
                y=activity_data['latitude'],
                mode='markers',
                name=activity.replace('_', ' ').title(),
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.7
                ),
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    'Lon: %{x:.4f}<br>'
                    'Lat: %{y:.4f}<br>'
                    'Speed: %{customdata[1]:.1f} km/h<br>'
                    'Time: %{customdata[2]}<br>'
                    '<extra></extra>'
                ),
                customdata=list(zip(
                    activity_data['activity_type'],
                    activity_data.get('speed_kmh', [0]*len(activity_data)),
                    activity_data['timestamp']
                )),
                legendgroup='activity',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add start/end markers
    fig.add_trace(
        go.Scatter(
            x=[trip_data['longitude'].iloc[0]],
            y=[trip_data['latitude'].iloc[0]],
            mode='markers',
            name='Start',
            marker=dict(color='green', size=15, symbol='triangle-up'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[trip_data['longitude'].iloc[-1]],
            y=[trip_data['latitude'].iloc[-1]],
            mode='markers',
            name='End',
            marker=dict(color='red', size=15, symbol='square'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Subplot 2: Speed over progress
    for activity, color in ACTIVITY_COLORS.items():
        activity_data = trip_data[trip_data['activity_type'] == activity]
        
        if len(activity_data) == 0:
            continue
        
        fig.add_trace(
            go.Scatter(
                x=activity_data['trip_progress'],
                y=activity_data.get('speed_kmh', [0]*len(activity_data)),
                mode='markers',
                name=activity.replace('_', ' ').title(),
                marker=dict(color=color, size=6, opacity=0.7),
                hovertemplate=(
                    '<b>%{customdata}</b><br>'
                    'Progress: %{x:.1%}<br>'
                    'Speed: %{y:.1f} km/h<br>'
                    '<extra></extra>'
                ),
                customdata=activity_data['activity_type'],
                legendgroup='activity',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Subplot 3: Distance to start/end
    fig.add_trace(
        go.Scatter(
            x=trip_data['trip_progress'],
            y=trip_data.get('dist_to_start_km', [0]*len(trip_data)),
            mode='lines',
            name='Distance to Start',
            line=dict(color='green', width=2),
            hovertemplate='Progress: %{x:.1%}<br>Distance: %{y:.2f} km<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=trip_data['trip_progress'],
            y=trip_data.get('dist_to_end_km', [0]*len(trip_data)),
            mode='lines',
            name='Distance to End',
            line=dict(color='red', width=2),
            hovertemplate='Progress: %{x:.1%}<br>Distance: %{y:.2f} km<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add threshold lines
    fig.add_hline(y=2.0, line_dash="dash", line_color="gray", 
                  annotation_text="Threshold (2 km)", row=2, col=1)
    
    # Subplot 4: Activity distribution
    activity_counts = trip_data['activity_type'].value_counts()
    
    fig.add_trace(
        go.Bar(
            x=activity_counts.index,
            y=activity_counts.values,
            marker=dict(color=[ACTIVITY_COLORS[a] for a in activity_counts.index]),
            text=[f'{v} ({100*v/len(trip_data):.1f}%)' for v in activity_counts.values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Longitude", row=1, col=1)
    fig.update_yaxes(title_text="Latitude", row=1, col=1)
    
    fig.update_xaxes(title_text="Trip Progress", row=1, col=2)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=2)
    
    fig.update_xaxes(title_text="Trip Progress", row=2, col=1)
    fig.update_yaxes(title_text="Distance (km)", row=2, col=1)
    
    fig.update_xaxes(title_text="Activity Type", row=2, col=2)
    fig.update_yaxes(title_text="Number of Points", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=f'Trip {trip_id} - Phase Analysis',
        height=800,
        showlegend=True,
        hovermode='closest',
        template='plotly_white'
    )
    
    # Save
    fig.write_html(save_path)
    print(f"✓ Charts saved: {save_path}")
    
    return fig


# ===============================================================
# Multi-Trip Comparison
# ===============================================================

def create_multi_trip_comparison(
    df: pd.DataFrame,
    trip_ids: Optional[List[int]] = None,
    save_path: str = 'multi_trip_comparison.html',
) -> go.Figure:
    """
    Create comparison chart for multiple trips.
    
    Args:
        df: DataFrame with predictions
        trip_ids: List of trip IDs (if None, uses first 5)
        save_path: Output HTML file path
    
    Returns:
        Plotly figure object
    """
    # Get trips
    if trip_ids is None:
        trip_ids = df['trip_id'].unique()[:5]
    
    # Filter
    df_trips = df[df['trip_id'].isin(trip_ids)].copy()
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for each trip
    for trip_id in trip_ids:
        trip_data = df_trips[df_trips['trip_id'] == trip_id].sort_values('timestamp')
        
        # Get activity percentages
        activity_pcts = trip_data['activity_type'].value_counts(normalize=True) * 100
        
        # Create hover text
        hover_text = f"Trip {trip_id}<br>"
        hover_text += f"Points: {len(trip_data):,}<br>"
        hover_text += f"Duration: {(pd.to_datetime(trip_data['timestamp'].max()) - pd.to_datetime(trip_data['timestamp'].min()))}<br>"
        for activity in ['fishing', 'sailing', 'starting_trip', 'ending_trip']:
            pct = activity_pcts.get(activity, 0)
            hover_text += f"{activity}: {pct:.1f}%<br>"
        
        # Plot route
        fig.add_trace(
            go.Scatter(
                x=trip_data['longitude'],
                y=trip_data['latitude'],
                mode='lines+markers',
                name=f'Trip {trip_id}',
                marker=dict(size=4, opacity=0.6),
                line=dict(width=2),
                hovertext=hover_text,
                hoverinfo='text'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Multi-Trip Comparison',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        height=600,
        hovermode='closest',
        template='plotly_white'
    )
    
    # Save
    fig.write_html(save_path)
    print(f"✓ Multi-trip comparison saved: {save_path}")
    
    return fig


# ===============================================================
# Main Interface
# ===============================================================

def visualize_trip_phases(
    df: pd.DataFrame,
    trip_id: Optional[int] = None,
    output_dir: str = 'trip_visualizations',
    create_map: bool = True,
    create_charts: bool = True,
    color_by: str = 'activity_type',
) -> dict:
    """
    Create all visualizations for trip phases.
    
    Args:
        df: DataFrame with trip phase predictions
        trip_id: Specific trip to visualize (if None, visualizes all/first)
        output_dir: Directory for output files
        create_map: Create interactive map
        create_charts: Create analysis charts
        color_by: Color scheme for map ('activity_type' or 'trip_phase')
    
    Returns:
        Dictionary with paths to created files
    
    Examples:
        >>> # Visualize specific trip
        >>> files = visualize_trip_phases(predictions, trip_id=12067898)
        >>> 
        >>> # Visualize all trips
        >>> files = visualize_trip_phases(predictions)
        >>> 
        >>> # Only create map
        >>> files = visualize_trip_phases(predictions, create_charts=False)
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    results = {}
    
    # Determine trip name for files
    if trip_id is not None:
        trip_name = f"trip_{trip_id}"
    else:
        trip_name = "all_trips"
    
    # Create map
    if create_map:
        map_path = f"{output_dir}/{trip_name}_map.html"
        try:
            create_trip_map(df, trip_id=trip_id, color_by=color_by, save_path=map_path)
            results['map'] = map_path
        except Exception as e:
            print(f"✗ Failed to create map: {e}")
    
    # Create charts
    if create_charts:
        chart_path = f"{output_dir}/{trip_name}_charts.html"
        try:
            create_trip_charts(df, trip_id=trip_id, save_path=chart_path)
            results['charts'] = chart_path
        except Exception as e:
            print(f"✗ Failed to create charts: {e}")
    
    return results


# ===============================================================
# Command-Line Interface
# ===============================================================

def main():
    """Command-line interface for trip visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize fishing trip phases interactively')
    parser.add_argument('input_file', help='Input parquet/csv file with predictions')
    parser.add_argument('--trip-id', type=int, help='Specific trip to visualize')
    parser.add_argument('--output-dir', default='trip_visualizations', help='Output directory')
    parser.add_argument('--color-by', choices=['activity_type', 'trip_phase'], 
                       default='activity_type', help='Color scheme for map')
    parser.add_argument('--no-map', action='store_true', help='Skip map creation')
    parser.add_argument('--no-charts', action='store_true', help='Skip charts creation')
    parser.add_argument('--compare', nargs='+', type=int, help='Compare multiple trips')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.input_file}...")
    if args.input_file.endswith('.parquet'):
        df = pd.read_parquet(args.input_file)
    else:
        df = pd.read_csv(args.input_file)
    
    print(f"✓ Loaded {len(df):,} points from {df['trip_id'].nunique()} trips")
    
    # Validate required columns
    required = ['latitude', 'longitude', 'trip_id', 'activity_type']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"✗ Missing required columns: {missing}")
        return
    
    # Multi-trip comparison
    if args.compare:
        print(f"\nCreating multi-trip comparison for trips: {args.compare}")
        create_multi_trip_comparison(
            df,
            trip_ids=args.compare,
            save_path=f"{args.output_dir}/multi_trip_comparison.html"
        )
        return
    
    # Single trip visualization
    print(f"\nCreating visualizations...")
    files = visualize_trip_phases(
        df,
        trip_id=args.trip_id,
        output_dir=args.output_dir,
        create_map=not args.no_map,
        create_charts=not args.no_charts,
        color_by=args.color_by
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nCreated files:")
    for file_type, path in files.items():
        print(f"  {file_type:10s}: {path}")
    
    print(f"\nOpen in browser to view!")


if __name__ == '__main__':
    main()
