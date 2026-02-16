"""
H3 Hexagonal Fishing Effort Aggregation Pipeline
=================================================

Aggregates GPS fishing points into H3 hexagons with rich metrics
and multi-temporal dimensions for dashboard visualization.

Input: DataFrame with fishing-classified GPS tracks
Output: Hexagon-level aggregated metrics at multiple temporal granularities

Key Design Decisions:
- Time-weighted effort (not point counts) to handle variable ping rates
- Time interval cap at 300s (5 min) to handle vessel gaps
- H3 resolution 7 (~5.16 km²) for regional view, 8 (~0.74 km²) for detail
- Multi-temporal: overall, year, season, month, day/night
"""

import pandas as pd
import numpy as np
import h3
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Time interval cap: any gap > this is capped to avoid inflating effort
# between fishing activity vs device-off / trip-end gaps
MAX_TIME_INTERVAL_SECONDS = 600  # 10 minutes

# H3 resolutions
H3_RES_REGIONAL = 7  # ~5.16 km² hexagons - regional overview
H3_RES_DETAIL = 8  # ~0.74 km² hexagons - detailed hotspots
H3_RES_FINE = 9  # ~0.10 km² hexagons - fine-grained (optional)

# Season definitions for Southern Hemisphere / WIO region
# Adjust if needed for your study area
SEASON_MAP = {
    1: 'NE_Monsoon',  # Jan - Northeast Monsoon (hot, wet)
    2: 'NE_Monsoon',  # Feb
    3: 'NE_Monsoon',  # Mar
    4: 'Inter_Monsoon',  # Apr - Transition
    5: 'SE_Monsoon',  # May - Southeast Monsoon (cool, dry)
    6: 'SE_Monsoon',  # Jun
    7: 'SE_Monsoon',  # Jul
    8: 'SE_Monsoon',  # Aug
    9: 'SE_Monsoon',  # Sep
    10: 'Inter_Monsoon',  # Oct - Transition
    11: 'NE_Monsoon',  # Nov
    12: 'NE_Monsoon',  # Dec
}

# Day/night boundaries (approximate for tropical WIO region)
# ~6 AM sunrise, ~6 PM sunset year-round near equator
DAY_START_HOUR = 6
DAY_END_HOUR = 18

# =============================================================================
# COLUMN AUTO-DETECTION (uses toolkit's column_mapper)
# =============================================================================

try:
    from ssfaitk.utils.column_mapper import resolve_column_name

    COLUMN_MAPPER_AVAILABLE = True
except ImportError:
    try:
        from utils.column_mapper import resolve_column_name

        COLUMN_MAPPER_AVAILABLE = True
    except ImportError:
        COLUMN_MAPPER_AVAILABLE = False
        print("⚠ column_mapper not found - falling back to basic column detection")


        def resolve_column_name(df, standard_name, user_override=None, required=True):
            """Fallback: basic column detection if column_mapper not available."""
            _FALLBACK = {
                'latitude': ['latitude', 'Latitude', 'lat', 'LAT'],
                'longitude': ['longitude', 'Longitude', 'lon', 'LON'],
                'timestamp': ['timestamp', 'time', 'ltime', 'datetime', 'Timestamp'],
                'trip_id': ['trip_id', 'Trip_ID', 'voyage_id', 'TRIP_ID', 'tripID'],
                'speed': ['speed', 'speed_kmh', 'Speed', 'sog', 'speed_knots'],
                'effort': ['is_fishing', 'fishing', 'effort_pred', 'Fishing'],
            }
            if user_override and user_override in df.columns:
                return user_override
            for c in _FALLBACK.get(standard_name, []):
                if c in df.columns:
                    return c
            if required:
                raise ValueError(
                    f"Column '{standard_name}' not found. "
                    f"Available: {df.columns.tolist()}"
                )
            return None


# =============================================================================
# STEP 1: PREPARE DATA WITH TEMPORAL DIMENSIONS
# =============================================================================

def prepare_temporal_dimensions(df, time_col):
    """
    Add all temporal dimension columns needed for aggregation.

    Adds: year, month, season, day_night, date, hour
    """
    df = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # Extract temporal dimensions
    df['year'] = df[time_col].dt.year
    df['month'] = df[time_col].dt.month
    df['month_name'] = df[time_col].dt.strftime('%b')  # Jan, Feb, ...
    df['season'] = df['month'].map(SEASON_MAP)
    df['hour'] = df[time_col].dt.hour
    df['day_night'] = np.where(
        (df['hour'] >= DAY_START_HOUR) & (df['hour'] < DAY_END_HOUR),
        'day', 'night'
    )
    df['date'] = df[time_col].dt.date

    return df


# =============================================================================
# STEP 2: COMPUTE TIME-WEIGHTED INTERVALS
# =============================================================================

def compute_time_intervals(df, time_col, trip_col, max_interval=MAX_TIME_INTERVAL_SECONDS):
    """
    Compute time interval (in hours) each GPS point represents.

    For each point, the interval is half the gap to the previous point
    plus half the gap to the next point (midpoint method), capped at max_interval.

    This correctly handles:
    - Variable ping rates (1-16+ seconds)
    - Trip boundaries (no bleeding between trips)
    - Device gaps (capped to avoid inflation)

    Returns DataFrame with 'time_hours' column.
    """
    df = df.copy()
    df = df.sort_values([trip_col, time_col]).reset_index(drop=True)

    # Time differences within each trip
    df['_dt_seconds'] = df.groupby(trip_col)[time_col].diff().dt.total_seconds()

    # Cap large gaps
    df['_dt_capped'] = df['_dt_seconds'].clip(upper=max_interval)

    # Forward gap (to next point) within same trip
    df['_dt_forward'] = df.groupby(trip_col)['_dt_capped'].shift(-1)

    # Midpoint method: each point represents half backward + half forward interval
    # First point of trip: only forward half
    # Last point of trip: only backward half
    # Middle points: half backward + half forward
    backward_half = df['_dt_capped'].fillna(0) / 2
    forward_half = df['_dt_forward'].fillna(0) / 2

    df['time_seconds'] = backward_half + forward_half

    # For first point of each trip (no backward), use full forward half
    first_mask = df['_dt_seconds'].isna()
    df.loc[first_mask, 'time_seconds'] = df.loc[first_mask, '_dt_forward'].fillna(0) / 2

    # Convert to hours
    df['time_hours'] = df['time_seconds'] / 3600

    # Cleanup temp columns
    df.drop(columns=['_dt_seconds', '_dt_capped', '_dt_forward'], inplace=True)

    return df


# =============================================================================
# STEP 3: ASSIGN H3 HEXAGONS
# =============================================================================

def assign_h3_hexagons(df, lat_col, lon_col, resolutions=None):
    """
    Assign each GPS point to H3 hexagons at multiple resolutions.

    Creates columns: h3_res7, h3_res8, (and h3_res9 if requested)
    """
    if resolutions is None:
        resolutions = [H3_RES_REGIONAL, H3_RES_DETAIL]

    df = df.copy()

    for res in resolutions:
        col_name = f'h3_res{res}'
        print(f"  Assigning H3 resolution {res} ({col_name})...")

        df[col_name] = [
            h3.latlng_to_cell(lat, lon, res)
            for lat, lon in zip(df[lat_col], df[lon_col])
        ]

    return df


# =============================================================================
# STEP 4: AGGREGATE METRICS PER HEXAGON
# =============================================================================

def aggregate_hexagon_metrics(df, h3_col, trip_col, time_col, speed_col=None):
    """
    Compute rich metrics per hexagon for a given grouping of data.

    Metrics:
    - fishing_hours: Total time-weighted fishing effort (PRIMARY METRIC)
    - n_points: Number of GPS points (for reference, not primary)
    - n_vessels: Unique vessel/trip count (proxy for vessel count)
    - n_trips: Unique trips that passed through
    - n_days: Number of unique calendar days with activity
    - vessel_days: Unique (vessel, day) combinations
    - mean_speed_kmh: Average speed while fishing
    - std_speed_kmh: Speed variability (indicates fishing behavior intensity)
    - temporal_coverage: Fraction of total days in period with activity
    - lat, lon: Hexagon center coordinates
    """

    agg_dict = {
        'time_hours': 'sum',  # Total fishing hours
        time_col: ['count', 'min', 'max'],  # Point count, time range
    }

    # Group and aggregate
    grouped = df.groupby(h3_col)

    # Basic aggregations
    result = grouped['time_hours'].sum().reset_index()
    result.columns = [h3_col, 'fishing_hours']

    # Point count
    result['n_points'] = grouped[time_col].count().values

    # Unique trips
    result['n_trips'] = grouped[trip_col].nunique().values

    # Unique days
    result['n_days'] = grouped['date'].nunique().values

    # Vessel-days (unique trip-date combinations)
    vessel_days = df.groupby(h3_col).apply(
        lambda x: x[[trip_col, 'date']].drop_duplicates().shape[0],
        include_groups=False
    )
    result['vessel_days'] = result[h3_col].map(vessel_days).values

    # Speed stats (if available)
    if speed_col and speed_col in df.columns:
        speed_stats = grouped[speed_col].agg(['mean', 'std'])
        result['mean_speed_kmh'] = speed_stats['mean'].values
        result['std_speed_kmh'] = speed_stats['std'].values

    # Time range
    time_range = grouped[time_col].agg(['min', 'max'])
    result['first_seen'] = time_range['min'].values
    result['last_seen'] = time_range['max'].values

    # Temporal coverage: n_days / total_possible_days
    if len(df) > 0:
        total_days = (df[time_col].max() - df[time_col].min()).days + 1
        if total_days > 0:
            result['temporal_coverage'] = result['n_days'] / total_days
        else:
            result['temporal_coverage'] = 1.0

    # Hexagon center coordinates
    result['lat'] = result[h3_col].apply(lambda x: h3.cell_to_latlng(x)[0])
    result['lon'] = result[h3_col].apply(lambda x: h3.cell_to_latlng(x)[1])

    # Hexagon boundary (for precise rendering)
    result['hex_boundary'] = result[h3_col].apply(
        lambda x: h3.cell_to_boundary(x)
    )

    # Derived metrics
    result['hours_per_trip'] = result['fishing_hours'] / result['n_trips']
    result['hours_per_day'] = result['fishing_hours'] / result['n_days']

    return result


# =============================================================================
# STEP 5: MULTI-TEMPORAL AGGREGATION
# =============================================================================

def aggregate_all_temporal(df, h3_col, trip_col, time_col, speed_col=None):
    """
    Run hexagon aggregation across all temporal dimensions.

    Returns a dictionary of DataFrames:
    - 'overall': All data combined
    - 'by_year': Grouped by year
    - 'by_month': Grouped by month
    - 'by_season': Grouped by season
    - 'by_day_night': Grouped by day/night
    - 'by_year_month': Grouped by year + month
    - 'by_year_season': Grouped by year + season
    - 'by_season_day_night': Grouped by season + day/night
    """
    results = {}

    # ----- OVERALL -----
    print("\n  [1/8] Overall aggregation...")
    results['overall'] = aggregate_hexagon_metrics(
        df, h3_col, trip_col, time_col, speed_col
    )
    print(f"        → {len(results['overall']):,} hexagons")

    # ----- BY YEAR -----
    print("  [2/8] By year...")
    yearly_dfs = []
    for year, group in df.groupby('year'):
        agg = aggregate_hexagon_metrics(group, h3_col, trip_col, time_col, speed_col)
        agg['year'] = year
        yearly_dfs.append(agg)
    results['by_year'] = pd.concat(yearly_dfs, ignore_index=True) if yearly_dfs else pd.DataFrame()
    print(f"        → {len(results['by_year']):,} hexagon-year combinations")

    # ----- BY MONTH -----
    print("  [3/8] By month...")
    monthly_dfs = []
    for month, group in df.groupby('month'):
        agg = aggregate_hexagon_metrics(group, h3_col, trip_col, time_col, speed_col)
        agg['month'] = month
        agg['month_name'] = SEASON_MAP.get(month, f'M{month}')  # will fix below
        monthly_dfs.append(agg)
    results['by_month'] = pd.concat(monthly_dfs, ignore_index=True) if monthly_dfs else pd.DataFrame()
    # Fix month_name
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    if len(results['by_month']) > 0:
        results['by_month']['month_name'] = results['by_month']['month'].map(month_names)
    print(f"        → {len(results['by_month']):,} hexagon-month combinations")

    # ----- BY SEASON -----
    print("  [4/8] By season...")
    season_dfs = []
    for season, group in df.groupby('season'):
        agg = aggregate_hexagon_metrics(group, h3_col, trip_col, time_col, speed_col)
        agg['season'] = season
        season_dfs.append(agg)
    results['by_season'] = pd.concat(season_dfs, ignore_index=True) if season_dfs else pd.DataFrame()
    print(f"        → {len(results['by_season']):,} hexagon-season combinations")

    # ----- BY DAY/NIGHT -----
    print("  [5/8] By day/night...")
    dn_dfs = []
    for dn, group in df.groupby('day_night'):
        agg = aggregate_hexagon_metrics(group, h3_col, trip_col, time_col, speed_col)
        agg['day_night'] = dn
        dn_dfs.append(agg)
    results['by_day_night'] = pd.concat(dn_dfs, ignore_index=True) if dn_dfs else pd.DataFrame()
    print(f"        → {len(results['by_day_night']):,} hexagon-daynight combinations")

    # ----- BY YEAR + MONTH -----
    print("  [6/8] By year + month...")
    ym_dfs = []
    for (year, month), group in df.groupby(['year', 'month']):
        agg = aggregate_hexagon_metrics(group, h3_col, trip_col, time_col, speed_col)
        agg['year'] = year
        agg['month'] = month
        agg['month_name'] = month_names.get(month, f'M{month}')
        ym_dfs.append(agg)
    results['by_year_month'] = pd.concat(ym_dfs, ignore_index=True) if ym_dfs else pd.DataFrame()
    print(f"        → {len(results['by_year_month']):,} hexagon-year-month combinations")

    # ----- BY YEAR + SEASON -----
    print("  [7/8] By year + season...")
    ys_dfs = []
    for (year, season), group in df.groupby(['year', 'season']):
        agg = aggregate_hexagon_metrics(group, h3_col, trip_col, time_col, speed_col)
        agg['year'] = year
        agg['season'] = season
        ys_dfs.append(agg)
    results['by_year_season'] = pd.concat(ys_dfs, ignore_index=True) if ys_dfs else pd.DataFrame()
    print(f"        → {len(results['by_year_season']):,} hexagon-year-season combinations")

    # ----- BY SEASON + DAY/NIGHT -----
    print("  [8/8] By season + day/night...")
    sdn_dfs = []
    for (season, dn), group in df.groupby(['season', 'day_night']):
        agg = aggregate_hexagon_metrics(group, h3_col, trip_col, time_col, speed_col)
        agg['season'] = season
        agg['day_night'] = dn
        sdn_dfs.append(agg)
    results['by_season_day_night'] = pd.concat(sdn_dfs, ignore_index=True) if sdn_dfs else pd.DataFrame()
    print(f"        → {len(results['by_season_day_night']):,} hexagon-season-daynight combinations")

    return results


# =============================================================================
# STEP 6: QUALITY FILTERS
# =============================================================================

def apply_quality_filters(hex_df, min_hours=0.1, min_trips=1, min_days=1):
    """
    Filter out low-confidence hexagons.

    Args:
        hex_df: Aggregated hexagon DataFrame
        min_hours: Minimum fishing hours to keep hexagon
        min_trips: Minimum number of unique trips
        min_days: Minimum number of unique days

    Returns:
        Filtered DataFrame with 'confidence' column added
    """
    df = hex_df.copy()
    original_count = len(df)

    # Apply filters
    mask = (
            (df['fishing_hours'] >= min_hours) &
            (df['n_trips'] >= min_trips) &
            (df['n_days'] >= min_days)
    )
    df = df[mask].copy()

    # Add confidence score (0-1)
    # Based on data richness: more trips, more days, more hours = higher confidence
    if len(df) > 0:
        trip_score = np.clip(df['n_trips'] / df['n_trips'].quantile(0.75), 0, 1)
        day_score = np.clip(df['n_days'] / max(df['n_days'].quantile(0.75), 1), 0, 1)
        hour_score = np.clip(df['fishing_hours'] / df['fishing_hours'].quantile(0.75), 0, 1)
        df['confidence'] = (trip_score + day_score + hour_score) / 3

    filtered_count = original_count - len(df)
    if filtered_count > 0:
        print(f"    Filtered out {filtered_count} low-confidence hexagons "
              f"({filtered_count / original_count * 100:.1f}%)")

    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_hex_aggregation(
        df,
        # Column names (auto-detected if None)
        lat_col=None,
        lon_col=None,
        time_col=None,
        trip_col=None,
        speed_col=None,
        fishing_col=None,
        # H3 settings
        resolutions=None,
        # Time interval settings
        max_time_interval=MAX_TIME_INTERVAL_SECONDS,
        # Quality filter settings
        min_hours=0.1,
        min_trips=1,
        min_days=1,
        # Output settings
        output_dir=None,
        save_parquet=True,
        save_csv=False,
):
    """
    Main pipeline: GPS fishing points → H3 hexagonal aggregation.

    Args:
        df: DataFrame with GPS fishing track data
        lat_col, lon_col, time_col, trip_col: Column names (auto-detected)
        speed_col: Optional speed column
        fishing_col: Column with fishing classification (0/1)
        resolutions: List of H3 resolutions [7, 8]
        max_time_interval: Cap for time intervals in seconds
        min_hours, min_trips, min_days: Quality filter thresholds
        output_dir: Directory to save results
        save_parquet: Save as parquet (recommended for dashboard)
        save_csv: Also save as CSV

    Returns:
        Dictionary with all temporal aggregation results per resolution
    """

    print("=" * 70)
    print("H3 HEXAGONAL FISHING EFFORT AGGREGATION")
    print("=" * 70)

    if resolutions is None:
        resolutions = [H3_RES_REGIONAL, H3_RES_DETAIL]

    # ----- AUTO-DETECT COLUMNS (toolkit column_mapper) -----
    print("\n[1/6] Detecting columns...")
    lat_col = resolve_column_name(df, 'latitude', lat_col)
    lon_col = resolve_column_name(df, 'longitude', lon_col)
    time_col = resolve_column_name(df, 'timestamp', time_col)
    trip_col = resolve_column_name(df, 'trip_id', trip_col)
    speed_col = resolve_column_name(df, 'speed', speed_col, required=False)
    fishing_col = resolve_column_name(df, 'effort', fishing_col, required=False)

    print(f"  latitude:  {lat_col}")
    print(f"  longitude: {lon_col}")
    print(f"  timestamp: {time_col}")
    print(f"  trip_id:   {trip_col}")
    print(f"  speed:     {speed_col or 'not found (skipping speed metrics)'}")
    print(f"  fishing:   {fishing_col or 'not found (using all points)'}")

    # ----- FILTER TO FISHING POINTS -----
    print(f"\n[2/6] Filtering data...")
    print(f"  Total points: {len(df):,}")

    if fishing_col and fishing_col in df.columns:
        fishing_df = df[df[fishing_col] == 1].copy()
        print(f"  Fishing points (is_fishing=1): {len(fishing_df):,}")
    else:
        fishing_df = df.copy()
        print(f"  No fishing column found → using ALL points")

    if len(fishing_df) == 0:
        raise ValueError("No fishing points found! Check your fishing_col filter.")

    # ----- ADD TEMPORAL DIMENSIONS -----
    print(f"\n[3/6] Adding temporal dimensions...")
    fishing_df = prepare_temporal_dimensions(fishing_df, time_col)

    year_range = f"{fishing_df['year'].min()}-{fishing_df['year'].max()}"
    print(f"  Year range: {year_range}")
    print(f"  Seasons: {fishing_df['season'].value_counts().to_dict()}")
    print(f"  Day/Night: {fishing_df['day_night'].value_counts().to_dict()}")

    # ----- COMPUTE TIME INTERVALS -----
    print(f"\n[4/6] Computing time-weighted intervals...")
    print(f"  Max interval cap: {max_time_interval}s ({max_time_interval / 60:.0f} min)")
    fishing_df = compute_time_intervals(fishing_df, time_col, trip_col, max_time_interval)

    total_hours = fishing_df['time_hours'].sum()
    print(f"  Total fishing effort: {total_hours:,.1f} hours ({total_hours / 24:,.1f} days)")

    # ----- ASSIGN H3 HEXAGONS -----
    print(f"\n[5/6] Assigning H3 hexagons...")
    fishing_df = assign_h3_hexagons(fishing_df, lat_col, lon_col, resolutions)

    for res in resolutions:
        n_hex = fishing_df[f'h3_res{res}'].nunique()
        print(f"  Resolution {res}: {n_hex:,} unique hexagons")

    # ----- AGGREGATE PER RESOLUTION -----
    print(f"\n[6/6] Aggregating metrics...")

    all_results = {}

    for res in resolutions:
        h3_col = f'h3_res{res}'
        print(f"\n{'=' * 50}")
        print(f"  Resolution {res} ({h3_col})")
        print(f"{'=' * 50}")

        # Run all temporal aggregations
        temporal_results = aggregate_all_temporal(
            fishing_df, h3_col, trip_col, time_col, speed_col
        )

        # Apply quality filters to overall
        print(f"\n  Applying quality filters (min_hours={min_hours}, "
              f"min_trips={min_trips}, min_days={min_days})...")
        for key in temporal_results:
            if len(temporal_results[key]) > 0:
                temporal_results[key] = apply_quality_filters(
                    temporal_results[key], min_hours, min_trips, min_days
                )

        all_results[f'res{res}'] = temporal_results

    # ----- SAVE RESULTS -----
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"Saving results to {output_dir}")
        print(f"{'=' * 50}")

        for res_key, temporal_results in all_results.items():
            for temp_key, hex_df in temporal_results.items():
                if len(hex_df) == 0:
                    continue

                # Drop hex_boundary for serialization (it's a list of tuples)
                save_df = hex_df.drop(columns=['hex_boundary'], errors='ignore')

                filename = f"hex_{res_key}_{temp_key}"

                if save_parquet:
                    save_df.to_parquet(output_dir / f"{filename}.parquet", index=False)

                if save_csv:
                    save_df.to_csv(output_dir / f"{filename}.csv", index=False)

                print(f"  Saved {filename}: {len(save_df):,} rows")

    # ----- SUMMARY -----
    print(f"\n{'=' * 70}")
    print("AGGREGATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nTotal fishing effort: {total_hours:,.1f} hours")
    print(f"Resolutions processed: {resolutions}")
    print(f"\nResult keys per resolution:")
    for res_key in all_results:
        print(f"  {res_key}:")
        for temp_key, hex_df in all_results[res_key].items():
            print(f"    {temp_key}: {len(hex_df):,} hexagons")

    return all_results, fishing_df


# =============================================================================
# CONVENIENCE: QUICK SUMMARY STATS
# =============================================================================

def print_hex_summary(results, resolution='res7'):
    """Print a quick summary of the overall aggregation for a resolution."""

    if resolution not in results:
        print(f"Resolution {resolution} not found. Available: {list(results.keys())}")
        return

    overall = results[resolution].get('overall')
    if overall is None or len(overall) == 0:
        print("No overall data found.")
        return

    print(f"\n{'=' * 50}")
    print(f"HEXAGON SUMMARY ({resolution})")
    print(f"{'=' * 50}")
    print(f"Total hexagons: {len(overall):,}")
    print(f"\nFishing Hours per hexagon:")
    print(f"  Mean:   {overall['fishing_hours'].mean():.2f} hrs")
    print(f"  Median: {overall['fishing_hours'].median():.2f} hrs")
    print(f"  Max:    {overall['fishing_hours'].max():.2f} hrs")
    print(f"  Total:  {overall['fishing_hours'].sum():,.1f} hrs")
    print(f"\nTrips per hexagon:")
    print(f"  Mean:   {overall['n_trips'].mean():.1f}")
    print(f"  Median: {overall['n_trips'].median():.0f}")
    print(f"  Max:    {overall['n_trips'].max()}")
    print(f"\nTop 10 hotspot hexagons (by fishing hours):")
    top10 = overall.nlargest(10, 'fishing_hours')[
        ['lat', 'lon', 'fishing_hours', 'n_trips', 'n_days', 'vessel_days']
    ]
    print(top10.to_string(index=False))


# =============================================================================
# STEP 7: DASHBOARD-READY EXPORT
# =============================================================================

def export_for_dashboard(
        fishing_df,
        country_name,
        resolutions=None,
        trip_col='trip_id',
        time_col='timestamp',
        speed_col=None,
        output_dir='dashboard_data',
):
    """
    Export pre-aggregated hex data as JSON for the web dashboard.

    Creates one compact JSON file per country containing:
    - Hex boundaries (stored once per unique hex)
    - Finest-grain temporal data (h3 × year × month × day_night)
    - Dashboard aggregates client-side from this grain

    Args:
        fishing_df: DataFrame WITH h3 columns and temporal columns already added
                    (the second return value from run_hex_aggregation)
        country_name: Country/region identifier (e.g., 'kenya', 'zanzibar')
        resolutions: H3 resolutions to export (auto-detected from columns if None)
        trip_col, time_col, speed_col: Column names
        output_dir: Where to save the JSON file

    Returns:
        Path to saved JSON file

    Usage:
        results, fishing_df = run_hex_aggregation(df, resolutions=[7, 8, 9])
        export_for_dashboard(fishing_df, 'kenya', output_dir='dashboard_data')
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect resolutions from columns
    if resolutions is None:
        resolutions = sorted([
            int(c.replace('h3_res', ''))
            for c in fishing_df.columns if c.startswith('h3_res')
        ])

    if not resolutions:
        raise ValueError("No h3_res* columns found. Run assign_h3_hexagons first.")

    # Ensure temporal columns exist
    for col in ['year', 'month', 'day_night', 'season', 'date', 'time_hours']:
        if col not in fishing_df.columns:
            raise ValueError(
                f"Column '{col}' missing. Run prepare_temporal_dimensions and compute_time_intervals first.")

    print(f"\nExporting dashboard data for '{country_name}'...")
    print(f"  Resolutions: {resolutions}")
    print(f"  Points: {len(fishing_df):,}")

    export = {
        'meta': {
            'country': country_name,
            'exported_at': datetime.now().isoformat(),
            'resolutions': resolutions,
            'total_points': int(len(fishing_df)),
            'total_fishing_hours': round(float(fishing_df['time_hours'].sum()), 2),
            'year_range': [int(fishing_df['year'].min()), int(fishing_df['year'].max())],
            'years': sorted(fishing_df['year'].unique().astype(int).tolist()),
            'months': sorted(fishing_df['month'].unique().astype(int).tolist()),
            'seasons': sorted(fishing_df['season'].unique().tolist()),
            'day_night': sorted(fishing_df['day_night'].unique().tolist()),
        },
        'boundaries': {},
        'centers': {},
        'data': {},
    }

    for res in resolutions:
        h3_col = f'h3_res{res}'
        if h3_col not in fishing_df.columns:
            print(f"  ⚠ Column {h3_col} not found, skipping res {res}")
            continue

        res_key = f'res{res}'
        print(f"\n  Processing resolution {res}...")

        # --- Boundaries (once per unique hex) ---
        unique_hexes = fishing_df[h3_col].unique()
        boundaries = {}
        centers = {}
        for hex_id in unique_hexes:
            boundary = h3.cell_to_boundary(hex_id)
            boundaries[hex_id] = [[round(lat, 6), round(lon, 6)] for lat, lon in boundary]
            center = h3.cell_to_latlng(hex_id)
            centers[hex_id] = [round(center[0], 6), round(center[1], 6)]

        export['boundaries'][res_key] = boundaries
        export['centers'][res_key] = centers
        print(f"    Unique hexagons: {len(unique_hexes):,}")

        # --- Aggregate at finest grain: h3 × year × month × day_night ---
        group_cols = [h3_col, 'year', 'month', 'day_night']
        grouped = fishing_df.groupby(group_cols)

        agg_df = grouped['time_hours'].sum().reset_index()
        agg_df.columns = group_cols + ['fishing_hours']

        agg_df['n_points'] = grouped[time_col].count().values
        agg_df['n_trips'] = grouped[trip_col].nunique().values
        agg_df['n_days'] = grouped['date'].nunique().values

        # Vessel-days
        vd = fishing_df.groupby(group_cols).apply(
            lambda x: x[[trip_col, 'date']].drop_duplicates().shape[0],
            include_groups=False
        ).reset_index(name='vessel_days')
        agg_df = agg_df.merge(vd, on=group_cols, how='left')

        # Speed
        if speed_col and speed_col in fishing_df.columns:
            speed_agg = grouped[speed_col].mean().reset_index()
            speed_agg.columns = group_cols + ['mean_speed_kmh']
            agg_df = agg_df.merge(speed_agg, on=group_cols, how='left')

        # Season (derived from month)
        agg_df['season'] = agg_df['month'].map(SEASON_MAP)

        # Convert to compact records
        records = []
        for _, row in agg_df.iterrows():
            rec = {
                'h': row[h3_col],
                'y': int(row['year']),
                'm': int(row['month']),
                'dn': row['day_night'],
                's': row['season'],
                'fh': round(float(row['fishing_hours']), 4),
                'nt': int(row['n_trips']),
                'nd': int(row['n_days']),
                'vd': int(row['vessel_days']),
            }
            if 'mean_speed_kmh' in agg_df.columns and pd.notna(row.get('mean_speed_kmh')):
                rec['sp'] = round(float(row['mean_speed_kmh']), 2)
            records.append(rec)

        export['data'][res_key] = records
        print(f"    Data records: {len(records):,}")

    # --- Save JSON ---
    filename = f"dashboard_{country_name.lower().replace(' ', '_')}.json"
    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        json.dump(export, f, separators=(',', ':'))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Saved: {output_path} ({size_mb:.1f} MB)")

    return str(output_path)


# =============================================================================
# STEP 8: QUICK PREVIEW MAP (folium)
# =============================================================================

def plot_hex_map(
        hex_df,
        h3_col=None,
        metric='fishing_hours',
        title='Fishing Effort Heatmap',
        output_path='hex_preview_map.html',
        tiles='CartoDB dark_matter',
        colormap='YlOrRd',
        opacity=0.7,
        show_legend=True,
        zoom_start=None,
):
    """
    Quick preview map of hexagonal aggregation results using folium.

    No Mapbox token needed. Lightweight — uses pre-aggregated hexagons,
    not raw GPS points.

    Args:
        hex_df: Aggregated hexagon DataFrame (from aggregate_hexagon_metrics)
        h3_col: H3 column name (auto-detected if None)
        metric: Column to color by ('fishing_hours', 'n_trips', 'vessel_days', etc.)
        title: Map title
        output_path: Where to save HTML file
        tiles: Basemap tiles ('CartoDB dark_matter', 'CartoDB positron',
               'OpenStreetMap', 'Esri.WorldImagery')
        colormap: Color palette ('YlOrRd', 'Viridis', 'Blues', 'RdYlGn_r', 'Plasma')
        opacity: Hexagon fill opacity (0-1)
        show_legend: Show color legend
        zoom_start: Initial zoom (auto-calculated if None)

    Returns:
        Path to saved HTML file
    """
    try:
        import folium
        from folium.plugins import FloatImage
        import branca.colormap as cm
    except ImportError:
        raise ImportError("folium required: pip install folium branca")

    if len(hex_df) == 0:
        print("⚠ No hexagons to plot!")
        return None

    # Auto-detect h3 column
    if h3_col is None:
        h3_candidates = [c for c in hex_df.columns if c.startswith('h3_res')]
        if h3_candidates:
            h3_col = h3_candidates[0]
        else:
            raise ValueError(f"No H3 column found. Available: {hex_df.columns.tolist()}")

    # Validate metric column
    if metric not in hex_df.columns:
        available_metrics = [c for c in hex_df.columns
                             if c not in [h3_col, 'lat', 'lon', 'hex_boundary',
                                          'first_seen', 'last_seen']]
        raise ValueError(f"Metric '{metric}' not found. Available: {available_metrics}")

    print(f"Plotting {len(hex_df):,} hexagons colored by '{metric}'...")

    # ----- COLOR SCALE -----
    vmin = hex_df[metric].min()
    vmax = hex_df[metric].max()

    # Handle edge case where all values are the same
    if vmin == vmax:
        vmax = vmin + 1

    # Built-in colormaps
    _COLORMAPS = {
        'YlOrRd': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'],
        'Viridis': ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
        'Blues': ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'],
        'RdYlGn_r': ['#1a9850', '#91cf60', '#fee08b', '#fc8d59', '#d73027'],
        'Plasma': ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921'],
        'Turbo': ['#30123b', '#4662d7', '#36aab8', '#b5de2b', '#d23105'],
        'Ocean': ['#023858', '#045a8d', '#0570b0', '#3690c0', '#74a9cf', '#fd8d3c', '#e31a1c'],
    }

    colors = _COLORMAPS.get(colormap, _COLORMAPS['YlOrRd'])

    color_scale = cm.LinearColormap(
        colors=colors,
        vmin=vmin,
        vmax=vmax,
        caption=f'{metric.replace("_", " ").title()}'
    )

    # ----- CREATE MAP -----
    # Tile options that don't need tokens
    _TILE_MAP = {
        'CartoDB dark_matter': {
            'tiles': 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
            'attr': '&copy; CartoDB',
        },
        'CartoDB positron': {
            'tiles': 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            'attr': '&copy; CartoDB',
        },
        'Esri.WorldImagery': {
            'tiles': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            'attr': '&copy; Esri',
        },
    }

    center_lat = hex_df['lat'].mean()
    center_lon = hex_df['lon'].mean()

    if zoom_start is None:
        # Auto-calculate zoom based on data extent
        lat_range = hex_df['lat'].max() - hex_df['lat'].min()
        lon_range = hex_df['lon'].max() - hex_df['lon'].min()
        extent = max(lat_range, lon_range)
        if extent > 10:
            zoom_start = 5
        elif extent > 5:
            zoom_start = 7
        elif extent > 1:
            zoom_start = 9
        elif extent > 0.1:
            zoom_start = 11
        else:
            zoom_start = 13

    if tiles in _TILE_MAP:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles=_TILE_MAP[tiles]['tiles'],
            attr=_TILE_MAP[tiles]['attr'],
        )
    elif tiles == 'OpenStreetMap':
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap',
        )
    else:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles=tiles,
        )

    # ----- ADD TITLE -----
    title_html = f'''
    <div style="position:fixed; top:10px; left:50%; transform:translateX(-50%);
                z-index:1000; background:rgba(0,0,0,0.7); color:white;
                padding:8px 20px; border-radius:6px; font-size:16px;
                font-family:Arial,sans-serif; font-weight:bold;">
        {title}
        <span style="font-size:11px; font-weight:normal; margin-left:10px;">
            ({len(hex_df):,} hexagons)
        </span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # ----- DRAW HEXAGONS -----
    hex_group = folium.FeatureGroup(name='Fishing Effort Hexagons')

    for _, row in hex_df.iterrows():
        h3_id = row[h3_col]
        value = row[metric]

        # Get hexagon boundary
        if 'hex_boundary' in hex_df.columns and row['hex_boundary'] is not None:
            boundary = row['hex_boundary']
        else:
            boundary = h3.cell_to_boundary(h3_id)

        # Convert to folium format (list of [lat, lon])
        coords = [[lat, lon] for lat, lon in boundary]

        # Color
        hex_color = color_scale(value)

        # Tooltip with metrics
        tooltip_parts = [f"<b>{metric.replace('_', ' ').title()}:</b> {value:.2f}"]
        for col in ['n_trips', 'n_days', 'vessel_days', 'mean_speed_kmh', 'confidence']:
            if col in hex_df.columns and pd.notna(row.get(col)):
                label = col.replace('_', ' ').title()
                val = row[col]
                if isinstance(val, float):
                    tooltip_parts.append(f"<b>{label}:</b> {val:.1f}")
                else:
                    tooltip_parts.append(f"<b>{label}:</b> {val}")

        # Add temporal context if present
        for col in ['year', 'month_name', 'season', 'day_night']:
            if col in hex_df.columns and pd.notna(row.get(col)):
                label = col.replace('_', ' ').title()
                tooltip_parts.append(f"<b>{label}:</b> {row[col]}")

        tooltip_html = "<br>".join(tooltip_parts)

        folium.Polygon(
            locations=coords,
            color=hex_color,
            weight=1,
            fill=True,
            fill_color=hex_color,
            fill_opacity=opacity,
            tooltip=folium.Tooltip(tooltip_html),
        ).add_to(hex_group)

    hex_group.add_to(m)

    # ----- LEGEND -----
    if show_legend:
        color_scale.add_to(m)

    # ----- LAYER CONTROL -----
    folium.LayerControl(collapsed=False).add_to(m)

    # ----- SAVE -----
    output_path = Path(output_path)
    m.save(str(output_path))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved: {output_path} ({size_mb:.1f} MB)")

    return str(output_path)


def plot_hex_comparison(
        hex_df,
        h3_col=None,
        group_col='season',
        metric='fishing_hours',
        output_path='hex_comparison_map.html',
        tiles='CartoDB dark_matter',
        colormap='YlOrRd',
):
    """
    Side-by-side temporal comparison: one layer per group value, togglable.

    E.g., toggle between seasons, day/night, or years on the same map.

    Args:
        hex_df: Temporal aggregation DataFrame (e.g., by_season, by_day_night)
        h3_col: H3 column name (auto-detected)
        group_col: Column to create layers from ('season', 'day_night', 'year', etc.)
        metric: Metric to color by
        output_path: Where to save
        tiles: Basemap
        colormap: Color palette
    """
    try:
        import folium
        import branca.colormap as cm
    except ImportError:
        raise ImportError("folium required: pip install folium branca")

    if group_col not in hex_df.columns:
        raise ValueError(f"Group column '{group_col}' not found. Available: {hex_df.columns.tolist()}")

    # Auto-detect h3 column
    if h3_col is None:
        h3_candidates = [c for c in hex_df.columns if c.startswith('h3_res')]
        if h3_candidates:
            h3_col = h3_candidates[0]
        else:
            raise ValueError(f"No H3 column found.")

    groups = sorted(hex_df[group_col].unique())
    print(f"Creating comparison map with {len(groups)} layers: {groups}")

    # Global color scale
    vmin = hex_df[metric].min()
    vmax = hex_df[metric].max()
    if vmin == vmax:
        vmax = vmin + 1

    _COLORMAPS = {
        'YlOrRd': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'],
        'Viridis': ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
        'Blues': ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'],
        'Ocean': ['#023858', '#045a8d', '#0570b0', '#3690c0', '#74a9cf', '#fd8d3c', '#e31a1c'],
    }
    colors = _COLORMAPS.get(colormap, _COLORMAPS['YlOrRd'])

    color_scale = cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax,
                                    caption=f'{metric.replace("_", " ").title()}')

    # Create map
    center_lat = hex_df['lat'].mean()
    center_lon = hex_df['lon'].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
        if tiles == 'CartoDB dark_matter' else tiles,
        attr='&copy; CartoDB' if 'CartoDB' in tiles else '',
    )

    # Title
    title_html = f'''
    <div style="position:fixed; top:10px; left:50%; transform:translateX(-50%);
                z-index:1000; background:rgba(0,0,0,0.7); color:white;
                padding:8px 20px; border-radius:6px; font-size:16px;
                font-family:Arial,sans-serif; font-weight:bold;">
        Fishing Effort by {group_col.replace('_', ' ').title()}
        <span style="font-size:11px; font-weight:normal; margin-left:10px;">
            (toggle layers to compare)
        </span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # One layer per group
    for i, group_val in enumerate(groups):
        group_data = hex_df[hex_df[group_col] == group_val]
        fg = folium.FeatureGroup(name=f'{group_col}: {group_val} ({len(group_data)} hex)',
                                 show=(i == 0))  # Only first group visible by default

        for _, row in group_data.iterrows():
            boundary = h3.cell_to_boundary(row[h3_col])
            coords = [[lat, lon] for lat, lon in boundary]
            hex_color = color_scale(row[metric])

            tooltip = (f"<b>{group_col}:</b> {group_val}<br>"
                       f"<b>{metric}:</b> {row[metric]:.2f}<br>"
                       f"<b>Trips:</b> {row.get('n_trips', 'N/A')}<br>"
                       f"<b>Days:</b> {row.get('n_days', 'N/A')}")

            folium.Polygon(
                locations=coords, color=hex_color, weight=1,
                fill=True, fill_color=hex_color, fill_opacity=0.7,
                tooltip=folium.Tooltip(tooltip),
            ).add_to(fg)

        fg.add_to(m)

    color_scale.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    output_path = Path(output_path)
    m.save(str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved: {output_path} ({size_mb:.1f} MB)")

    return str(output_path)
