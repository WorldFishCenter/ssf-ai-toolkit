# src/ssfaitk/utils/column_mapper.py
"""
Column name mapping utilities for SSF AI Toolkit.

Handles different naming conventions for common columns across datasets.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set
import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Standard Column Name Mappings
# ============================================================================

COLUMN_ALIASES = {
    # Spatial coordinates
    'latitude': ['latitude', 'Latitude', 'lat', 'Lat', 'LAT', 'LATITUDE', 'y', 'Y'],
    'longitude': ['longitude', 'Longitude', 'lon', 'Lon', 'Lng', 'LON', 'LONGITUDE', 'lng', 'x', 'X'],
    
    # Trip/voyage identifiers
    'trip_id': ['trip_id', 'Trip_ID', 'TRIP_ID', 'tripid', 'TripID','Trip'],
    
    # Temporal
    'timestamp': ['timestamp', 'Timestamp', 'time', 'Time', 'TIME', 'datetime', 
                  'ltime', 'date_time', 'DateTime', 'date'],
    
    # Speed
    'speed': ['speed', 'Speed', 'SPEED', 'sog', 'SOG', 'speed_kmh', 'speed_kn', 
              'velocity', 'Velocity', 'Speed (M/S)'],

    # Heading/course
    'heading': ['heading', 'Heading', 'HEADING', 'cog', 'COG', 'course', 
                'Course', 'bearing', 'Bearing'],
    
    # Altitude/elevation
    'altitude': ['altitude', 'Altitude', 'ALT', 'elevation', 'Elevation', 'height'],
    
    # Effort/activity
    'effort': ['effort_pred', 'effort', 'is_fishing', 'is_Fishing', 'IsFishing',
               'fishing', 'Fishing', 'Activity', 'activity', 'effort_class'],
    
    # Vessel information
    'vessel_id': ['vessel_id', 'Vessel_ID', 'VesselID', 'boat_id', 'BoatID', 
                  'fisher_id', 'FisherID'],
    'vessel_type': ['vessel_type', 'VesselType', 'boat_type', 'BoatType'],
    
    # Gear
    'gear_type': ['gear_type', 'GearType', 'gear', 'Gear', 'GEAR'],
    
    # Device/model
    'device_model': ['model', 'Model', 'device_model', 'DeviceModel', 'device'],
    
    # Catch
    'catch_weight': ['catch_weight', 'CatchWeight', 'weight', 'Weight', 'catch_kg'],
    'species': ['species', 'Species', 'SPECIES', 'fish_species', 'FishSpecies'],
}


# ============================================================================
# Column Detection and Mapping
# ============================================================================

def find_column(
    df: pd.DataFrame,
    standard_name: str,
    aliases: Optional[List[str]] = None,
    raise_if_missing: bool = False
) -> Optional[str]:
    """
    Find a column in DataFrame using standard name and known aliases.
    
    Args:
        df: DataFrame to search
        standard_name: Standard column name (key in COLUMN_ALIASES)
        aliases: Optional custom aliases to check (in addition to defaults)
        raise_if_missing: Raise error if column not found
    
    Returns:
        Actual column name found in DataFrame, or None if not found
    
    Raises:
        ValueError: If raise_if_missing=True and column not found
    
    Example:
        >>> df = pd.DataFrame({'Latitude': [1, 2], 'Longitude': [3, 4]})
        >>> find_column(df, 'latitude')
        'Latitude'
        >>> find_column(df, 'latitude', aliases=['custom_lat'])
        'Latitude'
    """
    # Get default aliases for this standard name
    default_aliases = COLUMN_ALIASES.get(standard_name, [standard_name])
    
    # Combine with custom aliases if provided
    all_aliases = list(default_aliases)
    if aliases:
        all_aliases.extend(aliases)
    
    # Remove duplicates while preserving order
    seen: Set[str] = set()
    unique_aliases = []
    for alias in all_aliases:
        if alias not in seen:
            seen.add(alias)
            unique_aliases.append(alias)
    
    # Search for first match
    for alias in unique_aliases:
        if alias in df.columns:
            logger.debug(f"Found '{standard_name}' as '{alias}'")
            return alias
    
    # Not found
    if raise_if_missing:
        raise ValueError(
            f"Column '{standard_name}' not found in DataFrame. "
            f"Searched for: {unique_aliases}. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    logger.debug(f"Column '{standard_name}' not found (searched: {unique_aliases[:3]}...)")
    return None


def map_columns(
    df: pd.DataFrame,
    column_map: Dict[str, str | List[str]],
    rename: bool = False,
    drop_unmapped: bool = False
) -> pd.DataFrame:
    """
    Map DataFrame columns to standard names.
    
    Args:
        df: DataFrame to map
        column_map: Dict of {standard_name: aliases_to_search}
        rename: If True, rename columns to standard names
        drop_unmapped: If True, drop columns not in mapping
    
    Returns:
        DataFrame with mapped/renamed columns
    
    Example:
        >>> df = pd.DataFrame({'Latitude': [1], 'Longitude': [2], 'Extra': [3]})
        >>> column_map = {'latitude': 'latitude', 'longitude': 'longitude'}
        >>> df_mapped = map_columns(df, column_map, rename=True)
        >>> list(df_mapped.columns)
        ['latitude', 'longitude', 'Extra']
    """
    df = df.copy()
    
    rename_dict = {}
    
    for standard_name, search_cols in column_map.items():
        # Convert single string to list
        if isinstance(search_cols, str):
            search_cols = [search_cols]
        
        # Find column
        found_col = find_column(df, standard_name, aliases=search_cols)
        
        if found_col and rename:
            rename_dict[found_col] = standard_name
    
    # Apply renaming
    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.info(f"Renamed columns: {rename_dict}")
    
    # Drop unmapped columns if requested
    if drop_unmapped:
        keep_cols = list(column_map.keys()) if rename else list(rename_dict.keys())
        existing_keep = [c for c in keep_cols if c in df.columns]
        df = df[existing_keep]
        logger.info(f"Kept only mapped columns: {existing_keep}")
    
    return df


def get_column_mapping(
    df: pd.DataFrame,
    required: Optional[List[str]] = None,
    optional: Optional[List[str]] = None,
    custom_aliases: Optional[Dict[str, List[str]]] = None
) -> Dict[str, str]:
    """
    Get mapping of standard names to actual column names in DataFrame.
    
    Args:
        df: DataFrame to map
        required: List of required standard column names
        optional: List of optional standard column names
        custom_aliases: Custom aliases to check (merged with defaults)
    
    Returns:
        Dict mapping standard names to actual column names
    
    Raises:
        ValueError: If required columns are missing
    
    Example:
        >>> df = pd.DataFrame({'Latitude': [1], 'lon': [2]})
        >>> mapping = get_column_mapping(df, required=['latitude', 'longitude'])
        >>> mapping
        {'latitude': 'Latitude', 'longitude': 'lon'}
    """
    mapping = {}
    missing_required = []
    
    # Process required columns
    if required:
        for std_name in required:
            custom = custom_aliases.get(std_name) if custom_aliases else None
            found = find_column(df, std_name, aliases=custom)
            
            if found:
                mapping[std_name] = found
            else:
                missing_required.append(std_name)
    
    # Process optional columns
    if optional:
        for std_name in optional:
            custom = custom_aliases.get(std_name) if custom_aliases else None
            found = find_column(df, std_name, aliases=custom)
            
            if found:
                mapping[std_name] = found
    
    # Check for missing required columns
    if missing_required:
        raise ValueError(
            f"Required columns not found: {missing_required}. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    logger.info(f"Column mapping: {mapping}")
    return mapping


def standardize_columns(
    df: pd.DataFrame,
    required: Optional[List[str]] = None,
    optional: Optional[List[str]] = None,
    custom_aliases: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Standardize DataFrame column names to canonical names.
    
    Renames columns to standard names (e.g., 'Latitude' -> 'latitude').
    
    Args:
        df: DataFrame to standardize
        required: List of required standard column names
        optional: List of optional standard column names
        custom_aliases: Custom aliases to check
    
    Returns:
        DataFrame with standardized column names
    
    Example:
        >>> df = pd.DataFrame({'Latitude': [1], 'lon': [2]})
        >>> df_std = standardize_columns(df, required=['latitude', 'longitude'])
        >>> list(df_std.columns)
        ['latitude', 'longitude']
    """
    mapping = get_column_mapping(df, required, optional, custom_aliases)
    
    # Create rename dict (actual -> standard)
    rename_dict = {actual: standard for standard, actual in mapping.items()}
    
    df_std = df.rename(columns=rename_dict)
    logger.info(f"Standardized {len(rename_dict)} columns")
    
    return df_std


# ============================================================================
# Convenience Functions for Common Use Cases
# ============================================================================

def get_spatial_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get latitude and longitude column names.
    
    Returns:
        Dict with 'latitude' and 'longitude' keys mapped to actual columns
    
    Raises:
        ValueError: If lat/lon columns not found
    """
    return get_column_mapping(df, required=['latitude', 'longitude'])


def get_temporal_column(df: pd.DataFrame) -> str:
    """
    Get timestamp/datetime column name.
    
    Returns:
        Actual column name for timestamp
    
    Raises:
        ValueError: If timestamp column not found
    """
    mapping = get_column_mapping(df, required=['timestamp'])
    return mapping['timestamp']


def get_trip_column(df: pd.DataFrame) -> str:
    """
    Get trip ID column name.
    
    Returns:
        Actual column name for trip_id
    
    Raises:
        ValueError: If trip_id column not found
    """
    mapping = get_column_mapping(df, required=['trip_id'])
    return mapping['trip_id']


def get_effort_column(df: pd.DataFrame) -> Optional[str]:
    """
    Get effort/fishing activity column name.
    
    Returns:
        Actual column name for effort, or None if not found
    """
    return find_column(df, 'effort')


# ============================================================================
# Helper for Function Parameters
# ============================================================================

def resolve_column_name(
    df: pd.DataFrame,
    standard_name: str,
    user_provided: Optional[str] = None,
    custom_aliases: Optional[List[str]] = None,
    required: bool = True
) -> str:
    """
    Resolve actual column name from user input or auto-detection.
    
    Priority:
    1. User-provided column name (if given and exists)
    2. Auto-detected from standard aliases
    3. Raise error (if required) or return None
    
    Args:
        df: DataFrame
        standard_name: Standard column name
        user_provided: Column name explicitly provided by user
        custom_aliases: Additional aliases to search
        required: Whether column is required
    
    Returns:
        Actual column name
    
    Example:
        >>> df = pd.DataFrame({'Latitude': [1]})
        >>> resolve_column_name(df, 'latitude')
        'Latitude'
        >>> resolve_column_name(df, 'latitude', user_provided='Latitude')
        'Latitude'
    """
    # 1. Check user-provided name first
    if user_provided:
        if user_provided in df.columns:
            logger.debug(f"Using user-provided column: '{user_provided}' for '{standard_name}'")
            return user_provided
        else:
            raise ValueError(
                f"User-provided column '{user_provided}' not found in DataFrame. "
                f"Available: {df.columns.tolist()}"
            )
    
    # 2. Auto-detect
    found = find_column(df, standard_name, aliases=custom_aliases, raise_if_missing=required)
    
    if found:
        return found
    
    if required:
        raise ValueError(
            f"Required column '{standard_name}' not found. "
            f"Please specify using the '{standard_name}_col' parameter."
        )
    
    return None


# ============================================================================
# Validation
# ============================================================================

def validate_columns(
    df: pd.DataFrame,
    required: List[str],
    optional: Optional[List[str]] = None,
    custom_aliases: Optional[Dict[str, List[str]]] = None
) -> Dict[str, str]:
    """
    Validate that DataFrame has required columns and return mapping.
    
    Args:
        df: DataFrame to validate
        required: List of required standard column names
        optional: List of optional standard column names
        custom_aliases: Custom aliases
    
    Returns:
        Mapping of standard names to actual columns
    
    Raises:
        ValueError: If required columns missing
    """
    return get_column_mapping(df, required, optional, custom_aliases)


# ============================================================================
# Add Custom Aliases Globally
# ============================================================================

def add_global_alias(standard_name: str, alias: str) -> None:
    """
    Add a custom alias to the global alias dictionary.
    
    Args:
        standard_name: Standard column name
        alias: New alias to add
    
    Example:
        >>> add_global_alias('latitude', 'custom_lat')
        >>> # Now 'custom_lat' will be searched when looking for latitude
    """
    if standard_name not in COLUMN_ALIASES:
        COLUMN_ALIASES[standard_name] = []
    
    if alias not in COLUMN_ALIASES[standard_name]:
        COLUMN_ALIASES[standard_name].append(alias)
        logger.info(f"Added global alias: '{alias}' -> '{standard_name}'")


def register_aliases(aliases: Dict[str, List[str]]) -> None:
    """
    Register multiple custom aliases at once.
    
    Args:
        aliases: Dict of {standard_name: [list_of_aliases]}
    
    Example:
        >>> register_aliases({
        ...     'latitude': ['custom_lat', 'my_lat'],
        ...     'longitude': ['custom_lon', 'my_lon']
        ... })
    """
    for standard_name, alias_list in aliases.items():
        for alias in alias_list:
            add_global_alias(standard_name, alias)
