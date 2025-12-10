# src/ssfaitk/models/effort/shore_distance_filter.py
"""
Distance-to-Shore Filtering for Fishing Effort Classification

Prevents misclassification of vessels near shore (harbors, landing sites, 
fish markets) as fishing activity.

Provides multiple methods:
1. Harbor-based: Distance to known harbor/landing site coordinates
2. Coastline-based: Distance to coastline using shapely/geopandas
3. Depth-based: Using bathymetry data (if available)
4. Exclusion zones: Predefined polygons for ports/harbors
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===============================================================
# Distance Calculation Utilities
# ===============================================================

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Calculate great circle distance in kilometers."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def distance_to_points(
    lats: np.ndarray,
    lons: np.ndarray,
    reference_points: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Calculate minimum distance to a set of reference points.
    
    Args:
        lats: Array of latitudes
        lons: Array of longitudes
        reference_points: List of (lat, lon) tuples for harbors/landing sites
    
    Returns:
        Array of minimum distances in kilometers
    """
    if len(reference_points) == 0:
        return np.full(len(lats), np.inf)
    
    # Calculate distance to each reference point
    distances = np.array([
        haversine_distance(lats, lons, ref_lat, ref_lon)
        for ref_lat, ref_lon in reference_points
    ])
    
    # Return minimum distance to any reference point
    return np.min(distances, axis=0)


# ===============================================================
# Method 1: Simple Harbor/Landing Site Distance
# ===============================================================

class HarborDistanceFilter:
    """
    Simple distance-based filter using known harbor/landing site locations.
    
    Best for: Quick implementation, known harbor locations
    Limitations: Doesn't account for full coastline
    """
    
    def __init__(
        self,
        harbor_coords: List[Tuple[float, float]],
        min_distance_km: float = 1.0,
        buffer_distance_km: float = 0.5,
    ):
        """
        Initialize harbor distance filter.
        
        Args:
            harbor_coords: List of (lat, lon) tuples for harbors/landing sites
            min_distance_km: Minimum distance from harbor to allow fishing (default: 1 km)
            buffer_distance_km: Buffer zone for gradual transition (default: 0.5 km)
        """
        self.harbor_coords = harbor_coords
        self.min_distance_km = min_distance_km
        self.buffer_distance_km = buffer_distance_km
        
        logger.info(f"Harbor filter initialized with {len(harbor_coords)} reference points")
        logger.info(f"  Min distance: {min_distance_km} km")
        logger.info(f"  Buffer zone: {buffer_distance_km} km")
    
    def compute_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add distance-to-harbor column.
        
        Args:
            df: DataFrame with latitude and longitude columns
        
        Returns:
            DataFrame with 'dist_to_shore_km' column added
        """
        df = df.copy()
        
        lats = df['latitude'].values
        lons = df['longitude'].values
        
        df['dist_to_shore_km'] = distance_to_points(lats, lons, self.harbor_coords)
        
        return df
    
    def apply_filter(
        self,
        df: pd.DataFrame,
        effort_col: str = 'is_fishing',
        score_col: str = 'fishing_score'
    ) -> pd.DataFrame:
        """
        Filter out fishing predictions near harbors.
        
        Args:
            df: DataFrame with fishing predictions
            effort_col: Column with binary fishing predictions
            score_col: Column with fishing scores (if available)
        
        Returns:
            DataFrame with filtered predictions
        """
        # Compute distances if not already present
        if 'dist_to_shore_km' not in df.columns:
            df = self.compute_distances(df)
        
        # Identify near-shore points
        df['is_near_shore'] = (df['dist_to_shore_km'] < self.min_distance_km).astype(int)
        
        # Create filtered prediction
        df[f'{effort_col}_filtered'] = df[effort_col].copy()
        
        # Override fishing classification for near-shore points
        near_shore_mask = df['is_near_shore'] == 1
        df.loc[near_shore_mask, f'{effort_col}_filtered'] = 0
        
        # Reduce fishing score in buffer zone
        if score_col in df.columns:
            buffer_mask = (
                (df['dist_to_shore_km'] >= self.min_distance_km) &
                (df['dist_to_shore_km'] < self.min_distance_km + self.buffer_distance_km)
            )
            
            if buffer_mask.any():
                # Gradual reduction in buffer zone
                reduction_factor = (
                    (df.loc[buffer_mask, 'dist_to_shore_km'] - self.min_distance_km) /
                    self.buffer_distance_km
                )
                df.loc[buffer_mask, f'{score_col}_filtered'] = (
                    df.loc[buffer_mask, score_col] * reduction_factor
                )
        
        # Log statistics
        n_filtered = (df[effort_col] == 1) & (df[f'{effort_col}_filtered'] == 0)
        n_filtered = n_filtered.sum()
        
        if n_filtered > 0:
            logger.info(f"Shore filter: {n_filtered} points reclassified from fishing to non-fishing")
            logger.info(f"  ({100 * n_filtered / len(df):.2f}% of total points)")
        
        return df


# ===============================================================
# Method 2: Coastline-Based Distance (Requires Shapely)
# ===============================================================

class CoastlineDistanceFilter:
    """
    Calculate distance to coastline using vector coastline data.
    
    Best for: Accurate distance to any land
    Requirements: shapely, geopandas, coastline shapefile
    """

    def __init__(
            self,
            coastline_shapefile: Optional[str] = None,
            min_distance_km: float = 1.0,
            buffer_distance_km: float = 0.5,
    ):
        """
        Initialize coastline distance filter.

        Args:
            coastline_shapefile: Path to coastline shapefile
            min_distance_km: Minimum distance from coast
            buffer_distance_km: Buffer zone width
        """
        self.coastline_shapefile = coastline_shapefile
        self.min_distance_km = min_distance_km
        self.buffer_distance_km = buffer_distance_km
        self.coastline_geom = None

        if self.coastline_shapefile:
            try:
                import geopandas as gpd
                from shapely.geometry import MultiLineString, LineString

                logger.info(f"Loading coastline from: {self.coastline_shapefile}")
                gdf = gpd.read_file(self.coastline_shapefile)

                # Handle different geometry types
                lines = []
                for geom in gdf.geometry:
                    if geom.geom_type == 'LineString':
                        # Already a line - use directly
                        lines.append(geom)
                    elif geom.geom_type == 'MultiLineString':
                        # Multiple lines - add all
                        lines.extend(geom.geoms)
                    elif geom.geom_type == 'Polygon':
                        # Polygon - extract boundary (exterior ring)
                        lines.append(geom.exterior)
                    elif geom.geom_type == 'MultiPolygon':
                        # Multiple polygons - extract all boundaries
                        for poly in geom.geoms:
                            lines.append(poly.exterior)
                    else:
                        logger.warning(f"Skipping unsupported geometry type: {geom.geom_type}")

                if len(lines) == 0:
                    raise ValueError("No valid coastline geometries found")

                # Combine all lines
                self.coastline_geom = MultiLineString(lines)
                logger.info(f"✓ Coastline loaded successfully")
                logger.info(f"  Extracted {len(lines)} coastline segments")

            except ImportError:
                logger.warning("geopandas not installed")
            except Exception as e:
                logger.error(f"Failed to load coastline: {e}")
                raise
    def compute_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate distance to coastline for each point.
        
        Args:
            df: DataFrame with latitude/longitude
        
        Returns:
            DataFrame with dist_to_shore_km column
        """
        if self.coastline_geom is None:
            raise ValueError("No coastline data loaded")
        
        from shapely.geometry import Point
        
        df = df.copy()
        
        # Calculate distance for each point
        distances = []
        for _, row in df.iterrows():
            point = Point(row['longitude'], row['latitude'])
            # Distance in degrees, approximate conversion to km
            dist_deg = point.distance(self.coastline_geom)
            dist_km = dist_deg * 111.0  # Rough conversion at equator
            distances.append(dist_km)
        
        df['dist_to_shore_km'] = distances
        
        return df
    
    def apply_filter(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply coastline-based filtering (same as HarborDistanceFilter)."""
        # Reuse the logic from HarborDistanceFilter
        if 'dist_to_shore_km' not in df.columns:
            df = self.compute_distances(df)
        
        # Use same filtering logic
        filter_obj = HarborDistanceFilter([], self.min_distance_km, self.buffer_distance_km)
        return filter_obj.apply_filter(df, **kwargs)


# ===============================================================
# Method 3: Simple Grid-Based Approach
# ===============================================================

class GridBasedShoreFilter:
    """
    Fast approximate shore filtering using a pre-computed distance grid.
    
    Best for: Very fast filtering, large datasets
    Trade-off: Less accurate than point-based methods
    """
    
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],  # (min_lat, max_lat, min_lon, max_lon)
        shore_points: List[Tuple[float, float]],
        grid_resolution: float = 0.01,  # degrees (~1 km)
        min_distance_km: float = 1.0,
    ):
        """
        Initialize grid-based filter.
        
        Args:
            bbox: Bounding box (min_lat, max_lat, min_lon, max_lon)
            shore_points: List of (lat, lon) points representing shore
            grid_resolution: Grid cell size in degrees
            min_distance_km: Minimum distance threshold
        """
        self.bbox = bbox
        self.grid_resolution = grid_resolution
        self.min_distance_km = min_distance_km
        
        logger.info("Building distance grid...")
        self.distance_grid = self._build_distance_grid(shore_points)
        logger.info("Distance grid ready")
    
    def _build_distance_grid(self, shore_points: List[Tuple[float, float]]) -> np.ndarray:
        """Pre-compute distance grid."""
        min_lat, max_lat, min_lon, max_lon = self.bbox
        
        # Create grid
        lat_bins = np.arange(min_lat, max_lat, self.grid_resolution)
        lon_bins = np.arange(min_lon, max_lon, self.grid_resolution)
        
        # Calculate distance for each grid cell
        distance_grid = np.zeros((len(lat_bins), len(lon_bins)))
        
        for i, lat in enumerate(lat_bins):
            for j, lon in enumerate(lon_bins):
                min_dist = min([
                    haversine_distance(lat, lon, shore_lat, shore_lon)
                    for shore_lat, shore_lon in shore_points
                ])
                distance_grid[i, j] = min_dist
        
        return distance_grid
    
    def get_distance(self, lat: float, lon: float) -> float:
        """Look up distance from grid."""
        min_lat, max_lat, min_lon, max_lon = self.bbox
        
        # Find grid cell
        i = int((lat - min_lat) / self.grid_resolution)
        j = int((lon - min_lon) / self.grid_resolution)
        
        # Check bounds
        if i < 0 or i >= self.distance_grid.shape[0] or \
           j < 0 or j >= self.distance_grid.shape[1]:
            return np.inf
        
        return self.distance_grid[i, j]
    
    def apply_filter(self, df: pd.DataFrame, effort_col: str = 'is_fishing') -> pd.DataFrame:
        """Apply grid-based filtering."""
        df = df.copy()
        
        # Lookup distances
        df['dist_to_shore_km'] = df.apply(
            lambda row: self.get_distance(row['latitude'], row['longitude']),
            axis=1
        )
        
        # Filter
        df['is_near_shore'] = (df['dist_to_shore_km'] < self.min_distance_km).astype(int)
        df[f'{effort_col}_filtered'] = df[effort_col].copy()
        df.loc[df['is_near_shore'] == 1, f'{effort_col}_filtered'] = 0
        
        return df


# ===============================================================
# Convenience Function for Common Regions
# ===============================================================

# Pre-defined harbor/landing site coordinates for common regions
REGIONAL_HARBORS = {
    'zanzibar': [
        (-6.1659, 39.1959),  # Stone Town
        (-6.1435, 39.1937),  # Malindi
        (-6.2285, 39.1835),  # Mkokotoni
        (-6.0755, 39.2992),  # Nungwi
        (-6.4760, 39.5068),  # Paje
    ],
    'kenya': [
        (-4.0435, 39.6682),  # Mombasa
        (-3.3693, 39.9618),  # Malindi
        (-4.6633, 39.6656),  # Shimoni
        (-2.2827, 40.1167),  # Lamu
    ],
    'timor_leste': [
        (-8.5569, 125.5603),  # Dili
        (-8.4754, 125.6432),  # Cristo Rei
        (-8.8390, 125.9558),  # Baucau
    ],
    'tanzania_mainland': [
        (-6.7924, 39.2083),  # Dar es Salaam
        (-6.8160, 39.2803),  # Kigamboni
        (-10.2796, 40.1963), # Mtwara
    ],
}


def create_shore_filter(
    region: str,
    method: str = 'harbor',
    min_distance_km: float = 1.0,
    custom_harbors: Optional[List[Tuple[float, float]]] = None,
    **kwargs
) -> HarborDistanceFilter:
    """
    Create shore distance filter for a specific region.
    
    Args:
        region: Region name ('zanzibar', 'kenya', 'timor_leste', 'tanzania_mainland')
        method: Filter method ('harbor', 'coastline', 'grid')
        min_distance_km: Minimum distance threshold
        custom_harbors: Optional custom harbor coordinates
        **kwargs: Additional filter-specific parameters
    
    Returns:
        Configured shore filter
    
    Example:
        >>> filter = create_shore_filter('zanzibar', min_distance_km=0.5)
        >>> df = filter.compute_distances(df)
        >>> df = filter.apply_filter(df)
    """
    if custom_harbors:
        harbor_coords = custom_harbors
    elif region.lower() in REGIONAL_HARBORS:
        harbor_coords = REGIONAL_HARBORS[region.lower()]
        logger.info(f"Using {len(harbor_coords)} pre-defined harbors for {region}")
    else:
        raise ValueError(
            f"Unknown region: {region}. "
            f"Available: {list(REGIONAL_HARBORS.keys())} or provide custom_harbors"
        )
    
    if method == 'harbor':
        return HarborDistanceFilter(
            harbor_coords,
            min_distance_km=min_distance_km,
            **kwargs
        )
    elif method == 'coastline':
        return CoastlineDistanceFilter(
            min_distance_km=min_distance_km,
            **kwargs
        )
    elif method == 'grid':
        # Need bbox for grid method
        if 'bbox' not in kwargs:
            raise ValueError("Grid method requires 'bbox' parameter")
        return GridBasedShoreFilter(
            bbox=kwargs['bbox'],
            shore_points=harbor_coords,
            min_distance_km=min_distance_km,
            **kwargs.get('grid_resolution', 0.01)
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# ===============================================================
# Integration with Statistical Classifier
# ===============================================================

def add_shore_filtering(
    df: pd.DataFrame,
    region: str,
    min_distance_km: float = 1.0,
    effort_col: str = 'is_fishing',
    custom_harbors: Optional[List[Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Convenience function to add shore filtering to classifier results.
    
    Args:
        df: DataFrame with fishing predictions
        region: Region name or 'custom' if using custom_harbors
        min_distance_km: Minimum distance from shore
        effort_col: Column with fishing predictions
        custom_harbors: Optional custom harbor coordinates
    
    Returns:
        DataFrame with filtered predictions
    
    Example:
        >>> # After classification
        >>> df = classifier.predict(df)
        >>> 
        >>> # Add shore filtering
        >>> df = add_shore_filtering(df, region='zanzibar', min_distance_km=0.5)
        >>> 
        >>> # Use filtered predictions
        >>> print(df['is_fishing_filtered'].value_counts())
    """
    shore_filter = create_shore_filter(
        region=region,
        min_distance_km=min_distance_km,
        custom_harbors=custom_harbors
    )
    
    df = shore_filter.compute_distances(df)
    df = shore_filter.apply_filter(df, effort_col=effort_col)
    
    return df
