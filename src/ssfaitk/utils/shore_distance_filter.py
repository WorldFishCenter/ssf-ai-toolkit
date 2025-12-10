# src/ssfaitk/models/effort/shore_distance_filter.py
"""
Distance-to-Shore Filtering for Fishing Effort Classification

ENHANCED VERSION with:
1. Regional cropping (10-100x faster!) ⚡
2. Land polygon filtering (removes points ON land) 🏝️
3. Column auto-detection (via column_mapper.py) 🔍

Prevents misclassification of vessels near shore (harbors, landing sites, 
fish markets) as fishing activity.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ===============================================================
# Regional Bounding Boxes for Cropping
# ===============================================================

REGIONAL_BBOXES = {
    'zanzibar': (-7.0, -4.5, 38.5, 40.5),
    'kenya': (-5.0, -1.5, 39.0, 42.0),
    'tanzania': (-11.5, -0.5, 38.5, 41.0),
    'mozambique': (-27.0, -10.0, 32.0, 41.0),
    'timor_leste': (-9.5, -8.0, 124.0, 127.5),
}

REGIONAL_HARBORS = {
    'zanzibar': [(-6.1659, 39.1959), (-6.1435, 39.1937), (-6.2285, 39.1835), (-6.0755, 39.2992), (-6.4760, 39.5068)],
    'kenya': [(-4.0435, 39.6682), (-3.3693, 39.9618), (-4.6633, 39.6656), (-2.2827, 40.1167)],
    'timor_leste': [(-8.5569, 125.5603), (-8.4754, 125.6432), (-8.8390, 125.9558)],
    'tanzania_mainland': [(-6.7924, 39.2083), (-6.8160, 39.2803), (-10.2796, 40.1963)],
    'mozambique': [(-25.9692, 32.5732), (-23.8607, 35.5296), (-18.6657, 35.5296)],
}


# ===============================================================
# Column Auto-Detection (with column_mapper.py integration)
# ===============================================================

def get_column_names(
        df: pd.DataFrame,
        lat_col: Optional[str] = None,
        lon_col: Optional[str] = None,
        effort_col: Optional[str] = None
) -> Dict[str, str]:
    """Auto-detect column names using column_mapper if available."""
    result = {}

    # Try column_mapper first
    try:
        from ssfaitk.utils.column_mapper import detect_column
        use_mapper = True
    except ImportError:
        use_mapper = False

    # Latitude
    if lat_col and lat_col in df.columns:
        result['lat'] = lat_col
    elif use_mapper:
        result['lat'] = detect_column(df, 'latitude')
    else:
        for col in ['latitude', 'Latitude', 'lat', 'LAT']:
            if col in df.columns:
                result['lat'] = col
                break

    # Longitude
    if lon_col and lon_col in df.columns:
        result['lon'] = lon_col
    elif use_mapper:
        result['lon'] = detect_column(df, 'longitude')
    else:
        for col in ['longitude', 'Longitude', 'lon', 'LON']:
            if col in df.columns:
                result['lon'] = col
                break

    # Effort (optional)
    if effort_col and effort_col in df.columns:
        result['effort'] = effort_col
    elif use_mapper:
        try:
            result['effort'] = detect_column(df, 'effort')
        except:
            result['effort'] = None
    else:
        for col in ['effort_pred', 'is_fishing', 'fishing']:
            if col in df.columns:
                result['effort'] = col
                break
        else:
            result['effort'] = None

    if not result.get('lat'):
        raise ValueError("Could not find latitude column")
    if not result.get('lon'):
        raise ValueError("Could not find longitude column")

    return result


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Calculate great circle distance in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def distance_to_points(lats: np.ndarray, lons: np.ndarray, reference_points: List[Tuple[float, float]]) -> np.ndarray:
    """Calculate minimum distance to reference points."""
    if len(reference_points) == 0:
        return np.full(len(lats), np.inf)
    distances = np.array([haversine_distance(lats, lons, ref_lat, ref_lon) for ref_lat, ref_lon in reference_points])
    return np.min(distances, axis=0)


# ===============================================================
# Harbor Distance Filter
# ===============================================================

class HarborDistanceFilter:
    """Simple harbor-based distance filter."""

    def __init__(self, harbor_coords: List[Tuple[float, float]], min_distance_km: float = 1.0,
                 buffer_distance_km: float = 0.5):
        self.harbor_coords = harbor_coords
        self.min_distance_km = min_distance_km
        self.buffer_distance_km = buffer_distance_km
        logger.info(f"Harbor filter: {len(harbor_coords)} points, {min_distance_km} km threshold")

    def compute_distances(self, df: pd.DataFrame, lat_col: Optional[str] = None,
                          lon_col: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        cols = get_column_names(df, lat_col, lon_col)
        df['dist_to_shore_km'] = distance_to_points(df[cols['lat']].values, df[cols['lon']].values, self.harbor_coords)
        return df

    def apply_filter(self, df: pd.DataFrame, effort_col: Optional[str] = None, **kwargs) -> pd.DataFrame:
        cols = get_column_names(df, effort_col=effort_col)
        if not cols['effort']:
            raise ValueError("Could not find effort column")

        if 'dist_to_shore_km' not in df.columns:
            df = self.compute_distances(df)

        df['is_near_shore'] = (df['dist_to_shore_km'] < self.min_distance_km).astype(int)
        df[f'{cols["effort"]}_filtered'] = df[cols['effort']].copy()
        df.loc[df['is_near_shore'] == 1, f'{cols["effort"]}_filtered'] = 0

        n_filtered = ((df[cols['effort']] == 1) & (df[f'{cols["effort"]}_filtered'] == 0)).sum()
        if n_filtered > 0:
            logger.info(f"Filtered {n_filtered} near-shore points ({100 * n_filtered / len(df):.2f}%)")

        return df


# ===============================================================
# Coastline Distance Filter (with Regional Cropping + Land Filtering)
# ===============================================================

class CoastlineDistanceFilter:
    """
    Enhanced coastline filter with:
    1. Regional cropping (10-100x faster!) ⚡
    2. Land polygon filtering (removes points ON land) 🏝️
    3. Smart filtering (only process fishing points) 🎯

    Key feature: filter_only_fishing=True
    - Only computes distances for fishing points
    - Skips all non-fishing points (sailing, starting, ending)
    - Saves 50-75% computation time!

    Example:
        2.4M total points, but only 573k fishing points
        Without filter_only_fishing: Process 2.4M points ❌
        With filter_only_fishing: Process 573k points ✅
        Savings: 76% less work!
    """

    def __init__(
            self,
            coastline_shapefile: Optional[str] = None,
            land_shapefile: Optional[str] = None,
            min_distance_km: float = 1.0,
            buffer_distance_km: float = 0.5,
            region: Optional[str] = None,
            custom_bbox: Optional[Tuple[float, float, float, float]] = None,
            filter_land_points: bool = True,
            filter_only_fishing: bool = True,
    ):
        self.coastline_shapefile = coastline_shapefile
        self.land_shapefile = land_shapefile or coastline_shapefile
        self.min_distance_km = min_distance_km
        self.buffer_distance_km = buffer_distance_km
        self.region = region
        self.custom_bbox = custom_bbox
        self.filter_land_points = filter_land_points
        self.filter_only_fishing = filter_only_fishing

        self.coastline_geom = None
        self.land_geom = None
        self.bbox = self._get_bbox()

        if coastline_shapefile:
            self._load_coastline_data()

    def _get_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        if self.custom_bbox:
            logger.info(f"Custom bbox: {self.custom_bbox}")
            return self.custom_bbox
        elif self.region and self.region.lower() in REGIONAL_BBOXES:
            bbox = REGIONAL_BBOXES[self.region.lower()]
            logger.info(f"Region '{self.region}' bbox: {bbox}")
            return bbox
        else:
            logger.warning("No region - using full coastline (slower!)")
            return None

    def _load_coastline_data(self):
        try:
            import geopandas as gpd
            from shapely.geometry import MultiLineString, LineString, MultiPolygon, box

            logger.info(f"Loading: {self.coastline_shapefile}")
            gdf_coast = gpd.read_file(self.coastline_shapefile)

            # FEATURE 1: Regional cropping
            if self.bbox:
                min_lat, max_lat, min_lon, max_lon = self.bbox
                bbox_geom = box(min_lon, min_lat, max_lon, max_lat)

                before = len(gdf_coast)
                gdf_coast = gdf_coast[gdf_coast.intersects(bbox_geom)]
                after = len(gdf_coast)

                logger.info(f"⚡ Cropped: {before} → {after} features ({100 * (1 - after / before):.1f}% reduction)")

            # Extract coastlines
            lines = []
            for geom in gdf_coast.geometry:
                if geom.geom_type == 'LineString':
                    lines.append(geom)
                elif geom.geom_type == 'MultiLineString':
                    lines.extend(geom.geoms)
                elif geom.geom_type == 'Polygon':
                    lines.append(geom.exterior)
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        lines.append(poly.exterior)

            self.coastline_geom = MultiLineString(lines)
            logger.info(f"✓ Coastline: {len(lines)} segments")

            # FEATURE 2: Land polygons
            if self.filter_land_points:
                logger.info(f"Loading land: {self.land_shapefile}")
                gdf_land = gpd.read_file(self.land_shapefile)

                if self.bbox:
                    gdf_land = gdf_land[gdf_land.intersects(bbox_geom)]

                polygons = []
                for geom in gdf_land.geometry:
                    if geom.geom_type == 'Polygon':
                        polygons.append(geom)
                    elif geom.geom_type == 'MultiPolygon':
                        polygons.extend(geom.geoms)

                if polygons:
                    self.land_geom = MultiPolygon(polygons)
                    logger.info(f"✓ Land: {len(polygons)} polygons")
                else:
                    logger.warning("No land polygons - land filtering disabled")
                    self.filter_land_points = False

        except ImportError:
            raise ImportError("geopandas required. Install: pip install geopandas")
        except Exception as e:
            raise RuntimeError(f"Failed to load coastline: {e}")

    def compute_distances(
            self,
            df: pd.DataFrame,
            lat_col: Optional[str] = None,
            lon_col: Optional[str] = None,
            effort_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate distance to coastline + check if on land.

        Args:
            df: DataFrame with GPS points
            lat_col: Latitude column (auto-detected if None)
            lon_col: Longitude column (auto-detected if None)
            effort_col: Effort column (auto-detected if None, needed if filter_only_fishing=True)

        Returns:
            DataFrame with dist_to_shore_km and is_on_land columns
        """
        if not self.coastline_geom:
            raise ValueError("No coastline loaded")

        from shapely.geometry import Point

        df = df.copy()
        cols = get_column_names(df, lat_col, lon_col, effort_col)

        # Determine which points to process
        if self.filter_only_fishing:
            if not cols['effort']:
                logger.warning("filter_only_fishing=True but no effort column found. Processing all points.")
                process_mask = pd.Series([True] * len(df), index=df.index)
            else:
                # Only process fishing points
                process_mask = df[cols['effort']] == 1
                n_fishing = process_mask.sum()
                logger.info(
                    f"Filter mode: Only processing {n_fishing:,} fishing points (skipping {len(df) - n_fishing:,} non-fishing)")
        else:
            # Process all points
            process_mask = pd.Series([True] * len(df), index=df.index)
            logger.info(f"Computing distances for {len(df):,} points...")

        # Initialize with default values
        df['dist_to_shore_km'] = 999.0  # Large value for non-fishing points
        df['is_on_land'] = False

        # Process only selected points
        points_to_process = df[process_mask]
        n_to_process = len(points_to_process)

        if n_to_process == 0:
            logger.warning("No points to process!")
            return df

        logger.info(f"Computing distances for {n_to_process:,} points...")

        distances = []
        on_land = []

        for idx, (_, row) in enumerate(points_to_process.iterrows()):
            if idx % 10000 == 0 and idx > 0:
                logger.info(f"  {idx:,}/{n_to_process:,} ({100 * idx / n_to_process:.1f}%)")

            point = Point(row[cols['lon']], row[cols['lat']])

            # Check land
            is_land = False
            if self.filter_land_points and self.land_geom:
                is_land = self.land_geom.contains(point)
            on_land.append(is_land)

            # Distance to coast
            dist_km = point.distance(self.coastline_geom) * 111.0
            distances.append(dist_km)

        # Update only processed points
        df.loc[process_mask, 'dist_to_shore_km'] = distances
        df.loc[process_mask, 'is_on_land'] = on_land

        if self.filter_land_points:
            n_on_land = sum(on_land)
            logger.info(f"✓ On land: {n_on_land:,} ({100 * n_on_land / n_to_process:.1f}% of processed)")

        if self.filter_only_fishing and cols['effort']:
            logger.info(f"✓ Skipped {len(df) - n_to_process:,} non-fishing points (dist set to 999 km)")

        return df

    def apply_filter(self, df: pd.DataFrame, effort_col: Optional[str] = None, **kwargs) -> pd.DataFrame:
        cols = get_column_names(df, effort_col=effort_col)
        if not cols['effort']:
            raise ValueError("Could not find effort column")

        if 'dist_to_shore_km' not in df.columns:
            df = self.compute_distances(df, effort_col=effort_col)

        df['is_near_shore'] = (df['dist_to_shore_km'] < self.min_distance_km).astype(int)
        df[f'{cols["effort"]}_filtered'] = df[cols['effort']].copy()

        # Filter near-shore
        df.loc[df['is_near_shore'] == 1, f'{cols["effort"]}_filtered'] = 0

        # Filter on-land
        if self.filter_land_points and 'is_on_land' in df.columns:
            land_fishing = ((df[cols['effort']] == 1) & (df['is_on_land'] == True)).sum()
            df.loc[df['is_on_land'] == True, f'{cols["effort"]}_filtered'] = 0
            if land_fishing > 0:
                logger.info(f"🏝️ On-land filtered: {land_fishing} points")

        total_filtered = ((df[cols['effort']] == 1) & (df[f'{cols["effort"]}_filtered'] == 0)).sum()
        logger.info(f"Total filtered: {total_filtered} ({100 * total_filtered / len(df):.2f}%)")

        return df


# ===============================================================
# Convenience Functions
# ===============================================================

def add_shore_filtering(
        df: pd.DataFrame,
        region: str,
        min_distance_km: float = 1.0,
        method: str = 'harbor',
        coastline_shapefile: Optional[str] = None,
        land_shapefile: Optional[str] = None,
        filter_land_points: bool = True,
        filter_only_fishing: bool = True,
        **kwargs
) -> pd.DataFrame:
    """
    Add shore filtering to data.

    Args:
        df: DataFrame with predictions
        region: Region name
        min_distance_km: Minimum distance threshold
        method: 'harbor' or 'coastline'
        coastline_shapefile: Path to coastline shapefile
        land_shapefile: Path to land shapefile
        filter_land_points: Filter points on land
        filter_only_fishing: If True, only compute distances for fishing points (MUCH faster!)
        **kwargs: Additional parameters

    Examples:
        >>> # Harbor (fast)
        >>> df = add_shore_filtering(df, region='zanzibar')
        >>>
        >>> # Coastline + land (accurate) - only process fishing points
        >>> df = add_shore_filtering(
        ...     df,
        ...     region='zanzibar',
        ...     method='coastline',
        ...     coastline_shapefile='GSHHS_f_L1.shp',
        ...     land_shapefile='GSHHS_f_L1.shp',
        ...     filter_only_fishing=True  # Only process fishing points!
        ... )
        >>>
        >>> # Process all points (slower but gets distances for everything)
        >>> df = add_shore_filtering(
        ...     df,
        ...     region='zanzibar',
        ...     method='coastline',
        ...     coastline_shapefile='GSHHS_f_L1.shp',
        ...     filter_only_fishing=False  # Process all points
        ... )
    """
    if method == 'harbor':
        if region.lower() not in REGIONAL_HARBORS:
            raise ValueError(f"Unknown region: {region}. Available: {list(REGIONAL_HARBORS.keys())}")
        filter_obj = HarborDistanceFilter(REGIONAL_HARBORS[region.lower()], min_distance_km)

    elif method == 'coastline':
        if not coastline_shapefile:
            raise ValueError("coastline_shapefile required for coastline method")
        filter_obj = CoastlineDistanceFilter(
            coastline_shapefile, land_shapefile, min_distance_km,
            region=region, filter_land_points=filter_land_points,
            filter_only_fishing=filter_only_fishing
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return filter_obj.apply_filter(df, **kwargs)


def add_custom_region(region_name: str, bbox: Tuple[float, float, float, float],
                      harbors: Optional[List[Tuple[float, float]]] = None):
    """
    Add custom region.

    Example:
        >>> add_custom_region('my_area', (-6, -4, 39, 40), harbors=[(-5, 39.5)])
    """
    REGIONAL_BBOXES[region_name.lower()] = bbox
    if harbors:
        REGIONAL_HARBORS[region_name.lower()] = harbors
    logger.info(f"✓ Added region: {region_name}")


def list_available_regions() -> Dict:
    """List all regions."""
    return {name: {'bbox': bbox, 'harbors': len(REGIONAL_HARBORS.get(name, []))}
            for name, bbox in REGIONAL_BBOXES.items()}
