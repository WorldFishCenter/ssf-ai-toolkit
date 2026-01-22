"""
PDS Trip File Loader

Utility for loading trip files from local PDS data directory.

Supports:
1. Load single trip by ID
2. Load specific set of trip IDs
3. Load random N trips
4. Load all trips
5. Filter by criteria (date, fishing percentage, etc.)
"""

from pathlib import Path
from typing import Union, List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ================================================================
# Configuration
# ================================================================

# Default data directory (adjust for your system)
DEFAULT_DATA_DIR = Path("~/Documents/GitHub/ssf-ai-toolkit/examples/data/pds-trips/zanzibar").expanduser()

# File naming pattern
FILE_PATTERN = "pds-tracks_{trip_id}.parquet"


# ================================================================
# Core Loading Functions
# ================================================================

def load_trip(
    trip_id: Union[str, int, float],
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load a single trip by ID.
    
    Args:
        trip_id: Trip identifier (can be string, int, or float)
        data_dir: Directory containing trip files (default: DEFAULT_DATA_DIR)
    
    Returns:
        DataFrame with trip data
    
    Examples:
        >>> df = load_trip(12067898)
        >>> df = load_trip('14119254')
        >>> df = load_trip(14119254.0)
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    
    # Handle different trip_id types
    if isinstance(trip_id, float):
        trip_id = int(trip_id)
    trip_id = str(trip_id)
    
    # Construct file path
    filename = f"pds-tracks_{trip_id}.parquet"
    filepath = data_dir / filename
    
    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(
            f"Trip file not found: {filepath}\n"
            f"Available trips: {list_available_trips(data_dir)[:10]}..."
        )
    
    # Load data
    logger.info(f"Loading trip {trip_id} from {filepath}")
    df = pd.read_parquet(filepath)
    
    logger.info(f"Loaded {len(df):,} points for trip {trip_id}")
    
    return df


def load_trips(
    trip_ids: List[Union[str, int, float]],
    data_dir: Optional[Path] = None,
    combine: bool = True
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Load multiple specific trips by ID.
    
    Args:
        trip_ids: List of trip identifiers
        data_dir: Directory containing trip files
        combine: If True, combine all trips into single DataFrame
    
    Returns:
        Single DataFrame (if combine=True) or list of DataFrames (if combine=False)
    
    Examples:
        >>> # Load multiple trips combined
        >>> df = load_trips([12067898, 14119254, 14232947])
        >>> print(f"Total points: {len(df):,}")
        
        >>> # Load separately
        >>> dfs = load_trips([12067898, 14119254], combine=False)
        >>> print(f"Trip 1: {len(dfs[0]):,} points")
        >>> print(f"Trip 2: {len(dfs[1]):,} points")
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    
    logger.info(f"Loading {len(trip_ids)} trips...")
    
    dataframes = []
    failed_trips = []
    
    for i, trip_id in enumerate(trip_ids, 1):
        try:
            df = load_trip(trip_id, data_dir)
            dataframes.append(df)
            logger.info(f"  [{i}/{len(trip_ids)}] Trip {trip_id}: {len(df):,} points ✓")
        except FileNotFoundError as e:
            logger.warning(f"  [{i}/{len(trip_ids)}] Trip {trip_id}: Not found ✗")
            failed_trips.append(trip_id)
    
    if failed_trips:
        logger.warning(f"Failed to load {len(failed_trips)} trips: {failed_trips}")
    
    if not dataframes:
        raise ValueError("No trips were successfully loaded")
    
    if combine:
        logger.info(f"Combining {len(dataframes)} trips into single DataFrame...")
        combined = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined: {len(combined):,} total points from {len(dataframes)} trips")
        return combined
    else:
        return dataframes


def load_random_trips(
    n: int,
    data_dir: Optional[Path] = None,
    combine: bool = True,
    seed: Optional[int] = None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Load N random trips from available files.
    
    Args:
        n: Number of random trips to load
        data_dir: Directory containing trip files
        combine: If True, combine into single DataFrame
        seed: Random seed for reproducibility
    
    Returns:
        Single DataFrame (if combine=True) or list of DataFrames
    
    Examples:
        >>> # Load 10 random trips
        >>> df = load_random_trips(10)
        >>> print(f"Loaded {df['trip_id'].nunique()} trips")
        
        >>> # Load 5 random trips separately
        >>> dfs = load_random_trips(5, combine=False)
        >>> for i, df in enumerate(dfs, 1):
        ...     print(f"Trip {i}: {len(df):,} points")
        
        >>> # Reproducible random selection
        >>> df = load_random_trips(10, seed=42)
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    
    # Get all available trips
    available_trips = list_available_trips(data_dir)
    
    if n > len(available_trips):
        logger.warning(
            f"Requested {n} trips but only {len(available_trips)} available. "
            f"Loading all {len(available_trips)} trips."
        )
        n = len(available_trips)
    
    # Random selection
    if seed is not None:
        np.random.seed(seed)
    
    selected_trips = np.random.choice(available_trips, size=n, replace=False)
    
    logger.info(f"Randomly selected {n} trips (seed={seed})")
    logger.info(f"Selected trip IDs: {selected_trips[:10]}{'...' if len(selected_trips) > 10 else ''}")
    
    # Load selected trips
    return load_trips(selected_trips, data_dir, combine)


def load_all_trips(
    data_dir: Optional[Path] = None,
    combine: bool = True,
    max_trips: Optional[int] = None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Load all available trips from directory.
    
    Args:
        data_dir: Directory containing trip files
        combine: If True, combine into single DataFrame
        max_trips: Maximum number of trips to load (optional)
    
    Returns:
        Single DataFrame (if combine=True) or list of DataFrames
    
    Warning:
        Loading all trips can use significant memory!
        Consider using max_trips to limit.
    
    Examples:
        >>> # Load all trips (use with caution!)
        >>> df = load_all_trips()
        
        >>> # Load first 50 trips
        >>> df = load_all_trips(max_trips=50)
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    
    available_trips = list_available_trips(data_dir)
    
    if max_trips and max_trips < len(available_trips):
        logger.warning(f"Loading first {max_trips} of {len(available_trips)} available trips")
        available_trips = available_trips[:max_trips]
    else:
        logger.warning(f"Loading ALL {len(available_trips)} trips - this may take a while!")
    
    return load_trips(available_trips, data_dir, combine)


# ================================================================
# Helper Functions
# ================================================================

def list_available_trips(data_dir: Optional[Path] = None) -> List[str]:
    """
    List all available trip IDs in the data directory.
    
    Args:
        data_dir: Directory containing trip files
    
    Returns:
        List of trip IDs (as strings)
    
    Examples:
        >>> trips = list_available_trips()
        >>> print(f"Found {len(trips)} trips")
        >>> print(f"First 10: {trips[:10]}")
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please check the path or download data first."
        )
    
    # Find all parquet files matching pattern
    files = list(data_dir.glob("pds-tracks_*.parquet"))
    
    # Extract trip IDs from filenames
    trip_ids = []
    for file in files:
        # Extract ID from "pds-tracks_{trip_id}.parquet"
        filename = file.stem  # Remove .parquet
        trip_id = filename.replace("pds-tracks_", "")
        trip_ids.append(trip_id)
    
    # Sort numerically if possible
    try:
        trip_ids.sort(key=lambda x: float(x))
    except ValueError:
        trip_ids.sort()  # Alphabetical if not numeric
    
    logger.info(f"Found {len(trip_ids)} trips in {data_dir}")
    
    return trip_ids


def get_trip_info(
    trip_id: Union[str, int, float],
    data_dir: Optional[Path] = None
) -> dict:
    """
    Get information about a trip without loading full data.
    
    Args:
        trip_id: Trip identifier
        data_dir: Directory containing trip files
    
    Returns:
        Dictionary with trip metadata
    
    Examples:
        >>> info = get_trip_info(12067898)
        >>> print(f"Points: {info['n_points']}")
        >>> print(f"File size: {info['file_size_mb']:.1f} MB")
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    
    # Handle trip_id types
    if isinstance(trip_id, float):
        trip_id = int(trip_id)
    trip_id = str(trip_id)
    
    filepath = data_dir / f"pds-tracks_{trip_id}.parquet"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Trip file not found: {filepath}")
    
    # Get file info
    file_size = filepath.stat().st_size
    
    # Read just metadata (fast)
    df = pd.read_parquet(filepath)
    
    info = {
        'trip_id': trip_id,
        'filepath': str(filepath),
        'file_size_bytes': file_size,
        'file_size_mb': file_size / (1024 * 1024),
        'n_points': len(df),
        'columns': df.columns.tolist(),
    }
    
    # Add optional info if columns exist
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        info['start_time'] = df['timestamp'].min()
        info['end_time'] = df['timestamp'].max()
        info['duration_hours'] = (info['end_time'] - info['start_time']).total_seconds() / 3600
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        info['lat_range'] = (df['latitude'].min(), df['latitude'].max())
        info['lon_range'] = (df['longitude'].min(), df['longitude'].max())
    
    return info


def summarize_trips(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Create summary table of all available trips.
    
    Args:
        data_dir: Directory containing trip files
    
    Returns:
        DataFrame with trip summaries
    
    Examples:
        >>> summary = summarize_trips()
        >>> print(summary.head())
        >>> 
        >>> # Find trips with most points
        >>> top_trips = summary.nlargest(10, 'n_points')
        >>> print(top_trips)
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    
    trip_ids = list_available_trips(data_dir)
    
    logger.info(f"Summarizing {len(trip_ids)} trips...")
    
    summaries = []
    for i, trip_id in enumerate(trip_ids, 1):
        try:
            info = get_trip_info(trip_id, data_dir)
            summaries.append(info)
            if i % 10 == 0:
                logger.info(f"  Processed {i}/{len(trip_ids)} trips...")
        except Exception as e:
            logger.warning(f"  Trip {trip_id}: Error - {e}")
    
    summary_df = pd.DataFrame(summaries)
    
    logger.info(f"✓ Summary complete: {len(summary_df)} trips")
    
    return summary_df


# ================================================================
# Advanced Filtering
# ================================================================

def find_trips_by_criteria(
    min_points: Optional[int] = None,
    max_points: Optional[int] = None,
    min_duration_hours: Optional[float] = None,
    max_duration_hours: Optional[float] = None,
    data_dir: Optional[Path] = None
) -> List[str]:
    """
    Find trips matching specific criteria.
    
    Args:
        min_points: Minimum number of GPS points
        max_points: Maximum number of GPS points
        min_duration_hours: Minimum trip duration in hours
        max_duration_hours: Maximum trip duration in hours
        data_dir: Directory containing trip files
    
    Returns:
        List of trip IDs matching criteria
    
    Examples:
        >>> # Find trips with 1000-5000 points
        >>> trips = find_trips_by_criteria(min_points=1000, max_points=5000)
        >>> 
        >>> # Find long trips (>12 hours)
        >>> long_trips = find_trips_by_criteria(min_duration_hours=12)
        >>> 
        >>> # Find short, dense trips
        >>> trips = find_trips_by_criteria(
        ...     min_points=2000,
        ...     max_duration_hours=6
        ... )
    """
    # Get summary of all trips
    summary = summarize_trips(data_dir)
    
    # Apply filters
    mask = pd.Series([True] * len(summary))
    
    if min_points is not None:
        mask &= summary['n_points'] >= min_points
    
    if max_points is not None:
        mask &= summary['n_points'] <= max_points
    
    if min_duration_hours is not None and 'duration_hours' in summary.columns:
        mask &= summary['duration_hours'] >= min_duration_hours
    
    if max_duration_hours is not None and 'duration_hours' in summary.columns:
        mask &= summary['duration_hours'] <= max_duration_hours
    
    filtered = summary[mask]
    
    logger.info(f"Found {len(filtered)} trips matching criteria")
    
    return filtered['trip_id'].tolist()


# ================================================================
# Convenience Class
# ================================================================

class TripLoader:
    """
    Convenient class-based interface for loading trips.
    
    Examples:
        >>> loader = TripLoader()
        >>> 
        >>> # Load single trip
        >>> df = loader.load_one(12067898)
        >>> 
        >>> # Load multiple trips
        >>> df = loader.load_many([12067898, 14119254])
        >>> 
        >>> # Load random trips
        >>> df = loader.load_random(10)
        >>> 
        >>> # List available trips
        >>> trips = loader.list_trips()
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize trip loader.
        
        Args:
            data_dir: Directory containing trip files (default: DEFAULT_DATA_DIR)
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Please check the path or download data first."
            )
        
        logger.info(f"TripLoader initialized with data_dir: {self.data_dir}")
    
    def load_one(self, trip_id: Union[str, int, float]) -> pd.DataFrame:
        """Load single trip."""
        return load_trip(trip_id, self.data_dir)
    
    def load_many(
        self,
        trip_ids: List[Union[str, int, float]],
        combine: bool = True
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Load multiple specific trips."""
        return load_trips(trip_ids, self.data_dir, combine)
    
    def load_random(
        self,
        n: int,
        combine: bool = True,
        seed: Optional[int] = None
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Load N random trips."""
        return load_random_trips(n, self.data_dir, combine, seed)
    
    def load_all(
        self,
        combine: bool = True,
        max_trips: Optional[int] = None
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Load all available trips."""
        return load_all_trips(self.data_dir, combine, max_trips)
    
    def list_trips(self) -> List[str]:
        """List all available trip IDs."""
        return list_available_trips(self.data_dir)
    
    def get_info(self, trip_id: Union[str, int, float]) -> dict:
        """Get trip information."""
        return get_trip_info(trip_id, self.data_dir)
    
    def summarize(self) -> pd.DataFrame:
        """Get summary of all trips."""
        return summarize_trips(self.data_dir)
    
    def find_by_criteria(self, **kwargs) -> List[str]:
        """Find trips by criteria."""
        return find_trips_by_criteria(data_dir=self.data_dir, **kwargs)


# ================================================================
# Main Examples
# ================================================================

if __name__ == '__main__':
    """Usage examples."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("PDS TRIP FILE LOADER - EXAMPLES")
    print("="*60)
    
    # Example 1: Load single trip
    print("\n1. Load single trip:")
    df = load_trip(12067898)
    print(f"   Trip 12067898: {len(df):,} points")
    
    # Example 2: Load specific trips
    print("\n2. Load specific trips:")
    df = load_trips([12067898, 14119254, 14232947])
    print(f"   Combined: {len(df):,} points, {df['trip_id'].nunique()} trips")
    
    # Example 3: Load random trips
    print("\n3. Load 5 random trips:")
    df = load_random_trips(5, seed=42)
    print(f"   Random selection: {len(df):,} points, {df['trip_id'].nunique()} trips")
    
    # Example 4: List available trips
    print("\n4. List available trips:")
    trips = list_available_trips()
    print(f"   Found {len(trips)} trips")
    print(f"   First 10: {trips[:10]}")
    
    # Example 5: Get trip info
    print("\n5. Get trip info:")
    info = get_trip_info(12067898)
    print(f"   Trip {info['trip_id']}:")
    print(f"     Points: {info['n_points']:,}")
    print(f"     File size: {info['file_size_mb']:.1f} MB")
    if 'duration_hours' in info:
        print(f"     Duration: {info['duration_hours']:.1f} hours")
    
    # Example 6: Using TripLoader class
    print("\n6. Using TripLoader class:")
    loader = TripLoader()
    df = loader.load_random(3)
    print(f"   Loaded {df['trip_id'].nunique()} random trips")
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60)
