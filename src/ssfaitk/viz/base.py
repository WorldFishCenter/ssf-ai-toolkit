# src/ssfaitk/viz/base.py
"""Base visualization utilities for SSF AI Toolkit."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def setup_matplotlib_style() -> None:
    """Configure matplotlib with clean, publication-ready defaults."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 8),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
        }
    )


def save_figure(
    fig: plt.Figure,
    output_path: str | Path,
    title: str | None = None,
    close: bool = True,
) -> Path:
    """
    Save matplotlib figure with consistent formatting.
    
    Args:
        fig: matplotlib figure object
        output_path: Where to save the figure
        title: Optional title to set before saving
        close: Whether to close the figure after saving
    
    Returns:
        Path object to saved file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    
    fig.savefig(path, bbox_inches="tight", dpi=300)
    logger.info(f"Saved visualization: {path}")
    
    if close:
        plt.close(fig)
    
    return path


def calculate_grid_bounds(
    lats: np.ndarray | pd.Series,
    lons: np.ndarray | pd.Series,
    padding: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    Calculate bounding box with padding for spatial data.
    
    Args:
        lats: Array of latitudes
        lons: Array of longitudes
        padding: Fraction of range to add as padding (default 5%)
    
    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon)
    """
    lat_range = np.ptp(lats)
    lon_range = np.ptp(lons)
    
    min_lat = np.min(lats) - lat_range * padding
    max_lat = np.max(lats) + lat_range * padding
    min_lon = np.min(lons) - lon_range * padding
    max_lon = np.max(lons) + lon_range * padding
    
    return min_lat, max_lat, min_lon, max_lon


def create_spatial_grid(
    lats: np.ndarray | pd.Series,
    lons: np.ndarray | pd.Series,
    grid_size: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D spatial grid for heatmap generation.
    
    Args:
        lats: Latitude values
        lons: Longitude values
        grid_size: Grid cell size in degrees (default ~1.1 km at equator)
    
    Returns:
        Tuple of (lat_edges, lon_edges, grid_shape)
    """
    min_lat, max_lat, min_lon, max_lon = calculate_grid_bounds(lats, lons)
    
    lat_edges = np.arange(min_lat, max_lat + grid_size, grid_size)
    lon_edges = np.arange(min_lon, max_lon + grid_size, grid_size)
    
    grid_shape = (len(lat_edges) - 1, len(lon_edges) - 1)
    
    return lat_edges, lon_edges, grid_shape


def aggregate_to_grid(
    lats: np.ndarray | pd.Series,
    lons: np.ndarray | pd.Series,
    values: np.ndarray | pd.Series,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    aggregation: str = "sum",
) -> np.ndarray:
    """
    Aggregate point data onto a 2D spatial grid.
    
    Args:
        lats: Latitude values
        lons: Longitude values
        values: Values to aggregate (e.g., effort predictions, counts)
        lat_edges: Latitude bin edges
        lon_edges: Longitude bin edges
        aggregation: Aggregation method ('sum', 'mean', 'count', 'max')
    
    Returns:
        2D array of aggregated values
    """
    # Digitize coordinates
    lat_idx = np.digitize(lats, lat_edges) - 1
    lon_idx = np.digitize(lons, lon_edges) - 1
    
    # Create grid
    grid = np.zeros((len(lat_edges) - 1, len(lon_edges) - 1))
    counts = np.zeros_like(grid)
    
    # Aggregate
    for i, (li, loi, v) in enumerate(zip(lat_idx, lon_idx, values)):
        if 0 <= li < grid.shape[0] and 0 <= loi < grid.shape[1]:
            if aggregation == "sum":
                grid[li, loi] += v
            elif aggregation in ["mean", "count"]:
                grid[li, loi] += v
                counts[li, loi] += 1
            elif aggregation == "max":
                grid[li, loi] = max(grid[li, loi], v)
    
    # Apply mean if requested
    if aggregation == "mean":
        with np.errstate(divide="ignore", invalid="ignore"):
            grid = np.where(counts > 0, grid / counts, 0)
    elif aggregation == "count":
        grid = counts
    
    return grid


def format_coordinate(value: float, coord_type: str = "lat") -> str:
    """
    Format coordinate values with hemisphere indicators.
    
    Args:
        value: Coordinate value in degrees
        coord_type: Either 'lat' or 'lon'
    
    Returns:
        Formatted string (e.g., "12.5°N" or "125.3°E")
    """
    if coord_type == "lat":
        hemisphere = "N" if value >= 0 else "S"
    else:
        hemisphere = "E" if value >= 0 else "W"
    
    return f"{abs(value):.2f}°{hemisphere}"
