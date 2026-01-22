# src/ssfaitk/viz/effort_maps.py
"""Visualization functions for fishing effort analysis and mapping."""
from __future__ import annotations

from ..utils.column_mapper import resolve_column_name
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from ..utils.logging import get_logger
from .base import (
    aggregate_to_grid,
    calculate_grid_bounds,
    create_spatial_grid,
    save_figure,
    setup_matplotlib_style,
)

logger = get_logger(__name__)

# Color schemes
EFFORT_COLORS = {
    "fishing": "#d62728",  # red
    "non_fishing": "#1f77b4",  # blue
    "mixed": "#ff7f0e",  # orange
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "effort", ["#f7fbff", "#6baed6", "#2171b5", "#08519c", "#08306b"]
)


def plot_effort_tracks(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    effort_col: str = "is_fishing",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    trip_col: str = "trip_id",
    figsize: tuple[float, float] = (14, 10),
    alpha: float = 0.6,
    linewidth: float = 1.5,
) -> plt.Figure:
    """
    Plot vessel tracks colored by fishing effort classification.
    
    Args:
        df: DataFrame with track points and effort predictions
        output_path: Where to save the plot (optional)
        effort_col: Column with effort predictions (0=non-fishing, 1=fishing)
        lat_col: Latitude column name
        lon_col: Longitude column name
        trip_col: Trip identifier column
        figsize: Figure size
        alpha: Line transparency
        linewidth: Width of track lines
    
    Returns:
        matplotlib Figure object
    """

    
    setup_matplotlib_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate fishing and non-fishing segments
    fishing_mask = df[effort_col] == 1
    
    # Plot non-fishing segments (blue)
    for trip_id, group in df[~fishing_mask].groupby(trip_col):
        if len(group) > 1:
            ax.plot(
                group[lon_col],
                group[lat_col],
                color=EFFORT_COLORS["non_fishing"],
                alpha=alpha * 0.5,
                linewidth=linewidth * 0.8,
                label="Non-fishing" if trip_id == df[trip_col].iloc[0] else "",
            )
    
    # Plot fishing segments (red) - on top for visibility
    for trip_id, group in df[fishing_mask].groupby(trip_col):
        if len(group) > 1:
            ax.plot(
                group[lon_col],
                group[lat_col],
                color=EFFORT_COLORS["fishing"],
                alpha=alpha,
                linewidth=linewidth,
                label="Fishing" if trip_id == df[trip_col].iloc[0] else "",
            )
    
    # Styling
    ax.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=12, fontweight="bold")
    ax.set_title(
        "Fishing Effort Tracks\n(Red = Fishing Activity, Blue = Transport/Non-fishing)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", framealpha=0.9)
    
    # Statistics annotation
    total_points = len(df)
    fishing_points = fishing_mask.sum()
    fishing_pct = 100 * fishing_points / total_points if total_points > 0 else 0
    
    stats_text = (
        f"Total Points: {total_points:,}\n"
        f"Fishing: {fishing_points:,} ({fishing_pct:.1f}%)\n"
        f"Non-fishing: {total_points - fishing_points:,} ({100-fishing_pct:.1f}%)\n"
        f"Trips: {df[trip_col].nunique()}"
    )
    
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, close=False)
    
    return fig


def plot_effort_heatmap(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    effort_col: str = "is_fishing",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    grid_size: float = 0.01,
    aggregation: Literal["sum", "mean", "count"] = "sum",
    figsize: tuple[float, float] = (14, 10),
    cmap: str | LinearSegmentedColormap = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """
    Generate a heatmap showing spatial distribution of fishing effort.
    
    Args:
        df: DataFrame with track points and effort predictions
        output_path: Where to save the plot (optional)
        effort_col: Column with effort predictions (0/1)
        lat_col: Latitude column name
        lon_col: Longitude column name
        grid_size: Size of grid cells in degrees (~1.1 km at equator for 0.01)
        aggregation: How to aggregate ('sum'=total effort, 'mean'=intensity, 'count'=points)
        figsize: Figure size
        cmap: Colormap (default: blue gradient)
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
    
    Returns:
        matplotlib Figure object
    """
    setup_matplotlib_style()
    
    if cmap is None:
        cmap = HEATMAP_CMAP
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter to fishing points only
    fishing_df = df[df[effort_col] == 1].copy()
    
    if len(fishing_df) == 0:
        logger.warning("No fishing effort detected in data")
        ax.text(
            0.5,
            0.5,
            "No Fishing Effort Detected",
            ha="center",
            va="center",
            fontsize=16,
            transform=ax.transAxes,
        )
        return fig
    
    # Create grid
    lat_edges, lon_edges, grid_shape = create_spatial_grid(
        fishing_df[lat_col], fishing_df[lon_col], grid_size=grid_size
    )
    
    # Aggregate effort to grid
    effort_grid = aggregate_to_grid(
        fishing_df[lat_col],
        fishing_df[lon_col],
        fishing_df[effort_col],
        lat_edges,
        lon_edges,
        aggregation=aggregation,
    )
    
    # Mask zero values for cleaner visualization
    effort_grid_masked = np.ma.masked_where(effort_grid == 0, effort_grid)
    
    # Plot heatmap
    im = ax.imshow(
        effort_grid_masked,
        extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
    )
    
    # Overlay actual fishing points
    ax.scatter(
        fishing_df[lon_col],
        fishing_df[lat_col],
        c="red",
        s=1,
        alpha=0.3,
        label="Fishing points",
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if aggregation == "sum":
        cbar.set_label("Total Fishing Effort (points)", fontsize=11, fontweight="bold")
    elif aggregation == "mean":
        cbar.set_label("Fishing Intensity (mean)", fontsize=11, fontweight="bold")
    else:
        cbar.set_label("Fishing Point Count", fontsize=11, fontweight="bold")
    
    # Styling
    ax.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Fishing Effort Heatmap\n(Grid size: {grid_size}° ≈ {grid_size*111:.1f} km)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    
    # Statistics
    total_fishing = fishing_df[effort_col].sum()
    max_cell = effort_grid.max()
    hotspot_threshold = np.percentile(effort_grid[effort_grid > 0], 90)
    n_hotspots = (effort_grid >= hotspot_threshold).sum()
    
    stats_text = (
        f"Total Fishing Points: {int(total_fishing):,}\n"
        f"Grid Cells: {grid_shape[0]} × {grid_shape[1]}\n"
        f"Max Cell Value: {max_cell:.1f}\n"
        f"Hotspot Cells (>90th %ile): {n_hotspots}"
    )
    
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, close=False)
    
    return fig


def plot_effort_hotspots(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    effort_col: str = "is_fishing",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    grid_size: float = 0.01,
    percentile_threshold: float = 90.0,
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Identify and visualize fishing effort hotspots.
    
    Args:
        df: DataFrame with track points and effort predictions
        output_path: Where to save the plot (optional)
        effort_col: Column with effort predictions
        lat_col: Latitude column name
        lon_col: Longitude column name
        grid_size: Grid cell size in degrees
        percentile_threshold: Percentile to define hotspots (e.g., 90 = top 10%)
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    setup_matplotlib_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter to fishing points
    fishing_df = df[df[effort_col] == 1].copy()
    
    if len(fishing_df) == 0:
        logger.warning("No fishing effort detected")
        return fig
    
    # Create grid and aggregate
    lat_edges, lon_edges, _ = create_spatial_grid(
        fishing_df[lat_col], fishing_df[lon_col], grid_size=grid_size
    )
    
    effort_grid = aggregate_to_grid(
        fishing_df[lat_col],
        fishing_df[lon_col],
        fishing_df[effort_col],
        lat_edges,
        lon_edges,
        aggregation="sum",
    )
    
    # Define hotspots
    threshold = np.percentile(effort_grid[effort_grid > 0], percentile_threshold)
    hotspot_mask = effort_grid >= threshold
    
    # Create hotspot overlay
    hotspot_grid = np.ma.masked_where(~hotspot_mask, effort_grid)
    
    # Base heatmap (all effort)
    effort_masked = np.ma.masked_where(effort_grid == 0, effort_grid)
    ax.imshow(
        effort_masked,
        extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]],
        origin="lower",
        aspect="auto",
        cmap="Blues",
        alpha=0.4,
        interpolation="bilinear",
    )
    
    # Hotspot overlay (red)
    im = ax.imshow(
        hotspot_grid,
        extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]],
        origin="lower",
        aspect="auto",
        cmap="Reds",
        alpha=0.7,
        interpolation="bilinear",
    )
    
    # Hotspot centers
    hotspot_coords = np.argwhere(hotspot_mask)
    hotspot_lats = lat_edges[hotspot_coords[:, 0]]
    hotspot_lons = lon_edges[hotspot_coords[:, 1]]
    
    ax.scatter(
        hotspot_lons,
        hotspot_lats,
        c="darkred",
        s=30,
        marker="x",
        label=f"Hotspot Centers (n={len(hotspot_lats)})",
        alpha=0.8,
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fishing Effort in Hotspots", fontsize=11, fontweight="bold")
    
    # Styling
    ax.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Fishing Effort Hotspots (>{percentile_threshold:.0f}th Percentile)\n"
        f"Grid: {grid_size}°, Threshold: {threshold:.1f} points/cell",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    
    # Statistics
    total_cells = effort_grid.size
    active_cells = (effort_grid > 0).sum()
    hotspot_cells = hotspot_mask.sum()
    hotspot_pct = 100 * hotspot_cells / active_cells if active_cells > 0 else 0
    
    stats_text = (
        f"Total Grid Cells: {total_cells:,}\n"
        f"Active Cells: {active_cells:,}\n"
        f"Hotspot Cells: {hotspot_cells:,} ({hotspot_pct:.1f}%)\n"
        f"Threshold: ≥{threshold:.1f} points"
    )
    
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path, close=False)
    
    return fig


def plot_effort_comparison(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    effort_col: str = "is_fishing",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    trip_col: str = "trip_id",
    grid_size: float = 0.01,
    figsize: tuple[float, float] = (18, 10),
) -> plt.Figure:
    """
    Create a multi-panel comparison of tracks and heatmap.
    
    Args:
        df: DataFrame with track points and effort predictions
        output_path: Where to save the plot
        effort_col: Column with effort predictions
        lat_col: Latitude column
        lon_col: Longitude column
        trip_col: Trip identifier column
        grid_size: Grid cell size for heatmap
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    setup_matplotlib_style()
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    # Left: Tracks
    ax1 = fig.add_subplot(gs[0, 0])
    fishing_mask = df[effort_col] == 1
    
    for trip_id, group in df[~fishing_mask].groupby(trip_col):
        if len(group) > 1:
            ax1.plot(
                group[lon_col],
                group[lat_col],
                color=EFFORT_COLORS["non_fishing"],
                alpha=0.4,
                linewidth=1.2,
            )
    
    for trip_id, group in df[fishing_mask].groupby(trip_col):
        if len(group) > 1:
            ax1.plot(
                group[lon_col],
                group[lat_col],
                color=EFFORT_COLORS["fishing"],
                alpha=0.6,
                linewidth=1.5,
            )
    
    ax1.set_xlabel("Longitude", fontweight="bold")
    ax1.set_ylabel("Latitude", fontweight="bold")
    ax1.set_title("Vessel Tracks by Activity Type", fontweight="bold", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Right: Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    fishing_df = df[df[effort_col] == 1]
    
    if len(fishing_df) > 0:
        lat_edges, lon_edges, _ = create_spatial_grid(
            fishing_df[lat_col], fishing_df[lon_col], grid_size=grid_size
        )
        
        effort_grid = aggregate_to_grid(
            fishing_df[lat_col],
            fishing_df[lon_col],
            fishing_df[effort_col],
            lat_edges,
            lon_edges,
            aggregation="sum",
        )
        
        effort_masked = np.ma.masked_where(effort_grid == 0, effort_grid)
        
        im = ax2.imshow(
            effort_masked,
            extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]],
            origin="lower",
            aspect="auto",
            cmap=HEATMAP_CMAP,
            interpolation="bilinear",
            alpha=0.8,
        )
        
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Fishing Effort")
    
    ax2.set_xlabel("Longitude", fontweight="bold")
    ax2.set_ylabel("Latitude", fontweight="bold")
    ax2.set_title("Fishing Effort Density", fontweight="bold", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(
        "Fishing Effort Analysis: Tracks vs. Heatmap",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    
    if output_path:
        save_figure(fig, output_path, close=False)
    
    return fig


def generate_effort_report(
    df: pd.DataFrame,
    output_dir: str | Path,
    effort_col: str = "is_fishing",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    trip_col: str = "trip_id",
    grid_size: float = 0.01,
    prefix: str = "effort_analysis",
) -> dict[str, Path]:
    """
    Generate a complete set of effort visualizations.
    
    Args:
        df: DataFrame with predictions
        output_dir: Directory to save plots
        effort_col: Effort prediction column
        lat_col: Latitude column
        lon_col: Longitude column
        trip_col: Trip ID column
        grid_size: Grid size for spatial aggregation
        prefix: Filename prefix for outputs
    
    Returns:
        Dictionary mapping plot type to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating effort analysis report...")
    
    outputs = {}
    
    # 1. Track plot
    logger.info("Creating track plot...")
    fig1 = plot_effort_tracks(df, None, effort_col, lat_col, lon_col, trip_col)
    path1 = save_figure(fig1, output_dir / f"{prefix}_tracks.png")
    outputs["tracks"] = path1
    
    # 2. Heatmap
    logger.info("Creating heatmap...")
    fig2 = plot_effort_heatmap(df, None, effort_col, lat_col, lon_col, grid_size)
    path2 = save_figure(fig2, output_dir / f"{prefix}_heatmap.png")
    outputs["heatmap"] = path2
    
    # 3. Hotspots
    logger.info("Creating hotspot analysis...")
    fig3 = plot_effort_hotspots(df, None, effort_col, lat_col, lon_col, grid_size)
    path3 = save_figure(fig3, output_dir / f"{prefix}_hotspots.png")
    outputs["hotspots"] = path3
    
    # 4. Comparison
    logger.info("Creating comparison plot...")
    fig4 = plot_effort_comparison(df, None, effort_col, lat_col, lon_col, trip_col, grid_size)
    path4 = save_figure(fig4, output_dir / f"{prefix}_comparison.png")
    outputs["comparison"] = path4
    
    logger.info(f"Report complete! Files saved to: {output_dir}")
    
    return outputs
