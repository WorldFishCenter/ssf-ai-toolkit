#!/usr/bin/env python3
"""
Coastline Data Downloader

Downloads and prepares coastline shapefiles for shore distance filtering.
Supports multiple data sources with different detail levels.
"""

import urllib.request
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ================================================================
# Data Sources
# ================================================================

SOURCES = {
    'natural_earth_10m': {
        'name': 'Natural Earth 10m (High Detail)',
        'url': 'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip',
        'shapefile': 'ne_10m_coastline.shp',
        'size': '~4 MB',
        'detail': 'High',
        'recommended': True
    },
    'natural_earth_50m': {
        'name': 'Natural Earth 50m (Medium Detail)',
        'url': 'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/physical/ne_50m_coastline.zip',
        'shapefile': 'ne_50m_coastline.shp',
        'size': '~1 MB',
        'detail': 'Medium',
        'recommended': False
    },
    'natural_earth_110m': {
        'name': 'Natural Earth 110m (Low Detail)',
        'url': 'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/physical/ne_110m_coastline.zip',
        'shapefile': 'ne_110m_coastline.shp',
        'size': '~200 KB',
        'detail': 'Low',
        'recommended': False
    }
}


# ================================================================
# Download Functions
# ================================================================

def download_coastline(
    source: str = 'natural_earth_10m',
    output_dir: str = 'coastline_data',
    force: bool = False
) -> Path:
    """
    Download coastline shapefile.
    
    Args:
        source: Data source key (see SOURCES dict)
        output_dir: Directory to save data
        force: If True, re-download even if exists
    
    Returns:
        Path to downloaded shapefile
    
    Examples:
        >>> shapefile_path = download_coastline()
        >>> print(f"Downloaded: {shapefile_path}")
    """
    if source not in SOURCES:
        raise ValueError(
            f"Unknown source: {source}\n"
            f"Available: {list(SOURCES.keys())}"
        )
    
    source_info = SOURCES[source]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Expected shapefile path
    shapefile_name = source_info['shapefile']
    shapefile_path = output_dir / shapefile_name
    
    # Check if already exists
    if shapefile_path.exists() and not force:
        logger.info(f"✓ Coastline already exists: {shapefile_path}")
        return shapefile_path
    
    # Download
    logger.info(f"Downloading {source_info['name']}...")
    logger.info(f"  URL: {source_info['url']}")
    logger.info(f"  Size: {source_info['size']}")
    logger.info(f"  Detail: {source_info['detail']}")
    
    zip_path = output_dir / f"{source}.zip"
    
    try:
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(
            source_info['url'],
            zip_path,
            reporthook=show_progress
        )
        print()  # New line after progress
        
        logger.info(f"✓ Downloaded to: {zip_path}")
        
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        raise
    
    # Extract
    logger.info(f"Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        logger.info(f"✓ Extracted to: {output_dir}")
        
        # Clean up zip file
        zip_path.unlink()
        logger.info(f"✓ Cleaned up zip file")
        
    except Exception as e:
        logger.error(f"✗ Extraction failed: {e}")
        raise
    
    # Verify shapefile exists
    if not shapefile_path.exists():
        raise FileNotFoundError(
            f"Shapefile not found after extraction: {shapefile_path}\n"
            f"Files in {output_dir}: {list(output_dir.glob('*'))}"
        )
    
    logger.info(f"\n✓ SUCCESS! Coastline shapefile ready:")
    logger.info(f"  {shapefile_path}")
    
    return shapefile_path


def list_sources():
    """Print available data sources."""
    print("\n" + "="*60)
    print("AVAILABLE COASTLINE DATA SOURCES")
    print("="*60)
    
    for key, info in SOURCES.items():
        rec = " ⭐ RECOMMENDED" if info['recommended'] else ""
        print(f"\n{key}{rec}")
        print(f"  Name: {info['name']}")
        print(f"  Size: {info['size']}")
        print(f"  Detail: {info['detail']}")


def setup_coastline_for_region(
    region: str,
    output_dir: str = 'coastline_data'
) -> Path:
    """
    Download and prepare coastline data for a specific region.
    
    Args:
        region: Region name ('global', 'africa', 'zanzibar', etc.)
        output_dir: Output directory
    
    Returns:
        Path to shapefile
    
    Examples:
        >>> # For Zanzibar/Kenya/Tanzania
        >>> shapefile = setup_coastline_for_region('zanzibar')
        >>> 
        >>> # For global coverage
        >>> shapefile = setup_coastline_for_region('global')
    """
    logger.info(f"\nSetting up coastline data for region: {region}")
    
    # Recommend source based on region
    if region.lower() in ['global', 'world']:
        source = 'natural_earth_10m'
    elif region.lower() in ['africa', 'zanzibar', 'kenya', 'tanzania', 'timor']:
        source = 'natural_earth_10m'  # High detail for smaller regions
    else:
        source = 'natural_earth_10m'  # Default
    
    logger.info(f"Using data source: {source}")
    
    return download_coastline(source, output_dir)


def test_coastline_file(shapefile_path: str):
    """
    Test if coastline file can be loaded with geopandas.
    
    Args:
        shapefile_path: Path to shapefile
    """
    try:
        import geopandas as gpd
        
        logger.info(f"\nTesting coastline file...")
        logger.info(f"  File: {shapefile_path}")
        
        gdf = gpd.read_file(shapefile_path)
        
        logger.info(f"  ✓ Loaded successfully!")
        logger.info(f"  Features: {len(gdf)}")
        logger.info(f"  CRS: {gdf.crs}")
        logger.info(f"  Bounds: {gdf.total_bounds}")
        
        return True
        
    except ImportError:
        logger.warning("  ⚠ geopandas not installed - cannot test file")
        logger.warning("  Install with: pip install geopandas")
        return False
        
    except Exception as e:
        logger.error(f"  ✗ Error loading file: {e}")
        return False


# ================================================================
# Main CLI
# ================================================================

def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download coastline shapefiles for shore distance filtering'
    )
    parser.add_argument(
        '--source',
        choices=list(SOURCES.keys()),
        default='natural_earth_10m',
        help='Data source (default: natural_earth_10m)'
    )
    parser.add_argument(
        '--output-dir',
        default='coastline_data',
        help='Output directory (default: coastline_data)'
    )
    parser.add_argument(
        '--region',
        help='Region name (e.g., zanzibar, kenya, global)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available sources'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test loading the downloaded file'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if file exists'
    )
    
    args = parser.parse_args()
    
    # List sources
    if args.list:
        list_sources()
        return
    
    # Download
    if args.region:
        shapefile_path = setup_coastline_for_region(args.region, args.output_dir)
    else:
        shapefile_path = download_coastline(args.source, args.output_dir, args.force)
    
    # Test
    if args.test:
        test_coastline_file(shapefile_path)
    
    # Usage instructions
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\nUse this shapefile with CoastlineDistanceFilter:")
    print(f"""
from shore_distance_filter import CoastlineDistanceFilter

filter = CoastlineDistanceFilter(
    coastline_shapefile='{shapefile_path}',
    min_distance_km=1.0
)

df = filter.apply_filter(df)
""")


# ================================================================
# Quick Usage Functions
# ================================================================

def quick_setup(region: str = 'global') -> Path:
    """
    Quick setup for most common use case.
    
    Args:
        region: Region name ('zanzibar', 'kenya', 'global', etc.)
    
    Returns:
        Path to downloaded shapefile
    
    Examples:
        >>> # Quick setup for Zanzibar
        >>> shapefile = quick_setup('zanzibar')
        >>> print(f"Ready: {shapefile}")
    """
    return setup_coastline_for_region(region)


if __name__ == '__main__':
    # If run directly, use CLI
    main()
