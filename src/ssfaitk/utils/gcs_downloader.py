# src/ssfaitk/data/gcs_downloader.py
"""
Google Cloud Storage downloader for SSF AI Toolkit.

Downloads trip data from GCS buckets with smart skip logic to avoid
re-downloading existing files.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger
# from ssfaitk.utils.logger import get_logger

logger = get_logger(__name__)

# Optional dependencies
try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning(
        "google-cloud-storage not installed. "
        "Install with: pip install google-cloud-storage"
    )

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@dataclass
class DownloadResult:
    """Results from a download operation."""
    downloaded: List[str]
    skipped: List[str]
    failed: List[str]

    @property
    def total(self) -> int:
        """Total files processed."""
        return len(self.downloaded) + len(self.skipped) + len(self.failed)

    @property
    def success_rate(self) -> float:
        """Success rate (0-1)."""
        if self.total == 0:
            return 0.0
        return len(self.downloaded) / self.total

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Downloaded: {len(self.downloaded)}, "
            f"Skipped: {len(self.skipped)}, "
            f"Failed: {len(self.failed)}"
        )


def _check_gcs_available() -> None:
    """Raise error if GCS library not available."""
    if not GCS_AVAILABLE:
        raise ImportError(
            "google-cloud-storage is required for GCS downloads. "
            "Install with: pip install google-cloud-storage"
        )


def get_existing_trip_ids(
        country_name: str,
        base_dir: str | Path = "../data/pds-trips"
) -> Set[str]:
    """
    Get set of trip IDs already downloaded for a country.

    Extracts trip IDs from parquet filenames in the local directory.
    Assumes filename format: pds-tracks_{trip_id}.parquet

    Args:
        country_name: Country name (used as subdirectory)
        base_dir: Base directory for trip data

    Returns:
        Set of trip IDs already present locally

    Example:
        >>> existing = get_existing_trip_ids('timor-leste')
        >>> print(f"Found {len(existing)} existing trips")
    """
    local_dir = Path(base_dir) / country_name

    if not local_dir.exists():
        logger.info(f"Directory does not exist yet: {local_dir}")
        return set()

    trip_ids = set()
    for file in local_dir.glob('*.parquet'):
        # Extract trip ID from filename
        # Example: pds-tracks_14232947.parquet -> 14232947
        parts = file.stem.split('_')
        if len(parts) > 1:
            trip_id = parts[-1]
            trip_ids.add(trip_id)
        else:
            # Fallback: use entire stem as trip ID
            trip_ids.add(file.stem)

    logger.info(f"Found {len(trip_ids)} existing trip IDs in {local_dir}")
    return trip_ids


def list_gcs_files(
        bucket_name: str,
        prefix: str = "",
        suffix: str = ".parquet",
        credentials_path: Optional[str | Path] = None
) -> List[str]:
    """
    List files in a GCS bucket.

    Args:
        bucket_name: Name of the GCS bucket
        prefix: Optional prefix to filter files (e.g., 'pds-tracks/')
        suffix: File extension to filter (default: .parquet)
        credentials_path: Path to service account JSON key file

    Returns:
        List of file paths in the bucket

    Example:
        >>> files = list_gcs_files(
        ...     bucket_name='my-bucket',
        ...     prefix='pds-tracks/timor-leste/',
        ...     credentials_path='key.json'
        ... )
        >>> print(f"Found {len(files)} files")
    """
    _check_gcs_available()

    # Set credentials
    if credentials_path:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_path)

    # Initialize client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all blobs with prefix
    blobs = bucket.list_blobs(prefix=prefix)

    # Filter by suffix
    files = [blob.name for blob in blobs if blob.name.endswith(suffix)]

    logger.info(f"Found {len(files)} files in gs://{bucket_name}/{prefix}")
    return files


def download_gcs_file(
        bucket_name: str,
        source_blob_name: str,
        destination_file: str | Path,
        credentials_path: Optional[str | Path] = None
) -> Path:
    """
    Download a single file from GCS.

    Args:
        bucket_name: Name of the GCS bucket
        source_blob_name: Path to file in bucket
        destination_file: Local path to save file
        credentials_path: Path to service account JSON key

    Returns:
        Path to downloaded file

    Raises:
        Exception: If download fails

    Example:
        >>> download_gcs_file(
        ...     bucket_name='my-bucket',
        ...     source_blob_name='pds-tracks/file.parquet',
        ...     destination_file='data/file.parquet',
        ...     credentials_path='key.json'
        ... )
    """
    _check_gcs_available()

    if credentials_path:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_path)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    destination_path = Path(destination_file)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    blob.download_to_filename(destination_path)
    logger.info(f"Downloaded: {source_blob_name} -> {destination_path}")

    return destination_path


def download_missing_trips(
        bucket_name: str,
        gcs_prefix: str,
        country_name: str,
        credentials_path: Optional[str | Path] = None,
        base_dir: str | Path = "../data/pds-trips",
        show_progress: bool = True
) -> DownloadResult:
    """
    Download only trips with IDs not already in local storage.

    This function:
    1. Scans local directory for existing trip files
    2. Extracts trip IDs from filenames
    3. Lists files in GCS bucket
    4. Downloads only files with new trip IDs

    Args:
        bucket_name: GCS bucket name
        gcs_prefix: Prefix/folder in bucket (e.g., 'pds-tracks/timor-leste/')
        country_name: Country name for local directory structure
        credentials_path: Path to service account JSON key file
        base_dir: Base directory for downloads (default: ../data/pds-trips)
        show_progress: Show progress bar (requires tqdm)

    Returns:
        DownloadResult object with lists of downloaded, skipped, and failed files

    Example:
        >>> result = download_missing_trips(
        ...     bucket_name='fisheries-data',
        ...     gcs_prefix='pds-tracks/timor-leste/',
        ...     country_name='timor-leste',
        ...     credentials_path='service-account-key.json'
        ... )
        >>> print(result.summary())
        Downloaded: 5, Skipped: 10, Failed: 0
    """
    _check_gcs_available()

    logger.info(f"Starting download for country: {country_name}")
    logger.info(f"GCS bucket: gs://{bucket_name}/{gcs_prefix}")

    # Get existing trip IDs
    existing_ids = get_existing_trip_ids(country_name, base_dir)
    logger.info(f"Found {len(existing_ids)} existing trip IDs locally")

    # Create local directory
    local_dir = Path(base_dir) / country_name
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Local directory: {local_dir}")

    # Set credentials
    if credentials_path:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_path)

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List files in bucket
    logger.info("Listing files in GCS bucket...")
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    parquet_blobs = [b for b in blobs if b.name.endswith('.parquet')]
    logger.info(f"Found {len(parquet_blobs)} parquet files in bucket")

    # Initialize results
    result = DownloadResult(downloaded=[], skipped=[], failed=[])

    # Setup progress bar
    if show_progress and TQDM_AVAILABLE:
        iterator = tqdm(parquet_blobs, desc=f"Downloading {country_name}")
    else:
        iterator = parquet_blobs

    # Process each file
    for blob in iterator:
        filename = Path(blob.name).name

        # Extract trip ID from filename
        # Example: pds-tracks_14232947.parquet -> 14232947
        parts = filename.split('_')
        if len(parts) > 1:
            trip_id = parts[-1].replace('.parquet', '')
        else:
            trip_id = filename.replace('.parquet', '')

        # Skip if trip ID already exists
        if trip_id in existing_ids:
            logger.debug(f"Skipping trip {trip_id} (already exists)")
            result.skipped.append(trip_id)
            continue

        # Download new trip
        local_path = local_dir / filename
        try:
            logger.info(f"Downloading trip {trip_id}: {filename}")
            blob.download_to_filename(local_path)
            result.downloaded.append(str(local_path))
            existing_ids.add(trip_id)  # Add to set for this session
            logger.info(f"✓ Saved: {local_path}")
        except Exception as e:
            logger.error(f"✗ Failed to download trip {trip_id}: {e}")
            result.failed.append(trip_id)

    # Log summary
    logger.info("=" * 60)
    logger.info(f"Download Summary for {country_name}:")
    logger.info(f"  New trips downloaded: {len(result.downloaded)}")
    logger.info(f"  Trips skipped (existing): {len(result.skipped)}")
    logger.info(f"  Failed downloads: {len(result.failed)}")
    logger.info("=" * 60)

    return result


def download_trips_for_countries(
        bucket_name: str,
        countries: List[str],
        credentials_path: Optional[str | Path] = None,
        base_dir: str | Path = "../data/pds-trips",
        gcs_prefix_template: str = "pds-tracks/{country}/",
        show_progress: bool = True
) -> Dict[str, DownloadResult]:
    """
    Download trips for multiple countries.

    Args:
        bucket_name: GCS bucket name
        countries: List of country names (must match GCS folder structure)
        credentials_path: Path to service account key
        base_dir: Base directory for downloads
        gcs_prefix_template: Template for GCS prefix (use {country} placeholder)
        show_progress: Show progress bars

    Returns:
        Dictionary mapping country names to DownloadResult objects

    Example:
        >>> results = download_trips_for_countries(
        ...     bucket_name='fisheries-data',
        ...     countries=['timor-leste', 'kenya', 'zanzibar'],
        ...     credentials_path='key.json'
        ... )
        >>> for country, result in results.items():
        ...     print(f"{country}: {result.summary()}")
    """
    _check_gcs_available()

    logger.info(f"Starting multi-country download for {len(countries)} countries")

    all_results = {}

    for i, country in enumerate(countries, 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Processing {i}/{len(countries)}: {country.upper()}")
        logger.info("=" * 60)

        gcs_prefix = gcs_prefix_template.format(country=country)

        try:
            result = download_missing_trips(
                bucket_name=bucket_name,
                gcs_prefix=gcs_prefix,
                country_name=country,
                credentials_path=credentials_path,
                base_dir=base_dir,
                show_progress=show_progress
            )
            all_results[country] = result
        except Exception as e:
            logger.error(f"Failed to process {country}: {e}")
            all_results[country] = DownloadResult(
                downloaded=[],
                skipped=[],
                failed=['ALL']
            )

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY - ALL COUNTRIES")
    logger.info("=" * 60)
    total_downloaded = sum(len(r.downloaded) for r in all_results.values())
    total_skipped = sum(len(r.skipped) for r in all_results.values())
    total_failed = sum(len(r.failed) for r in all_results.values())

    logger.info(f"Total downloaded: {total_downloaded}")
    logger.info(f"Total skipped: {total_skipped}")
    logger.info(f"Total failed: {total_failed}")
    logger.info("")

    for country, result in all_results.items():
        logger.info(f"  {country}: {result.summary()}")

    logger.info("=" * 60)

    return all_results


def sync_bucket_to_local(
        bucket_name: str,
        gcs_prefix: str,
        local_dir: str | Path,
        credentials_path: Optional[str | Path] = None,
        file_extension: str = ".parquet",
        show_progress: bool = True
) -> DownloadResult:
    """
    Sync all files from GCS bucket to local directory.

    Downloads all files matching the extension that don't already exist locally.
    This is a simpler version that just checks filenames, not trip IDs.

    Args:
        bucket_name: GCS bucket name
        gcs_prefix: Prefix/folder in bucket
        local_dir: Local directory to download files
        credentials_path: Path to service account JSON key
        file_extension: File extension to download (default: .parquet)
        show_progress: Show progress bar

    Returns:
        DownloadResult object

    Example:
        >>> result = sync_bucket_to_local(
        ...     bucket_name='my-bucket',
        ...     gcs_prefix='data/trips/',
        ...     local_dir='./data/trips',
        ...     credentials_path='key.json'
        ... )
        >>> print(f"Downloaded {len(result.downloaded)} new files")
    """
    _check_gcs_available()

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    # Get existing local files
    existing_files = {f.name for f in local_path.glob(f'*{file_extension}')}
    logger.info(f"Found {len(existing_files)} existing files in {local_path}")

    # Set credentials
    if credentials_path:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_path)

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List files in bucket
    logger.info(f"Listing files in gs://{bucket_name}/{gcs_prefix}")
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    target_blobs = [b for b in blobs if b.name.endswith(file_extension)]
    logger.info(f"Found {len(target_blobs)} files in bucket")

    # Initialize results
    result = DownloadResult(downloaded=[], skipped=[], failed=[])

    # Setup progress bar
    if show_progress and TQDM_AVAILABLE:
        iterator = tqdm(target_blobs, desc="Syncing files")
    else:
        iterator = target_blobs

    # Process each file
    for blob in iterator:
        filename = Path(blob.name).name
        local_file = local_path / filename

        # Skip if exists
        if filename in existing_files:
            logger.debug(f"Skipping {filename} (exists)")
            result.skipped.append(filename)
            continue

        # Download
        try:
            logger.info(f"Downloading: {filename}")
            blob.download_to_filename(local_file)
            result.downloaded.append(str(local_file))
            logger.info(f"✓ Saved: {local_file}")
        except Exception as e:
            logger.error(f"✗ Failed to download {filename}: {e}")
            result.failed.append(filename)

    logger.info(f"\nSync complete: {result.summary()}")
    return result


# Convenience function for common use case
def download_pds_tracks(
        country: str,
        credentials_path: str | Path,
        bucket_name: Optional[str] = None,
        base_dir: str | Path = "../data/pds-trips",
        gcs_prefix: str = ""
) -> DownloadResult:
    """
    Convenience function to download PDS tracks for a single country.

    Supports two GCS structures:
    1. Country-specific buckets: pds-{country} (default)
    2. Single bucket with folders: {bucket}/pds-tracks/{country}/

    Args:
        country: Country name (e.g., 'zanzibar', 'timor-leste', 'kenya')
        credentials_path: Path to service account JSON key
        bucket_name: GCS bucket name (optional, defaults to 'pds-{country}')
        base_dir: Base directory for downloads
        gcs_prefix: Prefix within bucket (default: '' for root)

    Returns:
        DownloadResult object

    Examples:
        >>> # Country-specific bucket (default)
        >>> result = download_pds_tracks(
        ...     country='zanzibar',
        ...     credentials_path='key.json'
        ... )
        >>> # Uses bucket: 'pds-zanzibar'

        >>> # Custom bucket structure
        >>> result = download_pds_tracks(
        ...     country='timor-leste',
        ...     credentials_path='key.json',
        ...     bucket_name='fisheries-data',
        ...     gcs_prefix='pds-tracks/timor-leste/'
        ... )
    """
    # Default: country-specific bucket (e.g., 'pds-zanzibar')
    if bucket_name is None:
        bucket_name = f"pds-{country}"
        logger.info(f"Using default bucket name: {bucket_name}")

    return download_missing_trips(
        bucket_name=bucket_name,
        gcs_prefix=gcs_prefix,
        country_name=country,
        credentials_path=credentials_path,
        base_dir=base_dir,
        show_progress=True
    )


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 4:
        print("Usage: python gcs_downloader.py <bucket> <country> <credentials_path>")
        print("\nExample:")
        print("  python gcs_downloader.py my-bucket timor-leste key.json")
        sys.exit(1)

    bucket = sys.argv[1]
    country = sys.argv[2]
    creds = sys.argv[3]

    print(f"Downloading trips for {country} from {bucket}...")
    result = download_pds_tracks(
        bucket_name=bucket,
        country=country,
        credentials_path=creds
    )

    print(f"\n✓ Complete: {result.summary()}")