# ssfaitk: Small-Scale Fisheries AI Toolkit for R

R interface to the SSF AI Toolkit Python library for analyzing small-scale fisheries data using machine learning.

## Features

- **Fishing Effort Classification**: Distinguish fishing vs. non-fishing activity from GPS tracks
  - **Shore Distance Filtering**: Remove on-land and near-shore GPS points to prevent false positives
- **Gear Type Prediction**: Classify fishing gear types (gillnet, handline, longline, etc.)
- **Vessel Type Prediction**: Classify vessel types (motorized vs. non-motorized)
- **Seamless Python Integration**: Uses `reticulate` for automatic data conversion
- **Flexible Column Mapping**: Automatically detects common column name variations
- **Remote Environment Support**: Works in GitHub Actions, Docker, and cloud compute

## Installation

### Prerequisites

1. **R** (>= 4.0.0)
2. **Python** (>= 3.9)
3. **reticulate** R package

### Install from GitHub

```r
# Install devtools if needed
install.packages("devtools")

# Install the R package
devtools::install_github("WorldFishCenter/ssf-ai-toolkit", subdir = "Rplug")
```

### Install Python Dependencies

After installing the R package, you need to install the Python `ssfaitk` package:

```r
library(ssfaitk)

# Option 1: Install from local source (for development)
install_ssfaitk()

# Option 2: Install from GitHub
install_ssfaitk(
  package_url = "git+https://github.com/WorldFishCenter/ssf-ai-toolkit.git"
)

# Verify installation
ssfaitk_available()  # Should return TRUE
```

## Quick Start

### Fishing Effort Classification

```r
library(ssfaitk)

# Load GPS tracking data
tracks <- read.csv("gps_tracks.csv")
# Required columns: trip_id, timestamp, latitude, longitude

# Option 1: Statistical prediction (no training required!)
predictions <- effort_predict_statistical(tracks)
head(predictions)
table(predictions$is_fishing)

# With shore filtering (removes on-land/near-shore points)
predictions_filtered <- effort_predict_statistical(tracks, filter = TRUE)

# With custom column names
custom_tracks <- read.csv("custom_data.csv")
predictions_custom <- effort_predict_statistical(
  custom_tracks,
  filter = TRUE,
  colmap = list(
    trip_id = "VesselID",
    time = "GPS_DateTime",
    lat = "Lat_DD",
    lon = "Lon_DD"
  )
)

# Option 2: Train a custom model
labeled_tracks <- read.csv("labeled_tracks.csv")
# Additional column: Activity ("Fishing", "Searching", "Sailing", "Traveling")

model <- effort_fit(
  labeled_tracks,
  label_col = "Activity",
  save_path = "my_effort_model.joblib"
)

# Use the trained model
new_tracks <- read.csv("new_tracks.csv")
predictions <- effort_predict(new_tracks, model_path = "my_effort_model.joblib")
head(predictions[, c("trip_id", "effort_pred", "effort_prob")])
```

### Gear Type Prediction

```r
# Load trip-level data
trips <- read.csv("trip_features.csv")
# Required columns: duration_hours, distance_nm, mean_sog

# Train a gear prediction model
labeled_trips <- trips
labeled_trips$gear_label <- sample(
  c("gillnet", "handline", "longline"),
  nrow(trips),
  replace = TRUE
)

model <- gear_fit(
  labeled_trips,
  label_col = "gear_label",
  save_path = "gear_model.joblib"
)

# Predict gear types
new_trips <- read.csv("new_trips.csv")
predictions <- gear_predict(new_trips, model_path = "gear_model.joblib")
table(predictions$gear_pred)
```

### Vessel Type Prediction

```r
# Train a vessel type model
labeled_trips$vessel_type_label <- sample(
  c("motorized", "non-motorized"),
  nrow(labeled_trips),
  replace = TRUE
)

model <- vessel_fit(
  labeled_trips,
  label_col = "vessel_type_label",
  save_path = "vessel_model.joblib"
)

# Predict vessel types
predictions <- vessel_predict(new_trips, model_path = "vessel_model.joblib")
table(predictions$vessel_type_pred)
```

## Data Requirements

### Effort Classification (Point-Level GPS Data)

Your data frame should contain:

- **Trip ID**: `trip_id`, `Trip_ID`, `TRIP_ID`, or `tripid`
- **Timestamp**: `timestamp`, `time`, `ltime`, `datetime`, or `date_time`
- **Latitude**: `latitude`, `Latitude`, `lat`, `LAT`, or `y`
- **Longitude**: `longitude`, `Longitude`, `lon`, `LON`, `lng`, or `x`

Column names are **auto-detected**! The toolkit recognizes common variations.

### Gear & Vessel Prediction (Trip-Level Data)

Your data frame must contain (exact names):

- `duration_hours`: Trip duration in hours
- `distance_nm`: Total distance in nautical miles
- `mean_sog`: Mean speed over ground

## Shore Distance Filtering

The statistical effort classifier includes optional shore distance filtering to remove GPS points that are on land or too close to shore. This helps prevent false positives from coastal navigation, vessel berthing, or GPS errors.

```r
# Enable shore filtering
predictions <- effort_predict_statistical(tracks, filter = TRUE)

# Without filtering (may include on-land/near-shore points)
predictions_unfiltered <- effort_predict_statistical(tracks, filter = FALSE)
```

**When to use shore filtering:**
- ✅ When analyzing coastal fisheries with frequent near-shore navigation
- ✅ When GPS tracks include port entry/exit activities
- ✅ To reduce false positives from vessels alongside docks
- ❌ When analyzing deep-sea/offshore fisheries (minimal benefit)
- ❌ If coastline data is not available for your region

**Requirements:**
- Coastline data must be available in the Python package
- Adds processing time proportional to number of GPS points

**Advanced usage:**

The `effort_predict_statistical()` function accepts additional parameters:

```r
predictions <- effort_predict_statistical(
  df = tracks,
  filter = TRUE,                    # Enable shore filtering
  trip_col = NULL,                  # Custom trip ID column (auto-detected)
  lat_col = NULL,                   # Custom latitude column (auto-detected)
  lon_col = NULL,                   # Custom longitude column (auto-detected)
  time_col = NULL,                  # Custom time column (auto-detected)
  colmap = NULL,                    # Named list for column mapping
  config = NULL                     # Custom behavioral thresholds (advanced)
)
```

### Customizing Behavioral Thresholds with `config`

The `config` parameter allows fine-tuning behavioral thresholds and shore filtering parameters.

**Default shore distance:** GPS points within **0.5 km (500 meters)** of the coastline are filtered when `filter = TRUE`.

**Example: Customize shore distance threshold**

```r
# Default shore filtering uses 0.5 km (500 meters) from coast
predictions_default <- effort_predict_statistical(tracks, filter = TRUE)

# Increase threshold to 1 km (more conservative - filters more points)
predictions_1km <- effort_predict_statistical(
  tracks,
  filter = TRUE,
  config = list(shore_min_distance_km = 1.0)
)

# Decrease threshold to 0.2 km (less conservative - filters fewer points)
predictions_200m <- effort_predict_statistical(
  tracks,
  filter = TRUE,
  config = list(shore_min_distance_km = 0.2)
)
```

**Example: Customize behavioral thresholds**

```r
# Adjust fishing speed range (default: 0.5 - 8.0 km/h)
custom_config <- list(
  min_fishing_speed = 1.0,        # Minimum speed for fishing (km/h)
  max_fishing_speed = 10.0,       # Maximum speed for fishing (km/h)
  shore_min_distance_km = 0.75,   # Shore distance threshold (km)
  fishing_score_threshold = 0.6   # Higher threshold = more conservative
)

predictions <- effort_predict_statistical(tracks, filter = TRUE, config = custom_config)
```

**Available configuration parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Shore Filtering** | | |
| `shore_min_distance_km` | 0.5 | Minimum distance from coast (km) |
| `shore_filter_land_points` | TRUE | Remove points on land |
| `shore_filter_only_fishing` | TRUE | Only filter fishing points (faster) |
| **Speed Thresholds (km/h)** | | |
| `min_fishing_speed` | 0.5 | Minimum speed for fishing |
| `max_fishing_speed` | 8.0 | Maximum speed for fishing |
| `min_transit_speed` | 12.0 | Minimum speed for transit |
| **Turning Behavior** | | |
| `high_turn_threshold` | 45.0 | High turning angle threshold (degrees) |
| `min_distance_for_turn` | 0.1 | Minimum distance to calculate turn (km) |
| **Path Characteristics** | | |
| `low_straightness_threshold` | 0.4 | Low path straightness threshold |
| `high_sinuosity_threshold` | 1.5 | High path sinuosity threshold |
| `clustering_radius_km` | 0.5 | Spatial clustering radius (km) |
| **Multi-scale Windows** | | |
| `time_windows` | c(10.0) | Temporal window sizes (minutes) |
| `spatial_window_km` | 1.0 | Spatial window size (km) |
| **Classification** | | |
| `fishing_score_threshold` | 0.5 | Minimum score to classify as fishing |

See the [Python source code](https://github.com/WorldFishCenter/ssf-ai-toolkit/blob/main/src/ssfaitk/models/effort/statistical_effort_v2.py#L395-L439) for the complete configuration reference.

## Custom Column Mapping

If your data uses different column names, use the `colmap` parameter:

```r
# Your data has non-standard column names
custom_tracks <- read.csv("custom_data.csv")
# Columns: VesselID, GPS_DateTime, Lat_DD, Lon_DD

predictions <- effort_predict(
  custom_tracks,
  model_path = "model.joblib",
  colmap = list(
    trip_id = "VesselID",
    time = "GPS_DateTime",
    lat = "Lat_DD",
    lon = "Lon_DD"
  )
)
```

## Available Functions

### Effort Classification
- `effort_predict()` - Predict fishing effort using ML model
- `effort_predict_statistical()` - Rule-based prediction (no training needed)
- `effort_fit()` - Train custom effort model

### Gear Prediction
- `gear_predict()` - Predict gear type
- `gear_fit()` - Train custom gear model

### Vessel Prediction
- `vessel_predict()` - Predict vessel type
- `vessel_fit()` - Train custom vessel model

### Utilities
- `ssfaitk_available()` - Check if Python package is installed
- `install_ssfaitk()` - Install Python package
- `check_python_env()` - Display Python environment info

## Remote Environment Usage

### GitHub Actions

```yaml
name: Run Fisheries Analysis

on: [push]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup R
        uses: r-lib/actions/setup-r@v2

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Python package
        run: pip install -e .

      - name: Install R package
        run: Rscript -e 'devtools::install("Rplug", dependencies=TRUE)'

      - name: Run analysis
        run: Rscript analysis.R
```

### Docker

```dockerfile
FROM rocker/r-ver:4.3

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Install Python package
COPY . /app
WORKDIR /app
RUN pip3 install -e .

# Install R package
RUN R -e 'install.packages("devtools")'
RUN R -e 'devtools::install("Rplug")'

CMD ["R"]
```

## Troubleshooting

### Python Module Not Found

```r
# Check Python configuration
library(reticulate)
py_config()

# Verify ssfaitk installation
ssfaitk_available()

# Reinstall if needed
install_ssfaitk()
```

### Column Not Found Errors

```r
# Check your column names
colnames(your_data)

# Use colmap for custom names
predictions <- effort_predict(
  your_data,
  model_path = "model.joblib",
  colmap = list(trip_id = "your_trip_column")
)
```

### Path Issues (Windows)

Use forward slashes or double backslashes:

```r
# Good
model_path <- "models/my_model.joblib"
model_path <- "C:/Users/data/model.joblib"

# Avoid
model_path <- "C:\Users\data\model.joblib"  # Escape issues
```

## Documentation

- Function documentation: `?effort_predict`, `?gear_predict`, etc.
- Vignettes: `browseVignettes("ssfaitk")`
  - Getting Started
  - Effort Classification Workflow
  - Gear Prediction Workflow
  - Vessel Prediction Workflow
  - Remote Deployment Guide

## Contributing

Contributions are welcome! Please see the main repository for guidelines.

## License

MIT License

## Citation

If you use this package in your research, please cite:

```
[Citation information to be added]
```

## Support

- **Issues**: https://github.com/WorldFishCenter/ssf-ai-toolkit/issues
- **Discussions**: https://github.com/WorldFishCenter/ssf-ai-toolkit/discussions
- **Documentation**: https://WorldFishCenter.github.io/ssf-ai-toolkit/

## Related Packages

- **Python Package**: The underlying Python toolkit (see main README)
- **reticulate**: R interface to Python (https://rstudio.github.io/reticulate/)
