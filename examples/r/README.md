# Using SSF AI Toolkit from R

This directory contains examples for using the `ssfaitk` Python library from R via the `reticulate` package.

## Quick Setup

**For detailed local setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

### Quick Start (Local Installation)

1. **Install Python package locally:**
   ```bash
   cd /path/to/ssf-ai-toolkit
   pip install -e .
   ```

2. **Install reticulate in R:**
   ```r
   install.packages("reticulate")
   ```

3. **Configure and test:**
   ```r
   library(reticulate)
   # Point to your Python (adjust path as needed)
   use_python("/path/to/ssf-ai-toolkit/venv/bin/python")  # if using venv
   
   # Test installation
   source("test_setup.R")  # Run the test script
   ```

### Installation from PyPI (when available)

```r
library(reticulate)
py_install("ssfaitk")
ssfaitk <- import("ssfaitk.r_api")
```

## Available Examples

### `effort_prediction_example.R`
Comprehensive example showing:
- Predicting fishing effort from GPS tracks
- Using pre-trained models
- Using statistical (rule-based) classifier
- Training custom models
- Handling custom column names
- Basic visualization with ggplot2

## Quick Start

```r
library(reticulate)

# Import the R-friendly API
ssfaitk <- import("ssfaitk.r_api")

# Load your GPS track data
tracks <- read.csv("your_tracks.csv")

# Predict fishing effort
predictions <- ssfaitk$effort_predict(tracks)

# View results
head(predictions[, c("trip_id", "effort_pred", "effort_prob")])
table(predictions$effort_pred)
```

## Data Requirements

Your input data should have the following columns (flexible naming supported):

### Required Columns
- **Trip ID**: `trip_id`, `Trip_ID`, or `TRIP_ID`
- **Timestamp**: `timestamp`, `time`, `ltime`, or `TIME`
- **Latitude**: `latitude`, `lat`, or `LATITUDE`
- **Longitude**: `longitude`, `lon`, or `LONGITUDE`

### Optional Columns
- **Altitude**: `altitude` or `Altitude`
- **Device Model**: `model`, `device_model`, or `Model`

### Using Custom Column Names

If your data uses different column names, use the `colmap` parameter:

```r
predictions <- ssfaitk$effort_predict(
  df = tracks,
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

```r
# Supervised ML prediction (requires trained model)
ssfaitk$effort_predict(df, model_path = NULL, return_proba = TRUE)

# Statistical rule-based prediction (no training needed)
ssfaitk$effort_predict_statistical(df, trip_col = "trip_id",
                                   lat_col = "latitude",
                                   lon_col = "longitude",
                                   time_col = "timestamp")

# Train custom model
ssfaitk$effort_fit(df, label_col = "Activity", save_path = "model.joblib")
```

### Gear Prediction

```r
# Predict gear type (requires: duration_hours, distance_nm, mean_sog)
ssfaitk$gear_predict(trips_df, model_path = NULL)

# Train custom gear model
ssfaitk$gear_fit(labeled_trips, label_col = "gear_label",
                 save_path = "gear_model.joblib")
```

### Vessel Type Prediction

```r
# Predict vessel type (motorized vs non-motorized)
ssfaitk$vessel_predict(trips_df, model_path = NULL)

# Train custom vessel model
ssfaitk$vessel_fit(labeled_trips, label_col = "vessel_type_label",
                   save_path = "vessel_model.joblib")
```

## Output Format

All prediction functions return R data.frames with the original data plus prediction columns:

### Effort Prediction Output
- `effort_pred`: Binary prediction (0 = non-fishing, 1 = fishing)
- `effort_prob`: Fishing probability (0-1)

### Statistical Effort Prediction Output
- `is_fishing`: Binary prediction (0 or 1)
- `fishing_score`: Continuous fishing likelihood score (0-1)
- Additional feature columns (speed, acceleration, turning behavior, etc.)

### Gear Prediction Output
- `gear_pred`: Predicted gear type (e.g., "gillnet", "handline", "longline")

### Vessel Prediction Output
- `vessel_type_pred`: Predicted vessel type (e.g., "motorized", "non-motorized")

## Tips for R Users

### 1. Data Type Conversion
`reticulate` automatically converts between R and Python data types:
- R `data.frame` ↔ pandas `DataFrame`
- R `vector` ↔ pandas `Series`
- R `list` ↔ Python `dict`

### 2. Handling Timestamps
Ensure timestamps are in a standard format:
```r
tracks$timestamp <- as.character(tracks$timestamp)  # Convert to string
# or
tracks$timestamp <- format(tracks$timestamp, "%Y-%m-%d %H:%M:%S")
```

### 3. Memory Management
For large datasets, consider:
```r
# Process in chunks
chunk_size <- 100000
predictions_list <- list()

for (i in seq(1, nrow(tracks), chunk_size)) {
  end_idx <- min(i + chunk_size - 1, nrow(tracks))
  chunk <- tracks[i:end_idx, ]
  predictions_list[[length(predictions_list) + 1]] <- ssfaitk$effort_predict(chunk)
}

predictions <- do.call(rbind, predictions_list)
```

### 4. Error Handling
Wrap predictions in try-catch for robust code:
```r
result <- tryCatch({
  ssfaitk$effort_predict(tracks)
}, error = function(e) {
  cat("Error:", conditionMessage(e), "\n")
  NULL
})
```

### 5. Checking Available Models
```r
# Check if model file exists before loading
model_path <- "models/my_model.joblib"
if (file.exists(model_path)) {
  predictions <- ssfaitk$effort_predict(tracks, model_path = model_path)
} else {
  cat("Model not found, using default\n")
  predictions <- ssfaitk$effort_predict(tracks)
}
```

## Troubleshooting

### Issue: Module not found
```r
# Check Python configuration
py_config()

# Verify ssfaitk is installed
py_list_packages() |> grep("ssfaitk")

# Reinstall if needed
py_install("ssfaitk", force = TRUE)
```

### Issue: Column not found errors
Check your column names match expected formats or use `colmap`:
```r
# Check your column names
colnames(tracks)

# Use colmap to map your names
predictions <- ssfaitk$effort_predict(
  tracks,
  colmap = list(trip_id = "your_trip_column")
)
```

### Issue: Path errors on Windows
Use forward slashes or double backslashes:
```r
# Good
model_path <- "models/my_model.joblib"
model_path <- "C:/Users/data/model.joblib"

# Bad
model_path <- "C:\Users\data\model.joblib"  # Escape issues
```

## Additional Resources

- Python API documentation: See main README.md
- Column name mappings: See `src/ssfaitk/models/effort/effort_classifier.py`
- Custom thresholds: See `StatisticalEffortClassifier._default_config()`

## Contributing

If you develop useful R wrapper functions or visualizations, please consider contributing them back to the project!
