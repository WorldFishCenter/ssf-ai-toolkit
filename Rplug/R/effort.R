#' Predict fishing effort using a trained model
#'
#' @description
#' Predicts fishing effort on GPS tracks using a machine learning model.
#' Requires a trained model file (`.joblib`) from prior training via `effort_fit()`.
#'
#' @param df Data frame with GPS tracking data. Must contain columns for trip ID,
#'   timestamp, latitude, and longitude. Column names are auto-detected but can be
#'   customized using `colmap`.
#' @param model_path Path to a trained model file (`.joblib`). If `NULL`, attempts
#'   to load a default model (if available).
#' @param return_proba Logical; whether to return probability scores (default: `TRUE`)
#' @param colmap Optional named list mapping standard column names to your actual
#'   column names. Valid keys: `trip_id`, `time`/`timestamp`, `lat`/`latitude`,
#'   `lon`/`longitude`. Example: `list(trip_id = "VesselID", time = "GPS_Time")`
#'
#' @return Data frame with original data plus prediction columns:
#'   - `effort_pred`: Binary prediction (0 = non-fishing, 1 = fishing)
#'   - `effort_prob`: Fishing probability (0-1), if `return_proba = TRUE`
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Basic usage
#' tracks <- read.csv("gps_tracks.csv")
#' predictions <- effort_predict(tracks, model_path = "effort_model.joblib")
#'
#' # With custom column mapping
#' predictions <- effort_predict(
#'   tracks,
#'   model_path = "effort_model.joblib",
#'   colmap = list(
#'     trip_id = "VesselID",
#'     time = "GPS_DateTime",
#'     lat = "Lat_DD",
#'     lon = "Lon_DD"
#'   )
#' )
#'
#' # View results
#' table(predictions$effort_pred)
#' }
effort_predict <- function(df,
                           model_path = NULL,
                           return_proba = TRUE,
                           colmap = NULL) {
  # Get Python module
  ssfaitk <- .get_ssfaitk()

  # Call Python function
  result <- ssfaitk$r_api$effort_predict(
    df = df,
    model_path = model_path,
    return_proba = return_proba,
    colmap = colmap
  )

  return(result)
}

#' Predict fishing effort using statistical (rule-based) classifier
#'
#' @description
#' Predicts fishing effort using a high-performance rule-based classifier that
#' requires no training data. Analyzes GPS movement patterns (speed, turning
#' behavior, path sinuosity, spatial clustering) to identify fishing activity.
#'
#' Uses parallel processing across trips by default for fast computation on
#' large datasets. Includes optional shore filtering using bundled WIO coastline
#' data to remove GPS points on land or too close to shore.
#'
#' @param df Data frame with GPS tracking data. Must contain columns for
#'   latitude, longitude, and timestamp. A trip ID column is recommended
#'   but auto-detected if present.
#' @param filter Logical; remove points on land or within `0.5 km` of the
#'   coastline before classification. Uses bundled WIO coastline. Requires
#'   `geopandas` (`reticulate::py_install("geopandas")`). Default: `FALSE`
#' @param trip_col Column name for trip ID (auto-detected if `NULL`)
#' @param lat_col Column name for latitude (auto-detected if `NULL`)
#' @param lon_col Column name for longitude (auto-detected if `NULL`)
#' @param time_col Column name for timestamp (auto-detected if `NULL`)
#' @param colmap Optional named list for column mapping (alternative to
#'   individual `*_col` parameters). Valid keys: `trip_id`, `time`/`timestamp`,
#'   `lat`/`latitude`, `lon`/`longitude`.
#'   Example: `list(trip_id = "VesselID", time = "GPS_Time", lat = "Lat", lon = "Lon")`
#' @param use_parallel Logical; enable parallel processing across trips
#'   (default: `TRUE`). Recommended for datasets with more than 10 trips.
#' @param n_jobs Integer; number of parallel workers. `-1L` uses all available
#'   CPUs (default: `-1L`).
#' @param config Optional named list to override behavioral classification
#'   thresholds. All values are optional — only specify what you want to change:
#'   \describe{
#'     \item{`min_fishing_speed`}{Min speed classified as fishing, km/h (default: `0.5`)}
#'     \item{`max_fishing_speed`}{Max speed classified as fishing, km/h (default: `8.0`)}
#'     \item{`min_transit_speed`}{Speed threshold for transit, km/h (default: `12.0`)}
#'     \item{`high_turn_threshold`}{Turning angle indicating fishing, degrees (default: `45.0`)}
#'     \item{`low_straightness_threshold`}{Path straightness below = fishing (default: `0.4`)}
#'     \item{`high_sinuosity_threshold`}{Sinuosity above = fishing (default: `1.5`)}
#'     \item{`clustering_radius_km`}{Spatial clustering radius, km (default: `0.5`)}
#'     \item{`time_windows`}{Rolling window sizes in minutes, e.g. `list(5, 10, 30)` (default: `list(10.0)`)}
#'     \item{`spatial_window_km`}{Distance-based feature window, km (default: `1.0`)}
#'     \item{`min_state_duration`}{Min consecutive points for a state (default: `3L`)}
#'     \item{`fishing_score_threshold`}{Score above this = fishing, range 0-1 (default: `0.5`)}
#'     \item{`weight_speed`}{Scoring weight for speed indicator (default: `3.0`)}
#'     \item{`weight_turning`}{Scoring weight for turning indicator (default: `2.0`)}
#'     \item{`weight_straightness`}{Scoring weight for path straightness (default: `2.0`)}
#'     \item{`weight_sinuosity`}{Scoring weight for sinuosity (default: `1.5`)}
#'     \item{`weight_clustering`}{Scoring weight for spatial clustering (default: `2.0`)}
#'     \item{`weight_speed_variability`}{Scoring weight for speed variability (default: `1.5`)}
#'   }
#'
#' @return Data frame with original columns plus:
#'   \describe{
#'     \item{`is_fishing`}{Binary prediction: 1 = fishing, 0 = not fishing}
#'     \item{`fishing_score`}{Continuous likelihood score (0-1)}
#'     \item{`activity_type`}{Category: `"fishing"`, `"sailing"`, `"starting_trip"`, `"ending_trip"`}
#'     \item{`trip_phase`}{Phase: `"starting"`, `"in_progress"`, `"ending"`}
#'     \item{Additional feature columns}{Speed, acceleration, turning behavior, sinuosity, etc.}
#'   }
#'
#' @export
#'
#' @examples
#' \dontrun{
#' tracks <- read.csv("gps_tracks.csv")
#'
#' # Basic usage - no model or training required
#' predictions <- effort_predict_statistical(tracks)
#'
#' # With shore filtering
#' predictions <- effort_predict_statistical(tracks, filter = TRUE)
#'
#' # Custom column names
#' predictions <- effort_predict_statistical(
#'   tracks,
#'   colmap = list(trip_id = "VesselID", time = "GPS_Time", lat = "Lat", lon = "Lon")
#' )
#'
#' # Tune thresholds for a specific fishery (e.g. faster vessels)
#' predictions <- effort_predict_statistical(
#'   tracks,
#'   config = list(
#'     max_fishing_speed = 12.0,
#'     fishing_score_threshold = 0.6
#'   )
#' )
#'
#' # Disable parallelism (useful in constrained environments)
#' predictions <- effort_predict_statistical(tracks, use_parallel = FALSE)
#'
#' # Use 4 CPU cores instead of all
#' predictions <- effort_predict_statistical(tracks, n_jobs = 4L)
#' }
effort_predict_statistical <- function(df,
                                       filter = FALSE,
                                       trip_col = NULL,
                                       lat_col = NULL,
                                       lon_col = NULL,
                                       time_col = NULL,
                                       colmap = NULL,
                                       use_parallel = TRUE,
                                       n_jobs = -1L,
                                       config = NULL) {
  # Get Python module
  ssfaitk <- .get_ssfaitk()

  # Call Python function (use apply_filter to avoid Python built-in name conflict)
  result <- ssfaitk$r_api$effort_predict_statistical(
    df = df,
    apply_filter = filter,
    trip_col = trip_col,
    lat_col = lat_col,
    lon_col = lon_col,
    time_col = time_col,
    colmap = colmap,
    use_parallel = use_parallel,
    n_jobs = as.integer(n_jobs),
    config = config
  )

  return(result)
}

#' Train a custom fishing effort prediction model
#'
#' @description
#' Trains a machine learning model to predict fishing effort from GPS tracks.
#' Requires labeled training data with activity classifications (e.g., "Fishing",
#' "Sailing", "Traveling"). The trained model can be saved and used with
#' `effort_predict()`.
#'
#' @param df Data frame with GPS tracking data and activity labels
#' @param label_col Column name containing activity labels. Expected values:
#'   "Fishing", "Searching" (classified as fishing), "Sailing", "Traveling"
#'   (classified as non-fishing). Default: `"Activity"`
#' @param save_path Optional path to save the trained model (`.joblib` file).
#'   Creates parent directories if needed.
#' @param trip_col Column name for trip ID (auto-detected if `NULL`)
#' @param lat_col Column name for latitude (auto-detected if `NULL`)
#' @param lon_col Column name for longitude (auto-detected if `NULL`)
#' @param time_col Column name for timestamp (auto-detected if `NULL`)
#' @param colmap Optional named list for column mapping
#'
#' @return Trained model object (Python EffortClassifier). Can be used directly
#'   or saved via the `save_path` parameter.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Load labeled training data
#' training_data <- read.csv("labeled_tracks.csv")
#'
#' # Train and save model
#' model <- effort_fit(
#'   training_data,
#'   label_col = "Activity",
#'   save_path = "my_effort_model.joblib"
#' )
#'
#' # Later, use the model for predictions
#' new_tracks <- read.csv("new_tracks.csv")
#' predictions <- effort_predict(new_tracks, model_path = "my_effort_model.joblib")
#'
#' # Custom column names
#' model <- effort_fit(
#'   training_data,
#'   label_col = "ActType",
#'   colmap = list(
#'     trip_id = "VesselID",
#'     time = "GPS_Time"
#'   ),
#'   save_path = "custom_model.joblib"
#' )
#' }
effort_fit <- function(df,
                       label_col = "Activity",
                       save_path = NULL,
                       trip_col = NULL,
                       lat_col = NULL,
                       lon_col = NULL,
                       time_col = NULL,
                       colmap = NULL) {
  # Get Python module
  ssfaitk <- .get_ssfaitk()

  # Call Python function
  model <- ssfaitk$r_api$effort_fit(
    df = df,
    label_col = label_col,
    save_path = save_path,
    trip_col = trip_col,
    lat_col = lat_col,
    lon_col = lon_col,
    time_col = time_col,
    colmap = colmap
  )

  return(model)
}
