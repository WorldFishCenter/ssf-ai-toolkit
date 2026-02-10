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
#' Predicts fishing effort using a rule-based classifier that doesn't require training.
#' Analyzes movement patterns (speed, acceleration, turning behavior) to identify
#' fishing activity. This is useful when no labeled training data is available.
#'
#' Includes optional shore distance filtering to remove GPS points that are on land
#' or too close to shore, which helps prevent false positives from coastal navigation.
#'
#' @param df Data frame with GPS tracking data
#' @param filter Logical; enable shore distance filtering to remove on-land and
#'   near-shore points. Requires coastline data. Default: `FALSE`
#' @param trip_col Column name for trip ID (auto-detected if `NULL`)
#' @param lat_col Column name for latitude (auto-detected if `NULL`)
#' @param lon_col Column name for longitude (auto-detected if `NULL`)
#' @param time_col Column name for timestamp (auto-detected if `NULL`)
#' @param colmap Optional named list for column mapping (alternative to individual parameters)
#' @param config Optional named list with behavioral thresholds for classification.
#'   See Python documentation for available parameters.
#'
#' @return Data frame with original data plus prediction columns:
#'   - `is_fishing`: Binary prediction (0 or 1)
#'   - `fishing_score`: Continuous fishing likelihood score (0-1)
#'   - Additional feature columns (speed, acceleration, turning behavior, etc.)
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Basic usage - no model required!
#' tracks <- read.csv("gps_tracks.csv")
#' predictions <- effort_predict_statistical(tracks)
#'
#' # With shore filtering (removes on-land/near-shore points)
#' predictions <- effort_predict_statistical(tracks, filter = TRUE)
#'
#' # View fishing activity summary
#' table(predictions$is_fishing)
#'
#' # With custom column mapping
#' predictions <- effort_predict_statistical(
#'   tracks,
#'   filter = TRUE,
#'   colmap = list(
#'     trip_id = "VesselID",
#'     time = "GPS_DateTime",
#'     lat = "Lat",
#'     lon = "Lon"
#'   )
#' )
#' }
effort_predict_statistical <- function(df,
                                       filter = FALSE,
                                       trip_col = NULL,
                                       lat_col = NULL,
                                       lon_col = NULL,
                                       time_col = NULL,
                                       colmap = NULL,
                                       config = NULL) {
  # Get Python module
  ssfaitk <- .get_ssfaitk()

  # Call Python function
  result <- ssfaitk$r_api$effort_predict_statistical(
    df = df,
    filter = filter,
    trip_col = trip_col,
    lat_col = lat_col,
    lon_col = lon_col,
    time_col = time_col,
    colmap = colmap,
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
