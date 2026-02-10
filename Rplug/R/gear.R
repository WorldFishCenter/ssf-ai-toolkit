#' Predict fishing gear type from trip-level features
#'
#' @description
#' Predicts fishing gear type (e.g., gillnet, handline, longline) from aggregated
#' trip-level features. Requires a trained model file (`.joblib`) from prior
#' training via `gear_fit()`.
#'
#' @param df Data frame with trip-level features. Must contain columns:
#'   - `duration_hours`: Trip duration in hours
#'   - `distance_nm`: Total distance traveled in nautical miles
#'   - `mean_sog`: Mean speed over ground
#' @param model_path Path to a trained gear model file (`.joblib`). If `NULL`,
#'   attempts to load a default model (if available).
#'
#' @return Data frame with original data plus prediction column:
#'   - `gear_pred`: Predicted gear type (e.g., "gillnet", "handline", "longline")
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Load trip-level data
#' trips <- read.csv("trip_features.csv")
#'
#' # Required columns: duration_hours, distance_nm, mean_sog
#' head(trips[, c("trip_id", "duration_hours", "distance_nm", "mean_sog")])
#'
#' # Predict gear type
#' predictions <- gear_predict(trips, model_path = "gear_model.joblib")
#'
#' # View results
#' table(predictions$gear_pred)
#' }
gear_predict <- function(df, model_path = NULL) {
  # Validate required columns
  required_cols <- c("duration_hours", "distance_nm", "mean_sog")
  missing <- setdiff(required_cols, names(df))

  if (length(missing) > 0) {
    stop(
      "Missing required columns: ", paste(missing, collapse = ", "), "\n",
      "Gear prediction requires: ", paste(required_cols, collapse = ", "),
      call. = FALSE
    )
  }

  # Get Python module
  ssfaitk <- .get_ssfaitk()

  # Call Python function
  result <- ssfaitk$r_api$gear_predict(
    df = df,
    model_path = model_path
  )

  return(result)
}

#' Train a custom fishing gear prediction model
#'
#' @description
#' Trains a machine learning model to predict fishing gear type from trip-level
#' features. Requires labeled training data with gear type classifications.
#' The trained model can be saved and used with `gear_predict()`.
#'
#' @param df Data frame with trip-level features and gear labels. Must contain:
#'   - `duration_hours`: Trip duration in hours
#'   - `distance_nm`: Total distance in nautical miles
#'   - `mean_sog`: Mean speed over ground
#'   - Gear label column (specified by `label_col`)
#' @param label_col Column name containing gear type labels (e.g., "gillnet",
#'   "handline", "longline"). Default: `"gear_label"`
#' @param save_path Optional path to save the trained model (`.joblib` file).
#'   Creates parent directories if needed.
#'
#' @return Trained model object (Python GearPredictor). Can be used directly
#'   or saved via the `save_path` parameter.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Load labeled training data
#' training_trips <- data.frame(
#'   trip_id = 1:100,
#'   duration_hours = runif(100, 2, 8),
#'   distance_nm = runif(100, 5, 20),
#'   mean_sog = runif(100, 1.5, 4.0),
#'   gear_label = sample(c("gillnet", "handline", "longline"), 100, replace = TRUE)
#' )
#'
#' # Train and save model
#' model <- gear_fit(
#'   training_trips,
#'   label_col = "gear_label",
#'   save_path = "my_gear_model.joblib"
#' )
#'
#' # Later, use the model for predictions
#' new_trips <- read.csv("new_trips.csv")
#' predictions <- gear_predict(new_trips, model_path = "my_gear_model.joblib")
#' }
gear_fit <- function(df,
                     label_col = "gear_label",
                     save_path = NULL) {
  # Validate required columns
  required_cols <- c("duration_hours", "distance_nm", "mean_sog")
  missing <- setdiff(required_cols, names(df))

  if (length(missing) > 0) {
    stop(
      "Missing required feature columns: ", paste(missing, collapse = ", "), "\n",
      "Gear prediction requires: ", paste(required_cols, collapse = ", "),
      call. = FALSE
    )
  }

  if (!label_col %in% names(df)) {
    stop(
      "Label column '", label_col, "' not found in data frame.\n",
      "Available columns: ", paste(names(df), collapse = ", "),
      call. = FALSE
    )
  }

  # Get Python module
  ssfaitk <- .get_ssfaitk()

  # Call Python function
  model <- ssfaitk$r_api$gear_fit(
    df = df,
    label_col = label_col,
    save_path = save_path
  )

  return(model)
}
