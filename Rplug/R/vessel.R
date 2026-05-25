#' Predict vessel type from trip-level features
#'
#' @description
#' Predicts vessel type (motorized vs. non-motorized) from aggregated trip-level
#' features. Requires a trained model file (`.joblib`) from prior training via
#' `vessel_fit()`.
#'
#' @param df Data frame with trip-level features. Must contain columns:
#'   - `duration_hours`: Trip duration in hours
#'   - `distance_nm`: Total distance traveled in nautical miles
#'   - `mean_sog`: Mean speed over ground
#' @param model_path Path to a trained vessel model file (`.joblib`). If `NULL`,
#'   attempts to load a default model (if available).
#'
#' @return Data frame with original data plus prediction column:
#'   - `vessel_type_pred`: Predicted vessel type (e.g., "motorized", "non-motorized")
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
#' # Predict vessel type
#' predictions <- vessel_predict(trips, model_path = "vessel_model.joblib")
#'
#' # View results
#' table(predictions$vessel_type_pred)
#' }
vessel_predict <- function(df, model_path = NULL) {
  # Validate required columns
  required_cols <- c("duration_hours", "distance_nm", "mean_sog")
  missing <- setdiff(required_cols, names(df))

  if (length(missing) > 0) {
    stop(
      "Missing required columns: ", paste(missing, collapse = ", "), "\n",
      "Vessel type prediction requires: ", paste(required_cols, collapse = ", "),
      call. = FALSE
    )
  }

  # Get Python module
  ssfaitk <- .get_ssfaitk()

  # Call Python function
  result <- ssfaitk$r_api$vessel_predict(
    df = df,
    model_path = model_path
  )

  return(result)
}

#' Train a custom vessel type prediction model
#'
#' @description
#' Trains a machine learning model to predict vessel type (motorized vs.
#' non-motorized) from trip-level features. Requires labeled training data with
#' vessel type classifications. The trained model can be saved and used with
#' `vessel_predict()`.
#'
#' @param df Data frame with trip-level features and vessel type labels. Must contain:
#'   - `duration_hours`: Trip duration in hours
#'   - `distance_nm`: Total distance in nautical miles
#'   - `mean_sog`: Mean speed over ground
#'   - Vessel type label column (specified by `label_col`)
#' @param label_col Column name containing vessel type labels (e.g., "motorized",
#'   "non-motorized"). Default: `"vessel_type_label"`
#' @param save_path Optional path to save the trained model (`.joblib` file).
#'   Creates parent directories if needed.
#'
#' @return Trained model object (Python VesselTypePredictor). Can be used directly
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
#'   vessel_type_label = sample(
#'     c("motorized", "non-motorized"),
#'     100,
#'     replace = TRUE,
#'     prob = c(0.7, 0.3)
#'   )
#' )
#'
#' # Train and save model
#' model <- vessel_fit(
#'   training_trips,
#'   label_col = "vessel_type_label",
#'   save_path = "my_vessel_model.joblib"
#' )
#'
#' # Later, use the model for predictions
#' new_trips <- read.csv("new_trips.csv")
#' predictions <- vessel_predict(new_trips, model_path = "my_vessel_model.joblib")
#' }
vessel_fit <- function(df,
                       label_col = "vessel_type_label",
                       save_path = NULL) {
  # Validate required columns
  required_cols <- c("duration_hours", "distance_nm", "mean_sog")
  missing <- setdiff(required_cols, names(df))

  if (length(missing) > 0) {
    stop(
      "Missing required feature columns: ", paste(missing, collapse = ", "), "\n",
      "Vessel type prediction requires: ", paste(required_cols, collapse = ", "),
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
  model <- ssfaitk$r_api$vessel_fit(
    df = df,
    label_col = label_col,
    save_path = save_path
  )

  return(model)
}
