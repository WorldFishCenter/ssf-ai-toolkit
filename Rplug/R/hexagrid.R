#' Aggregate GPS fishing tracks into H3 hexagonal grid cells
#'
#' @description
#' Aggregates GPS fishing predictions into H3 hexagonal grid cells with
#' time-weighted fishing effort metrics. Computes multiple temporal dimensions
#' (overall, year, season, month, day/night) suitable for dashboard generation
#' and spatial visualization.
#'
#' Requires that the input data frame already contains fishing effort predictions
#' (e.g., from `effort_predict_statistical()`).
#'
#' @param df Data frame with GPS fishing track data including effort predictions.
#'   Must contain columns for latitude, longitude, timestamp, and trip ID.
#'   Should also include a fishing classification column (0/1).
#' @param resolutions Integer vector of H3 resolutions to compute. Default is
#'   `c(7L, 8L)` (regional and detail scales).
#' @param lat_col Column name for latitude (auto-detected if `NULL`)
#' @param lon_col Column name for longitude (auto-detected if `NULL`)
#' @param time_col Column name for timestamp (auto-detected if `NULL`)
#' @param trip_col Column name for trip ID (auto-detected if `NULL`)
#' @param fishing_col Column name for fishing classification 0/1
#'   (auto-detected if `NULL`, looks for `is_fishing`, `effort_pred`, etc.)
#' @param output_dir Directory path to save output files. If `NULL`, results
#'   are only returned in memory.
#' @param save_parquet Logical; save results as parquet files (default: `TRUE`).
#'   Recommended for large datasets and dashboard use.
#' @param save_csv Logical; also save results as CSV files (default: `FALSE`)
#' @param min_hours Minimum fishing hours per hexagon to include (default: `0.1`)
#' @param min_trips Minimum number of trips per hexagon to include (default: `1`)
#' @param min_days Minimum number of days per hexagon to include (default: `1`)
#'
#' @return Named list with aggregation results per resolution (e.g., `res7`, `res8`).
#'   Each element contains temporal breakdowns: `overall`, `year`, `season`,
#'   `month`, `daynight`.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Step 1: Get fishing effort predictions
#' tracks <- read.csv("gps_tracks.csv")
#' predictions <- effort_predict_statistical(tracks)
#'
#' # Step 2: Aggregate into hexagons
#' hex_results <- hexagrid_aggregate(predictions)
#'
#' # Access overall summary for resolution 7
#' hex_results$res7$overall
#'
#' # Save results to disk for dashboard use
#' hex_results <- hexagrid_aggregate(
#'   predictions,
#'   resolutions = c(7L, 8L),
#'   output_dir = "hex_output",
#'   save_parquet = TRUE
#' )
#' }
hexagrid_aggregate <- function(df,
                               resolutions = c(7L, 8L),
                               lat_col = NULL,
                               lon_col = NULL,
                               time_col = NULL,
                               trip_col = NULL,
                               fishing_col = NULL,
                               output_dir = NULL,
                               save_parquet = TRUE,
                               save_csv = FALSE,
                               min_hours = 0.1,
                               min_trips = 1L,
                               min_days = 1L) {
  # Get Python module
  ssfaitk <- .get_ssfaitk()

  # Convert R integer vector to Python list
  resolutions_py <- as.list(as.integer(resolutions))

  # Call Python function
  result <- ssfaitk$r_api$hexagrid_aggregate(
    df = df,
    resolutions = resolutions_py,
    lat_col = lat_col,
    lon_col = lon_col,
    time_col = time_col,
    trip_col = trip_col,
    fishing_col = fishing_col,
    output_dir = output_dir,
    save_parquet = save_parquet,
    save_csv = save_csv,
    min_hours = min_hours,
    min_trips = as.integer(min_trips),
    min_days = as.integer(min_days)
  )

  return(result)
}
