test_that("effort functions exist", {
  expect_true(is.function(effort_predict))
  expect_true(is.function(effort_predict_statistical))
  expect_true(is.function(effort_fit))
})

test_that("effort_predict requires model_path or default model", {
  skip_if_not(ssfaitk_available(), "Python package not available")

  # Create minimal test data
  test_data <- data.frame(
    trip_id = rep(1, 5),
    timestamp = seq(as.POSIXct("2024-01-01 10:00"), by = "5 min", length.out = 5),
    latitude = seq(-6.0, -6.01, length.out = 5),
    longitude = seq(39.0, 39.01, length.out = 5)
  )

  # Should error if no model and no default
  expect_error(
    effort_predict(test_data),
    regexp = "No trained model found|Model file not found"
  )
})

test_that("effort_predict_statistical works with minimal data", {
  skip_if_not(ssfaitk_available(), "Python package not available")

  # Create minimal test data
  test_data <- data.frame(
    trip_id = rep(1, 10),
    timestamp = seq(as.POSIXct("2024-01-01 10:00"), by = "5 min", length.out = 10),
    latitude = seq(-6.0, -6.05, length.out = 10),
    longitude = seq(39.0, 39.05, length.out = 10)
  )

  # Should work without error
  result <- effort_predict_statistical(test_data)

  # Check output structure
  expect_s3_class(result, "data.frame")
  expect_true("is_fishing" %in% colnames(result))
  expect_true("fishing_score" %in% colnames(result))
  expect_equal(nrow(result), nrow(test_data))
})

test_that("effort functions validate column names with colmap", {
  skip_if_not(ssfaitk_available(), "Python package not available")

  # Create test data with custom column names
  test_data <- data.frame(
    VesselID = rep(1, 5),
    GPS_Time = seq(as.POSIXct("2024-01-01 10:00"), by = "5 min", length.out = 5),
    Lat = seq(-6.0, -6.01, length.out = 5),
    Lon = seq(39.0, 39.01, length.out = 5)
  )

  # Should work with colmap
  expect_no_error(
    effort_predict_statistical(
      test_data,
      colmap = list(
        trip_id = "VesselID",
        time = "GPS_Time",
        lat = "Lat",
        lon = "Lon"
      )
    )
  )
})
