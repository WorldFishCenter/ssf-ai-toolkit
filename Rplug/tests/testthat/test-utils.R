test_that("ssfaitk_available returns logical", {
  result <- ssfaitk_available()
  expect_type(result, "logical")
})

test_that("check_python_env runs without error", {
  expect_no_error(check_python_env())
})

test_that("install_ssfaitk validates parameters", {
  # Skip on CI if Python not configured
  skip_on_ci()

  # Test that function exists and has correct parameters
  expect_true(is.function(install_ssfaitk))

  # Check that function signature includes expected parameters
  fn_args <- names(formals(install_ssfaitk))
  expect_true("method" %in% fn_args)
  expect_true("package_url" %in% fn_args)
})
