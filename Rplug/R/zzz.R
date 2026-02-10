# Package environment for caching Python module
.ssfaitk_env <- new.env(parent = emptyenv())

#' @importFrom reticulate import py_module_available
.onAttach <- function(libname, pkgname) {
  # Display startup message
  packageStartupMessage(
    "SSF AI Toolkit R Package\n",
    "Python package 'ssfaitk' required.\n",
    "Use install_ssfaitk() if not already installed.\n",
    "Check availability with ssfaitk_available()"
  )
}

#' Get SSF AI Toolkit Python module
#'
#' @description
#' Internal function to lazy-load the ssfaitk Python module.
#' Caches the module handle for subsequent calls.
#'
#' @return Python module handle
#' @keywords internal
#' @noRd
.get_ssfaitk <- function() {
  # Check if module is already cached
  if (exists("ssfaitk_module", envir = .ssfaitk_env)) {
    return(get("ssfaitk_module", envir = .ssfaitk_env))
  }

  # Try to import the module
  tryCatch({
    ssfaitk <- reticulate::import("ssfaitk", delay_load = TRUE)
    # Cache the module
    assign("ssfaitk_module", ssfaitk, envir = .ssfaitk_env)
    return(ssfaitk)
  }, error = function(e) {
    stop(
      "Failed to load Python module 'ssfaitk'.\n\n",
      "Please ensure:\n",
      "  1. Python is available (check with reticulate::py_config())\n",
      "  2. The ssfaitk package is installed (use install_ssfaitk())\n\n",
      "Error message: ", conditionMessage(e),
      call. = FALSE
    )
  })
}
