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
  # Return cached module if already loaded
  if (exists("ssfaitk_module", envir = .ssfaitk_env)) {
    return(get("ssfaitk_module", envir = .ssfaitk_env))
  }

  # Check availability first so we can give a helpful error
  if (!reticulate::py_module_available("ssfaitk")) {
    stop(
      "Python module 'ssfaitk' is not installed.\n\n",
      "Install it with:\n",
      "  install_ssfaitk()                          # latest from GitHub\n",
      "  install_ssfaitk(version = 'v0.2.0')       # specific version\n\n",
      "Or set RETICULATE_PYTHON to a Python where ssfaitk is already installed.",
      call. = FALSE
    )
  }

  ssfaitk <- reticulate::import("ssfaitk")
  assign("ssfaitk_module", ssfaitk, envir = .ssfaitk_env)
  ssfaitk
}
