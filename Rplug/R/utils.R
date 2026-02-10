#' Check if SSF AI Toolkit Python package is available
#'
#' @description
#' Checks whether the `ssfaitk` Python package is installed and importable
#' in the current Python environment.
#'
#' @return Logical value: `TRUE` if available, `FALSE` otherwise
#' @export
#'
#' @examples
#' \dontrun{
#' if (ssfaitk_available()) {
#'   message("SSF AI Toolkit is ready to use!")
#' } else {
#'   message("Please install with install_ssfaitk()")
#' }
#' }
ssfaitk_available <- function() {
  tryCatch({
    reticulate::py_module_available("ssfaitk")
  }, error = function(e) {
    FALSE
  })
}

#' Install SSF AI Toolkit Python package
#'
#' @description
#' Installs the `ssfaitk` Python package in the current Python environment.
#' By default, installs the package in editable mode from the parent directory
#' (assumes the R package is in `Rplug/` subdirectory). For remote installation,
#' use `method = "pip"` with a GitHub URL or PyPI (when available).
#'
#' @param method Installation method: "auto" (default), "virtualenv", "conda", or "pip"
#' @param conda Path to conda executable (for method = "conda")
#' @param pip Logical; whether to use pip for installation (default: TRUE)
#' @param pip_options Additional options to pass to pip
#' @param python_version Python version to use (for new environments)
#' @param package_url URL or path to install from. Default is `".."` (parent directory)
#'   for local editable install. Use a GitHub URL for remote installation, e.g.,
#'   "git+https://github.com/user/ssf-ai-toolkit.git"
#'
#' @return Invisibly returns `TRUE` if installation succeeds
#' @export
#'
#' @examples
#' \dontrun{
#' # Install from local source (if in development)
#' install_ssfaitk()
#'
#' # Install from GitHub
#' install_ssfaitk(
#'   package_url = "git+https://github.com/user/ssf-ai-toolkit.git"
#' )
#'
#' # Install with pip in a virtualenv
#' install_ssfaitk(method = "virtualenv")
#' }
install_ssfaitk <- function(method = "auto",
                            conda = "auto",
                            pip = TRUE,
                            pip_options = NULL,
                            python_version = NULL,
                            package_url = "..") {
  # Determine if we're doing a local editable install or remote install
  if (package_url == "..") {
    # Local editable install - find Python package root
    python_pkg_path <- .find_python_package_root()

    if (is.null(python_pkg_path)) {
      stop(
        "Could not locate Python package root automatically.\n",
        "Please specify the path explicitly:\n",
        "  install_ssfaitk(package_url = '/absolute/path/to/ssf-ai-toolkit')\n",
        "Or install from GitHub:\n",
        "  install_ssfaitk(package_url = 'git+https://github.com/WorldFishCenter/ssf-ai-toolkit.git')",
        call. = FALSE
      )
    }

    package_spec <- paste0("-e ", python_pkg_path)
    message("Installing ssfaitk from local source in editable mode...")
    message("Using path: ", python_pkg_path)
  } else {
    # Remote install or explicit path
    package_spec <- package_url
    message("Installing ssfaitk from: ", package_url)
  }

  # Install the package
  tryCatch({
    reticulate::py_install(
      packages = package_spec,
      method = method,
      conda = conda,
      pip = pip,
      pip_options = pip_options,
      python_version = python_version
    )
    message("\nSSF AI Toolkit installed successfully!")
    message("Verify with: ssfaitk_available()")
    invisible(TRUE)
  }, error = function(e) {
    stop(
      "Installation failed. Error: ", conditionMessage(e), "\n\n",
      "Troubleshooting:\n",
      "  1. Check Python is available: reticulate::py_config()\n",
      "  2. Try manual installation: pip install -e /path/to/ssf-ai-toolkit\n",
      "  3. For remote install: pip install git+https://github.com/WorldFishCenter/ssf-ai-toolkit.git",
      call. = FALSE
    )
  })
}

# Internal helper to find Python package root
.find_python_package_root <- function() {
  # Method 1: If R package is installed, get its location
  r_pkg_path <- system.file(".", package = "ssfaitk")
  if (r_pkg_path != "") {
    # Package is installed, go up one level
    potential_path <- normalizePath(file.path(r_pkg_path, ".."), mustWork = FALSE)
    if (.is_python_package_root(potential_path)) {
      return(potential_path)
    }
  }

  # Method 2: Check if we're in development mode (Rplug/ directory)
  cwd <- getwd()
  if (basename(cwd) == "Rplug") {
    potential_path <- normalizePath(file.path(cwd, ".."), mustWork = FALSE)
    if (.is_python_package_root(potential_path)) {
      return(potential_path)
    }
  }

  # Method 3: Check if we're in project root
  if (.is_python_package_root(cwd)) {
    return(cwd)
  }

  # Method 4: Walk up the directory tree looking for markers
  current <- cwd
  for (i in 1:10) {  # Max 10 levels up
    if (.is_python_package_root(current)) {
      return(current)
    }
    parent <- dirname(current)
    if (parent == current) break  # Reached filesystem root
    current <- parent
  }

  return(NULL)
}

# Internal helper to check if a path is the Python package root
.is_python_package_root <- function(path) {
  if (!dir.exists(path)) return(FALSE)

  files <- list.files(path)

  # Look for Python package markers: setup.py, pyproject.toml, or src/ssfaitk/
  has_setup <- "setup.py" %in% files || "pyproject.toml" %in% files
  has_src <- dir.exists(file.path(path, "src", "ssfaitk"))

  return(has_setup || has_src)
}

#' Check Python environment configuration
#'
#' @description
#' Displays information about the current Python environment being used
#' by reticulate. Helpful for debugging environment issues.
#'
#' @return Invisibly returns the result of `reticulate::py_config()`
#' @export
#'
#' @examples
#' \dontrun{
#' check_python_env()
#' }
check_python_env <- function() {
  cat("Python Environment Configuration:\n")
  cat("================================\n\n")

  config <- reticulate::py_config()
  print(config)

  cat("\n\nPython packages installed:\n")
  cat("=========================\n")

  # Try to list installed packages
  tryCatch({
    packages <- reticulate::py_list_packages()
    if ("ssfaitk" %in% packages$package) {
      ssfaitk_row <- packages[packages$package == "ssfaitk", ]
      cat("[OK] ssfaitk version:", ssfaitk_row$version, "\n")
      cat("     Location:", ssfaitk_row$location, "\n")
    } else {
      cat("[!] ssfaitk is NOT installed\n")
      cat("    Install with: install_ssfaitk()\n")
    }
  }, error = function(e) {
    cat("Could not list packages. Error:", conditionMessage(e), "\n")
  })

  cat("\n")
  invisible(config)
}
