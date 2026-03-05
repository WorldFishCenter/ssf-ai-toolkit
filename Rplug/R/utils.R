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

#' Get ssfaitk package versions
#'
#' @description
#' Returns the version of the R package and (if available) the Python package.
#' Useful for debugging and for pinning versions in downstream workflows.
#'
#' @return Named character vector with `r` and `python` versions.
#'   The `python` value is `NA` if the Python package is not installed.
#' @export
#'
#' @examples
#' \dontrun{
#' ssfaitk_version()
#' #>        r   python
#' #> "0.1.0" "0.1.0"
#' }
ssfaitk_version <- function() {
  r_ver <- as.character(utils::packageVersion("ssfaitk"))

  py_ver <- tryCatch({
    pkg <- reticulate::import("importlib.metadata")
    pkg$version("ssfaitk")
  }, error = function(e) NA_character_)

  versions <- c(r = r_ver, python = py_ver)
  print(versions)
  invisible(versions)
}

#' Update both the R and Python ssfaitk packages to a specific version
#'
#' @description
#' Convenience function that updates both the R package (via
#' `remotes::install_github`) and the Python package (via pip) in one call.
#' Use this in external projects to keep both components in sync.
#'
#' @param version Git tag or branch to install (default: `"main"`).
#'   Use a release tag, e.g. `"v0.2.0"`, to pin to a specific version.
#' @param force Logical; force reinstall even if already at this version
#'   (default: `FALSE`).
#'
#' @return Invisibly returns `TRUE`
#' @export
#'
#' @examples
#' \dontrun{
#' # Update to latest
#' update_ssfaitk()
#'
#' # Pin to a specific release
#' update_ssfaitk(version = "v0.2.0")
#' }
update_ssfaitk <- function(version = "main", force = FALSE) {
  pip_opts <- if (force) "--force-reinstall --no-cache-dir" else NULL

  message("--- Step 1/2: Updating R package ---")
  remotes::install_github(
    "WorldFishCenter/ssf-ai-toolkit",
    subdir = "Rplug",
    ref = version,
    force = force,
    quiet = FALSE
  )

  message("\n--- Step 2/2: Updating Python package ---")
  install_ssfaitk(version = version, pip_options = pip_opts)

  message("\nBoth packages updated. Restart your R session to use the new R package.")
  invisible(TRUE)
}

#' Install SSF AI Toolkit Python package
#'
#' @description
#' Installs the `ssfaitk` Python package from GitHub (default) or a local path.
#'
#' For production and CI use, always specify a `version` tag to pin the Python
#' package to a known release. When developing locally, pass `local = TRUE` to
#' install from the parent directory in editable mode.
#'
#' @param version Git tag or branch to install from GitHub (default: `"main"`).
#'   Use a release tag to pin to a specific version, e.g. `"v0.2.0"`.
#'   Ignored when `local = TRUE`.
#' @param local Logical; install from local source tree in editable mode.
#'   Assumes the R package lives inside the Python repo (e.g. `Rplug/`).
#'   Default: `FALSE`.
#' @param method Installation method: `"auto"` (default), `"virtualenv"`, `"conda"`
#' @param pip_options Additional pip options (character vector), e.g.
#'   `"--force-reinstall --no-cache-dir"` to force a clean re-install.
#' @param python_version Python version to use when creating new environments
#'
#' @return Invisibly returns `TRUE` if installation succeeds
#' @export
#'
#' @examples
#' \dontrun{
#' # Install latest from GitHub main branch
#' install_ssfaitk()
#'
#' # Pin to a specific release (recommended for production / CI)
#' install_ssfaitk(version = "v0.2.0")
#'
#' # Force reinstall (useful when updating)
#' install_ssfaitk(version = "v0.2.0", pip_options = "--force-reinstall --no-cache-dir")
#'
#' # Local development install (editable)
#' install_ssfaitk(local = TRUE)
#' }
install_ssfaitk <- function(version = "main",
                            local = FALSE,
                            method = "auto",
                            pip_options = NULL,
                            python_version = NULL) {
  if (local) {
    python_pkg_path <- .find_python_package_root()
    if (is.null(python_pkg_path)) {
      stop(
        "Could not locate Python package root automatically.\n",
        "Specify the path directly via pip:\n",
        "  reticulate::py_install('-e /absolute/path/to/ssf-ai-toolkit', pip = TRUE)\n",
        "Or install from GitHub:\n",
        "  install_ssfaitk()  # installs latest from main",
        call. = FALSE
      )
    }
    package_spec <- paste0("-e ", python_pkg_path)
    message("Installing ssfaitk from local source (editable): ", python_pkg_path)
  } else {
    package_spec <- paste0(
      "git+https://github.com/WorldFishCenter/ssf-ai-toolkit.git@", version
    )
    message("Installing ssfaitk ", version, " from GitHub...")
  }

  tryCatch({
    reticulate::py_install(
      packages = package_spec,
      method = method,
      pip = TRUE,
      pip_options = pip_options,
      python_version = python_version
    )
    # Invalidate cached module so next call picks up the new version
    if (exists("ssfaitk_module", envir = .ssfaitk_env)) {
      rm("ssfaitk_module", envir = .ssfaitk_env)
    }
    message("SSF AI Toolkit installed successfully. Verify with ssfaitk_available()")
    invisible(TRUE)
  }, error = function(e) {
    stop(
      "Installation failed: ", conditionMessage(e), "\n\n",
      "Troubleshooting:\n",
      "  1. Check Python: reticulate::py_config()\n",
      "  2. Set correct Python: Sys.setenv(RETICULATE_PYTHON = '/path/to/python')\n",
      "  3. Force reinstall: install_ssfaitk(pip_options = '--force-reinstall --no-cache-dir')",
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

#' Configure Python environment for use with ssfaitk
#'
#' @description
#' Points reticulate to a specific Python interpreter and optionally installs
#' `ssfaitk`. Call this **once at the top of your script or workflow**, before
#' any other `ssfaitk` function. This is especially important in GitHub Actions
#' and other CI environments where multiple Python installations may exist.
#'
#' In GitHub Actions, set the `RETICULATE_PYTHON` environment variable (via the
#' `env:` block or `Sys.setenv()`) to the Python interpreter where you installed
#' ssfaitk, and this function will pick it up automatically.
#'
#' @param python Path to the Python interpreter to use. If `NULL` (default),
#'   uses the `RETICULATE_PYTHON` environment variable. If neither is set,
#'   reticulate uses its own discovery logic.
#' @param install Logical; if `TRUE`, installs `ssfaitk` if not already present
#'   (default: `FALSE`). Pass `version` to pin the release.
#' @param version Version tag to install if `install = TRUE` (default: `"main"`).
#'   Use a release tag, e.g. `"v0.2.0"`, to pin to a specific version.
#'
#' @return Invisibly returns the configured Python path
#' @export
#'
#' @examples
#' \dontrun{
#' # In a GitHub Actions R script — Python path set via RETICULATE_PYTHON env var:
#' # (in workflow yaml)  env:
#' #                       RETICULATE_PYTHON: /usr/bin/python3
#' use_ssfaitk_python()
#'
#' # Point to a specific interpreter and auto-install if missing
#' use_ssfaitk_python(python = "/usr/bin/python3", install = TRUE, version = "v0.2.0")
#'
#' # In a virtualenv workflow
#' use_ssfaitk_python(python = "/path/to/venv/bin/python")
#' }
use_ssfaitk_python <- function(python = NULL, install = FALSE, version = "main") {
  # Resolve python path: argument > env var > reticulate default
  if (is.null(python)) {
    python <- Sys.getenv("RETICULATE_PYTHON", unset = NA)
    if (is.na(python) || nchar(python) == 0) python <- NULL
  }

  if (!is.null(python)) {
    reticulate::use_python(python, required = TRUE)
    message("Using Python: ", python)
  } else {
    message("No Python path specified; reticulate will auto-detect.")
    message("Set RETICULATE_PYTHON or pass python = '/path/to/python' to avoid surprises.")
  }

  if (install && !ssfaitk_available()) {
    message("ssfaitk not found — installing version '", version, "'...")
    install_ssfaitk(version = version)
  }

  invisible(python)
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
