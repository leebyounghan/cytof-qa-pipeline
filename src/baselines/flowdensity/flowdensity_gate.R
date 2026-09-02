#!/usr/bin/env Rscript
# flowdensity_gate.R — Stage 2 of the flowDensity pipeline.
#
# Scans results_dir for step directories that have input.csv but no
# thresholds.json, runs deGate on x and y channels, and writes thresholds.json.
#
# Input  (per step): results/{dataset}/{sample}/step_NN/input.csv
# Output (per step): results/{dataset}/{sample}/step_NN/thresholds.json
#
# thresholds.json format:
#   {
#     "x_thresholds": [t1, t2, ...],
#     "y_thresholds": [t1, t2, ...]
#   }
#
# Usage:
#   Rscript flowdensity_gate.R --results-dir results/flowdensity/
#   Rscript flowdensity_gate.R --results-dir results/flowdensity/ --overwrite
#   Rscript flowdensity_gate.R --results-dir results/flowdensity/ --cores 8

suppressPackageStartupMessages({
  library(flowCore)
  library(flowDensity)
  library(jsonlite)
  library(data.table)
  library(parallel)
})

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

results_dir <- NULL
overwrite   <- FALSE
n_cores     <- max(1L, detectCores() - 1L)

i <- 1
while (i <= length(args)) {
  if (args[i] == "--results-dir") {
    results_dir <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--overwrite") {
    overwrite <- TRUE; i <- i + 1
  } else if (args[i] == "--cores") {
    n_cores <- as.integer(args[i + 1]); i <- i + 2
  } else {
    i <- i + 1
  }
}

if (is.null(results_dir)) {
  stop("Usage: Rscript flowdensity_gate.R --results-dir <path> [--overwrite] [--cores N]")
}

# ---------------------------------------------------------------------------
# Core: run deGate on a single channel, return numeric threshold(s)
# ---------------------------------------------------------------------------

gate_channel <- function(ff, ch) {
  tryCatch({
    cuts <- deGate(ff, channel = ch, all.cuts = TRUE, verbose = FALSE)
    cuts <- as.numeric(cuts)
    cuts <- cuts[!is.na(cuts) & is.finite(cuts)]
    if (length(cuts) == 0) stop("no valid cuts")
    cuts
  }, error = function(e) {
    as.numeric(quantile(exprs(ff)[, ch], 0.5))
  })
}

# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

process_one <- function(csv_path, overwrite) {
  step_dir <- dirname(csv_path)
  out_path <- file.path(step_dir, "thresholds.json")

  if (!overwrite && file.exists(out_path)) {
    return("skipped")
  }

  dat <- tryCatch(fread(csv_path), error = function(e) NULL)
  if (is.null(dat) || nrow(dat) < 1) {
    return("too_few_rows")
  }

  mat <- matrix(c(dat$x, dat$y), ncol = 2, dimnames = list(NULL, c("x", "y")))
  ff  <- flowFrame(mat)

  result <- list(
    x_thresholds = gate_channel(ff, "x"),
    y_thresholds = gate_channel(ff, "y")
  )

  writeLines(toJSON(result, auto_unbox = FALSE), out_path)
  "done"
}

# ---------------------------------------------------------------------------
# Discover and dispatch
# ---------------------------------------------------------------------------

input_files <- list.files(
  results_dir,
  pattern   = "^input\\.csv$",
  recursive = TRUE,
  full.names = TRUE
)

cat(sprintf("Found %d input.csv files under %s\n", length(input_files), results_dir))
cat(sprintf("Using %d core(s)\n", n_cores))

if (n_cores > 1L && .Platform$OS.type != "windows") {
  statuses <- mclapply(
    input_files,
    function(p) tryCatch(process_one(p, overwrite), error = function(e) paste0("error:", conditionMessage(e))),
    mc.cores = n_cores,
    mc.preschedule = FALSE
  )
} else {
  statuses <- lapply(
    input_files,
    function(p) tryCatch(process_one(p, overwrite), error = function(e) paste0("error:", conditionMessage(e)))
  )
}

statuses <- unlist(statuses)
n_done <- sum(statuses == "done")
n_skip <- sum(statuses == "skipped") + sum(statuses == "too_few_rows")
n_err  <- sum(startsWith(statuses, "error:"))

if (n_err > 0) {
  err_paths <- input_files[startsWith(statuses, "error:")]
  for (k in seq_along(err_paths)) {
    cat(sprintf("  [ERR] %s: %s\n", err_paths[k], statuses[startsWith(statuses, "error:")][k]))
  }
}

cat(sprintf("\nDone. Written: %d  Skipped: %d  Errors: %d\n", n_done, n_skip, n_err))
