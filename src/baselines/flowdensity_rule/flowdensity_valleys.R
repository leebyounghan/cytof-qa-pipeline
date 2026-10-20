#!/usr/bin/env Rscript
# flowdensity_valleys.R — Stage 2 of the flowDensity + notes.yaml rule pipeline.
#
# For each step dir with input.csv + meta.json, finds density valleys with
# flowDensity::deGate on x and y and writes valleys.json:
#   {"x": [v...], "y": [v...], "x_error": "...", "y_error": "..."}
#
# Valley selection per axis, from meta.json rule.n_{x,y}_valleys:
#   0  → no valleys
#   1  → deGate default single cut
#   ≥2 → deGate(all.cuts = TRUE); count is checked in the post stage
# A deGate failure is recorded as {axis}_error (no substitute threshold).
#
# Usage:
#   Rscript flowdensity_valleys.R --results-dir results/flowdensity_rule/Bjornson --cores 4

suppressPackageStartupMessages({
  library(flowCore)
  library(flowDensity)
  library(jsonlite)
  library(data.table)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
results_dir <- NULL
n_cores <- 1L
i <- 1
while (i <= length(args)) {
  if (args[i] == "--results-dir") {
    results_dir <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--cores") {
    n_cores <- as.integer(args[i + 1]); i <- i + 2
  } else {
    stop(sprintf("unknown argument: %s", args[i]))
  }
}
if (is.null(results_dir)) stop("Usage: Rscript flowdensity_valleys.R --results-dir <path> [--cores N]")

axis_valleys <- function(ff, ch, k) {
  if (k == 0) return(list(v = numeric(0), err = NULL))
  tryCatch({
    cuts <- as.numeric(deGate(ff, channel = ch, all.cuts = (k >= 2), verbose = FALSE))
    cuts <- cuts[!is.na(cuts) & is.finite(cuts)]
    if (length(cuts) == 0) stop("deGate returned no finite cut")
    list(v = sort(cuts), err = NULL)
  }, error = function(e) list(v = numeric(0), err = conditionMessage(e)))
}

process_one <- function(csv_path) {
  step_dir <- dirname(csv_path)
  meta <- fromJSON(file.path(step_dir, "meta.json"))
  dat <- fread(csv_path)
  ff <- flowFrame(matrix(c(dat$x, dat$y), ncol = 2, dimnames = list(NULL, c("x", "y"))))
  xr <- axis_valleys(ff, "x", meta$rule$n_x_valleys)
  yr <- axis_valleys(ff, "y", meta$rule$n_y_valleys)
  out <- list(x = xr$v, y = yr$v)
  if (!is.null(xr$err)) out$x_error <- xr$err
  if (!is.null(yr$err)) out$y_error <- yr$err
  writeLines(toJSON(out, auto_unbox = FALSE, digits = NA), file.path(step_dir, "valleys.json"))
  "done"
}

input_files <- list.files(results_dir, pattern = "^input\\.csv$", recursive = TRUE, full.names = TRUE)
cat(sprintf("Found %d input.csv under %s; cores=%d\n", length(input_files), results_dir, n_cores))
statuses <- unlist(mclapply(input_files, function(p)
  tryCatch(process_one(p), error = function(e) paste0("error:", conditionMessage(e))),
  mc.cores = n_cores, mc.preschedule = FALSE))
bad <- startsWith(statuses, "error:")
if (any(bad)) {
  for (k in which(bad)) cat(sprintf("  [ERR] %s: %s\n", input_files[k], statuses[k]))
  stop(sprintf("%d step(s) failed outside deGate", sum(bad)))
}
cat(sprintf("Done. valleys.json written: %d\n", sum(statuses == "done")))
