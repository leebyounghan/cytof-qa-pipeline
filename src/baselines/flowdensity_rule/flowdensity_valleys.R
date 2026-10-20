#!/usr/bin/env Rscript
# flowdensity_valleys.R — Stage 2 of the flowDensity + notes.yaml rule pipeline.
#
# For each step dir with input.csv + meta.json, finds density valleys with
# flowDensity::deGate on x and y and writes valleys.json:
#   {"x": [v...], "y": [v...], "x_error": "...", "y_error": "..."}
#
# Valley selection per axis, from meta.json rule.n_{x,y}_valleys (k):
#   k = 0 → no valleys
#   k = 1 → deGate default single cut
#   k ≥ 2 → deGate(all.cuts = TRUE), then
#     more than k cuts → keep the k deepest            ({axis}_method "top_depth")
#     exactly k        → use as-is                     ("all_cuts")
#     1 cut, k = 2     → re-run deGate on each side of the cut; a side's cut
#                        counts if it lies inside that side and has positive
#                        depth; keep the deeper one     ("split_rescue")
#                        none found → keep the single cut ("middle_dropped";
#                        post maps cells to the outer labels only)
#     otherwise        → returned as-is; post fails the step on count
# Depth of a cut = min(highest density between it and each neighbouring cut
# or the data edge) − density at the cut, on density(x, adjust = 1) (the
# deGate default smoothing).
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

finite_cuts <- function(cuts) {
  cuts <- as.numeric(cuts)
  sort(cuts[!is.na(cuts) & is.finite(cuts)])
}

# Depth of each cut in `cuts` (sorted) on density d, with the other cuts as
# neighbours.
cut_depths <- function(cuts, d) {
  at <- approx(d$x, d$y, xout = cuts)$y
  edges <- c(-Inf, cuts, Inf)
  peak <- function(lo, hi) {
    y <- d$y[d$x > lo & d$x < hi]
    if (length(y) == 0) 0 else max(y)
  }
  vapply(seq_along(cuts), function(i)
    min(peak(edges[i], edges[i + 1]), peak(edges[i + 1], edges[i + 2])) - at[i],
    numeric(1))
}

side_cut <- function(vals, lo, hi) {
  sub <- vals[vals > lo & vals < hi]
  cut <- tryCatch(finite_cuts(deGate(flowFrame(matrix(sub, ncol = 1, dimnames = list(NULL, "v"))),
                                     channel = "v", verbose = FALSE)),
                  error = function(e) numeric(0))
  cut[cut > lo & cut < hi]
}

axis_valleys <- function(ff, ch, k) {
  if (k == 0) return(list(v = numeric(0), err = NULL, method = "none"))
  tryCatch({
    cuts <- finite_cuts(deGate(ff, channel = ch, all.cuts = (k >= 2), verbose = FALSE))
    if (length(cuts) == 0) stop("deGate returned no finite cut")
    if (k == 1 || length(cuts) == k) {
      return(list(v = cuts, err = NULL, method = if (k == 1) "default" else "all_cuts",
                  n_all_cuts = length(cuts)))
    }
    vals <- exprs(ff)[, ch]
    d <- density(vals, adjust = 1)
    if (length(cuts) > k) {
      keep <- order(cut_depths(cuts, d), decreasing = TRUE)[seq_len(k)]
      return(list(v = sort(cuts[keep]), err = NULL, method = "top_depth",
                  n_all_cuts = length(cuts)))
    }
    if (length(cuts) == 1 && k == 2) {
      c0 <- cuts[1]
      cand <- c(side_cut(vals, -Inf, c0), side_cut(vals, c0, Inf))
      depth <- vapply(cand, function(cc) {
        both <- sort(c(cc, c0))
        cut_depths(both, d)[match(cc, both)]
      }, numeric(1))
      cand <- cand[depth > 0]; depth <- depth[depth > 0]
      if (length(cand) > 0) {
        return(list(v = sort(c(c0, cand[which.max(depth)])), err = NULL,
                    method = "split_rescue", n_all_cuts = 1L))
      }
      return(list(v = cuts, err = NULL, method = "middle_dropped", n_all_cuts = 1L))
    }
    list(v = cuts, err = NULL, method = "all_cuts", n_all_cuts = length(cuts))
  }, error = function(e) list(v = numeric(0), err = conditionMessage(e), method = "error"))
}

process_one <- function(csv_path) {
  step_dir <- dirname(csv_path)
  meta <- fromJSON(file.path(step_dir, "meta.json"))
  dat <- fread(csv_path)
  ff <- flowFrame(matrix(c(dat$x, dat$y), ncol = 2, dimnames = list(NULL, c("x", "y"))))
  xr <- axis_valleys(ff, "x", meta$rule$n_x_valleys)
  yr <- axis_valleys(ff, "y", meta$rule$n_y_valleys)
  out <- list(x = xr$v, y = yr$v, x_method = unbox(xr$method), y_method = unbox(yr$method))
  if (!is.null(xr$n_all_cuts)) out$x_n_all_cuts <- unbox(xr$n_all_cuts)
  if (!is.null(yr$n_all_cuts)) out$y_n_all_cuts <- unbox(yr$n_all_cuts)
  if (!is.null(xr$err)) out$x_error <- unbox(xr$err)
  if (!is.null(yr$err)) out$y_error <- unbox(yr$err)
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
