# extract_from_gs.R
# Extract compensated expression + boolean gate columns from a legacy GatingSetList.
# Run from the Lyoplate data directory (sibling of gslist-<panel>/):
#   Rscript pipeline/extract_from_gs.R --panel bcell [--max_samples 3]

suppressMessages(suppressWarnings({
  library(flowWorkspace)
  library(flowCore)
  library(data.table)
}))

args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  args[idx + 1]
}

panel       <- parse_arg("--panel")
max_samples <- as.integer(parse_arg("--max_samples", "0"))

if (is.null(panel)) stop("--panel required (bcell, tcell, DC, treg)")

# Resolve paths (relative to the Lyoplate data directory)
gslist_dir <- file.path(".", paste0("gslist-", panel))
out_dir    <- file.path("pipeline", panel, "csv_tmp")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("Panel:", panel, "\n")
cat("GatingSet dir:", gslist_dir, "\n")

# Get all GatingSet subdirectories
gs_dirs <- list.dirs(gslist_dir, recursive = FALSE)
gs_dirs <- gs_dirs[!grepl("DS_Store", gs_dirs)]

manifest <- data.table(
  csv_file    = character(),
  sample_name = character(),
  site        = character(),
  n_events    = integer(),
  n_gates     = integer()
)

total_samples <- 0

for (gsd in gs_dirs) {
  site <- basename(gsd)
  tmp_dir <- file.path("/tmp", paste0("gs_lyoplate_", panel, "_", site))

  # Convert legacy format if needed
  if (!dir.exists(tmp_dir)) {
    tryCatch(
      suppressMessages(convert_legacy_gs(gsd, tmp_dir)),
      error = function(e) { cat("  SKIP site", site, ":", conditionMessage(e), "\n") }
    )
  }

  gs <- tryCatch(load_gs(tmp_dir), error = function(e) NULL)
  if (is.null(gs) || length(gs) == 0) {
    cat("  SKIP site", site, "- failed to load\n")
    next
  }

  cat("  Site:", site, "| Samples:", length(gs), "\n")

  for (i in seq_along(gs)) {
    if (max_samples > 0 && total_samples >= max_samples) break
    gh    <- gs[[i]]
    sname <- sampleNames(gs)[i]

    gate_paths     <- gh_get_pop_paths(gh)
    gate_paths     <- gate_paths[gate_paths != "root"]

    # Find the Live population gate
    # Paths vary across sites: "singlets/Live" or just "Live"
    live_path <- NULL
    for (gp in gate_paths) {
      if (grepl("/Live$", gp)) {
        live_path <- gp
        break
      }
    }

    if (is.null(live_path)) {
      cat("    SKIP", sname, "- no Live gate found\n")
      next
    }

    # Extract expression from Live population only (pre-filtered)
    fr       <- gh_pop_get_data(gh, live_path)
    expr_mat <- exprs(fr)
    dt       <- as.data.table(expr_mat)
    live_idx <- which(gh_pop_get_indices(gh, live_path))

    # Only keep gates downstream of Live
    downstream_paths <- gate_paths[startsWith(gate_paths, paste0(live_path, "/"))]
    gate_col_names   <- gsub("/", "__", sub("^/", "", downstream_paths))

    for (j in seq_along(downstream_paths)) {
      # Get full boolean, then subset to live cells
      # Some boolean gates reference nodes that fail after legacy conversion — skip them
      full_idx <- tryCatch(
        gh_pop_get_indices(gh, downstream_paths[j]),
        error = function(e) {
          cat("    SKIP gate", downstream_paths[j], ":", conditionMessage(e), "\n")
          NULL
        }
      )
      if (!is.null(full_idx)) {
        dt[, (gate_col_names[j]) := full_idx[live_idx]]
      }
    }

    dt[, sample_name := sname]
    dt[, site := site]

    csv_name <- paste0(gsub("[.]fcs$", "", sname), ".csv.gz")
    fwrite(dt, file.path(out_dir, csv_name))

    if (!file.exists(file.path(out_dir, "marker_map.csv"))) {
      params <- pData(parameters(fr))
      mm     <- data.table(channel = params$name, marker = params$desc)
      fwrite(mm, file.path(out_dir, "marker_map.csv"))
    }

    manifest <- rbind(manifest, data.table(
      csv_file    = csv_name,
      sample_name = sname,
      site        = site,
      n_events    = nrow(dt),
      n_gates     = length(downstream_paths)
    ))
    total_samples <- total_samples + 1
    cat("    ", sname, ":", nrow(dt), "events,", length(gate_paths), "gates\n")
  }
  if (max_samples > 0 && total_samples >= max_samples) break
}

fwrite(manifest, file.path(out_dir, "manifest.csv"))
cat("\nExtracted", nrow(manifest), "samples to", out_dir, "\n")
