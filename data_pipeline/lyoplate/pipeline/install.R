# Install R dependencies for extract_from_gs.R
# Tested with R 4.4.2 / Bioconductor 3.20.
#   flowWorkspace (Bioc, tested 4.18.1) — GatingSet load + legacy conversion
#   flowCore      (Bioc, tested 2.18.0) — flowFrame expression access
#   data.table    (CRAN, tested 1.16.2) — fast CSV I/O
# Requires R >= 4.4. Bioconductor packages are tied to the R version, so use a
# matching R; older R needs the matching older Bioc release.
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("flowWorkspace", "flowCore"), update = FALSE, ask = FALSE)
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
