# R / flowDensity environment

Verified on 2026-11-09 in LSF job `50112469` (`i2c2_normal`, host `sequoia`).
The job finished with status `DONE`, no stderr output, and a passing check of
`flowDensity::deGate`.

| Component | Version | Installation |
| --- | --- | --- |
| R | 4.4.2 | `module load R/4.4` (`/appl/R-4.4/bin/Rscript`) |
| flowDensity | 1.38.0 | User R library (`.libPaths()[1]`) |
| flowCore | 2.16.0 | User R library (`.libPaths()[1]`) |
| data.table | 1.15.4 | `/misc/appl/R-4.4/lib64/R/library/data.table` |
| jsonlite | 1.8.8 | `/misc/appl/R-4.4/lib64/R/library/jsonlite` |

The R libraries resolved in this order:

1. User R library for R 4.4 (`.libPaths()[1]`)
2. `/misc/appl/R-4.4/lib64/R/library`

`flowDensity` and `flowCore` currently come from the user's R library, so the
repository does not provision them. The verification job script and its full
output are local under `jobs/`, which is excluded from Git.
