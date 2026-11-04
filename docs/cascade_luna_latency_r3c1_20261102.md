# GPT-6 luna cascade inference accounting for R3 C1

Logged test-split rerun, medium reasoning effort. Per-sample table values describe the final successful run; archived interrupted attempts are excluded from those medians and ranges, and reported separately below. API time is the sum of API attempt durations; it excludes task preparation, label propagation, evaluation, and queueing. USD is estimated using the recorded pricing table, not a billed charge. Calls are sequential within each sample.

| Cohort | Plan | Samples | Planned nodes | Median calls | Median skipped | Samples with skips | Median API s | API s range | Median est. USD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Acute2020 | L1_coarse | 12 | 22 | 22 | 0 | 0 | 154.2 | 124.9–201.6 | 0.0106 |
| Acute2020 | L2_identity | 12 | 41 | 41 | 0 | 1 | 301.3 | 270.5–340.3 | 0.0204 |
| Acute2020 | default | 12 | 62 | 62 | 0 | 1 | 545.1 | 386.2–635.6 | 0.0355 |
| Acute2021 | L1_coarse | 33 | 22 | 22 | 0 | 3 | 172.5 | 140.3–214.5 | 0.0106 |
| Acute2021 | L2_identity | 33 | 41 | 41 | 0 | 16 | 350.2 | 237.2–433.5 | 0.0209 |
| Acute2021 | default | 33 | 62 | 61 | 1 | 21 | 612.3 | 240.7–746.7 | 0.0352 |
| Bjornson | default | 32 | 8 | 8 | 0 | 2 | 80.0 | 64.7–96.0 | 0.0045 |
| FR-FCM-Z74D_hc | default | 2 | 12 | 12 | 0 | 0 | 131.9 | 124.9–138.8 | 0.0073 |
| FR-FCM-Z74D_tissue | default | 15 | 12 | 12 | 0 | 0 | 161.5 | 124.3–188.2 | 0.0086 |
| FRDR_covid19 | default | 111 | 24 | 24 | 0 | 37 | 218.5 | 133.0–306.8 | 0.0131 |
| Lyoplate_DC | default | 24 | 5 | 5 | 0 | 0 | 61.8 | 51.9–81.0 | 0.0033 |
| Lyoplate_bcell | default | 24 | 5 | 5 | 0 | 1 | 47.8 | 38.7–53.5 | 0.0028 |
| Lyoplate_tcell | default | 24 | 6 | 6 | 0 | 0 | 75.0 | 57.6–113.6 | 0.0047 |
| Lyoplate_treg | default | 24 | 5 | 5 | 0 | 0 | 47.3 | 27.8–59.0 | 0.0028 |
| Vaccine | L1_coarse | 85 | 22 | 22 | 0 | 4 | 181.3 | 124.7–256.8 | 0.0109 |
| Vaccine | L2_identity | 85 | 41 | 41 | 0 | 14 | 352.5 | 122.2–442.5 | 0.0214 |
| Vaccine | default | 85 | 62 | 62 | 0 | 27 | 576.1 | 406.3–776.1 | 0.0360 |

Full-tree table scope: 386 samples across 11 cohorts; 11688 planned nodes, 11175 completed calls, 513 empty-parent skips; 30.01 final-run API hours and $6.56 final-run estimated cost.
Sample-plan pairs: 646 across 17 cohort-plan settings.
Planned nodes: 19878; completed steps: 19201; skipped steps: 677; samples with at least one skip: 127.
Successful API calls: 19201; API attempts: 19201.
Final successful runs: 49.03 API hours and $10.68 estimated cost.
Archived interrupted attempts in 3 repaired samples: 69 attempts, 0.16 API hours, $0.04 estimated cost.
Including archived attempts: 49.19 API hours and $10.72 estimated cost.

Interpretation: depth comparisons should use the same cohort and samples. These are observed sequential API times and estimated costs, not measured clinical throughput or end-to-end wall-clock latency. Parallel sibling branches, batching, and threshold-based early stopping were not evaluated.
