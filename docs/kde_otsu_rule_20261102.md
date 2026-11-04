# KDE landmarks with histogram Otsu fallback

Exploratory analysis only. Otsu introduces a histogram thresholding algorithm beyond the supplied peak/valley landmarks, so this result is not used as the direct control in the reviewer response. The response reports the direct KDE-landmark rule and the separately computed flowDensity rule.

Same 11-cohort test split and GT-informed region-to-label rules as the valley-only control. For an axis with fewer KDE valleys than its rule requires, all required axis cuts are chosen by one- or two-threshold Otsu on the exact stored 48-bin histogram. No evaluation GT labels enter cut selection. Cohort means receive equal weight. Failed steps remain in the denominator.

| Cohort | Steps | Valley-only F1 | Otsu fallback F1 | Valley-only BA | Otsu fallback BA | Valley-only IoU | Otsu fallback IoU | Fallback steps | Infeasible steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Acute2020 | 742 | 0.272 | 0.620 | 0.285 | 0.727 | 0.261 | 0.533 | 465 | 2 |
| Acute2021 | 2035 | 0.234 | 0.603 | 0.253 | 0.740 | 0.240 | 0.526 | 1348 | 2 |
| Bjornson | 256 | 0.687 | 0.788 | 0.683 | 0.786 | 0.662 | 0.745 | 53 | 0 |
| Vaccine | 5270 | 0.323 | 0.613 | 0.342 | 0.733 | 0.310 | 0.548 | 3026 | 0 |
| FR-FCM-Z74D_hc | 24 | 0.221 | 0.571 | 0.230 | 0.632 | 0.230 | 0.598 | 15 | 0 |
| FR-FCM-Z74D_tissue | 180 | 0.116 | 0.512 | 0.121 | 0.623 | 0.126 | 0.545 | 146 | 0 |
| FRDR_covid19 | 2660 | 0.288 | 0.639 | 0.317 | 0.733 | 0.333 | 0.621 | 1472 | 4 |
| Lyoplate_DC | 120 | 0.358 | 0.719 | 0.380 | 0.815 | 0.315 | 0.596 | 71 | 0 |
| Lyoplate_bcell | 120 | 0.247 | 0.588 | 0.251 | 0.708 | 0.225 | 0.463 | 89 | 0 |
| Lyoplate_tcell | 144 | 0.376 | 0.522 | 0.469 | 0.715 | 0.373 | 0.539 | 63 | 0 |
| Lyoplate_treg | 120 | 0.222 | 0.663 | 0.240 | 0.748 | 0.177 | 0.527 | 84 | 0 |
| **Equal-cohort mean / total** | **11671** | **0.304** | **0.622** | **0.325** | **0.724** | **0.296** | **0.567** | **6832** | **8** |

Cut-selection counts:
- fallback: 6832
- infeasible: 8
- valley_only: 3867
- x_all: 4928
- x_otsu_fallback: 5022
- x_otsu_infeasible: 6
- x_selected: 751
- y_all: 4471
- y_otsu_fallback: 5742
- y_otsu_infeasible: 7
- y_selected: 487

Postprocessing status counts:
- landmark_error_x: 6
- landmark_error_y: 2
- ok: 10699
- unsupported_all_target: 964
