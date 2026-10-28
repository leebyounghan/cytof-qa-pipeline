# Visual gate experiment: definition only

Six conditions: VLM-Gate and Agent-Gate, each with scatter, contour, and
density images. All use `gpt-6-luna`, API reasoning effort `medium`, model
temperature 1.0, the benchmark test split, and `definition` from
`benchmark/notes.yaml`. A snapshot is in `notes.yaml.used`.

The same `cells.md` text and parent-cell coordinate extent are reused for
the three image styles at each step. The scatter is the established
60,000-point rendering. Density is a log-scaled two-dimensional histogram;
contour is a Gaussian-smoothed version of the same histogram. The bin count
is `clip(1.5 * sqrt(min(n_cells, 60000)), 65, 180)`, contour smoothing
sigma is 1.8 bins, and images are 640 by 640 pixels before the shared
axis/tick/grid wrapper is applied. Background images contain no labels or
ground truth. Their rendered model-visible examples are in `preview/`.

The VLM runner makes one Chat Completions request per step. Agent-Gate uses
the Responses API to retain function tools with medium reasoning. Turn 1
requires `render_gate_overlay` with a proposed gate; later turns may call it
again or finish, with a maximum of four turns. Tool output contains the
same plot style with gate outlines and the existing coordinate, overlap,
and range diagnostics. The Agent trace stores turn count, tool-call count,
tool proposals, and diagnostic text.

The plot name in the system prompt and tool description is adjusted to match
the image style. All other prompt text is held fixed within each method.
The VLM and Agent methods retain their respective original system prompts.

Input generation completed in LSF array `49879565`. The initial full run
`49883257` and its dependent evaluation `49883288` were canceled after
identifying the low API concurrency (3 clients × 4 requests). Completed
outputs from that interrupted run remain isolated in the original condition
folders and are not part of the restarted comparison.

The one-sample-per-cohort pilot used 4 clients × 10 concurrent requests,
selected the first sorted prepared test sample in each cohort, and wrote to
`results/visual_luna_medium_definition/pilot_concurrency40/`. LSF array
`49887682` completed 61 of 66 conditions. Five jobs exited with code 255
before the script started on compute nodes; retries `49890754` and `49891708`
completed them. Validator `49892074` confirmed that all 66 outputs cover the
same 260 steps within each condition, with no missing gates, and that every
Agent step made an overlay tool call. Evaluation `49892656` and equal-cohort
summary `49892837` completed.

| Pilot condition | Macro F1 | Balanced accuracy | Hull IoU |
|---|---:|---:|---:|
| VLM scatter | 0.654 | 0.768 | 0.592 |
| VLM contour | 0.605 | 0.710 | 0.532 |
| VLM density | 0.646 | 0.760 | 0.580 |
| Agent scatter | 0.619 | 0.748 | 0.546 |
| Agent contour | 0.599 | 0.708 | 0.522 |
| Agent density | 0.635 | 0.759 | 0.559 |

The pilot checks throughput and end-to-end validity; it is not the full test
result. The restarted full inference is LSF array `49892404`, with 6 long
condition jobs and at most 4 running simultaneously, each allowing 10 API
requests. Outputs go to `full_concurrency40/`. Dependent full evaluation is
`49892620`; it must not be reported as completed until LSF marks it DONE.
