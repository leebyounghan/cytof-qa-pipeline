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

## Matched LLM-Gate pilot comparison

The saved LLM-Gate `gpt-6-luna` medium test gates use an identical
`notes.yaml.used` definition snapshot. LSF job `49893506` selected exactly the
same 11 pilot samples and 260 steps from those gates, then propagated and
reevaluated them with the same `smallest` tie break and GT evaluator. Each
cohort contributes equally to the table. The comparison is saved in
`pilot_concurrency40/llm_gate_matched/comparison.md` and `comparison.json`.

| Method | Macro F1 | Delta F1 vs LLM | Balanced accuracy | Hull IoU |
|---|---:|---:|---:|---:|
| LLM-Gate | 0.673 | +0.000 | 0.762 | 0.631 |
| VLM scatter | 0.654 | -0.019 | 0.768 | 0.592 |
| VLM contour | 0.605 | -0.068 | 0.710 | 0.532 |
| VLM density | 0.646 | -0.027 | 0.760 | 0.580 |
| Agent scatter | 0.619 | -0.054 | 0.748 | 0.546 |
| Agent contour | 0.599 | -0.074 | 0.708 | 0.522 |
| Agent density | 0.635 | -0.038 | 0.759 | 0.559 |

This is a one-sample-per-cohort pilot and is not a full test result. The
LLM-Gate predictions came from a previous full run; visual gates were generated
in the new pilot. The prompt and interaction setup differ by method, so the
differences cannot be attributed to image style alone.

## Full-run recovery and VLM-first evaluation

All three VLM conditions in `49892404` wrote 11 cohort predictions each. The
three Agent conditions exited after a few individual Responses API calls
returned `incomplete` with reason `max_output_tokens` at the 32,768-token cap.
Those cohort outputs were not written by the original runner, so their API
requests need to be repeated. The runner now checkpoints each successful batch
of 100 steps and resumes only missing steps. The Agent cap is 65,536 tokens;
the VLM cap remains 32,768. The Agent-only resume is LSF array `49963397`.
The old evaluation dependency could never become true and was canceled.

VLM evaluation completed in `49964049` and the equal-cohort summary in
`49964056`. The original Agent-only evaluation `49964050` was canceled after
Agent density exited. Replacement Agent evaluation `49968233` depends on the
successful scatter and contour array elements plus the density retry. Final
seven-method summary `49968265` depends on that evaluation.

The full VLM scatter and contour evaluations completed in LSF `49964049[1-2]`.
The same 11,671 test steps were evaluated in each method; an equal-cohort
comparison with the saved LLM-Gate Luna-medium run is in
`full_concurrency40/vlm_scatter_contour_partial.md`.

| Full-test method | Macro F1 | Balanced accuracy | Hull IoU |
|---|---:|---:|---:|
| LLM-Gate | 0.651 | 0.747 | 0.602 |
| VLM scatter | 0.655 | 0.763 | 0.603 |
| VLM contour | 0.625 | 0.725 | 0.554 |

The final VLM comparison, including density, is in
`full_concurrency40/vlm_full_comparison.md` and `vlm_full_comparison.json`:

| Full-test method | Macro F1 | Balanced accuracy | Hull IoU |
|---|---:|---:|---:|
| LLM-Gate | 0.651 | 0.747 | 0.602 |
| VLM scatter | 0.655 | 0.763 | 0.603 |
| VLM contour | 0.625 | 0.725 | 0.554 |
| VLM density | 0.646 | 0.756 | 0.591 |

Agent density encountered an `invalid_prompt` API rejection in one Acute2021
step after 1,499 successes had been checkpointed. LSF retry `49966270` is
resuming with the same prompt and 65,536-token cap. The retry passed that step
and completed Acute2021. The original Agent evaluation dependency was canceled
because its Agent inference array contained an exited element.
