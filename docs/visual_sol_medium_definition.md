# Visual Gate experiment: GPT-6 Sol medium, definition only

Six full-test conditions use GPT-6 Sol with medium reasoning: VLM-Gate and
Agent-Gate, each with scatter, contour, and density images. They reuse the
validated visual inputs in `results/visual_luna_medium_definition/inputs` and
the same `benchmark/notes.yaml` definition snapshot as the saved GPT-6 Sol
LLM-Gate baseline in `results/gate_gpt-6-sol-medium_full`.

VLM-Gate makes one Chat Completions request per step, with a 32,768-token
completion ceiling. Agent-Gate uses the Responses API, requires an overlay
tool call on turn 1, allows up to four turns, and uses a uniform 65,536-token
output ceiling. All methods use temperature 1.0. Each of the six LSF array
elements runs up to ten requests concurrently, with at most four elements
active at once. Agent outputs are checkpointed every 100 steps.

Inference: `jobs/run_visual_sol_full40.sh`, LSF array `50040908`.
Evaluation: `jobs/eval_visual_sol_full40.sh`, dependency `done(50040908)`,
LSF array `50040910`. Equal-cohort summary:
`jobs/summarize_visual_sol_full.sh`, dependency `done(50040910)`,
LSF job `50040911`. The output root is
`results/visual_sol_medium_definition/full_concurrency40/`.

The definition snapshot is saved as
`results/visual_sol_medium_definition/notes.yaml.used`.

## Interrupted by API quota (2026-10-28)

The inference array finished with only VLM scatter successful (11 of 11
cohort files). The other five array elements exited after the API returned
HTTP 429 `insufficient_quota` / `credit_balance_exhausted`; this is an account
credit limit, not a model or parsing error. At interruption, 33 of 66
condition-cohort `pred.json` files existed: VLM contour 10, VLM density 9,
Agent scatter 3, Agent contour 0, and Agent density 0 in addition to VLM
scatter 11. The three Agent conditions also have partial cohort checkpoints.

The dependent evaluation array `50040910` and summary `50040911` were canceled
because complete inference is unavailable. No full six-condition comparison
has been calculated. After API credit is restored, resubmit the inference
script; it skips completed cohort files and Agent resumes from checkpoints.
Then submit evaluation and summary with dependencies on the new job IDs.

## Credit restored and resumed (2026-10-28)

LSF probe `50041562` successfully completed one GPT-6 Sol medium request
through Chat Completions and one through Responses. Full inference was
resubmitted as array `50041564`, reusing completed cohort files and Agent
checkpoints. Evaluation `50041565` depends on successful inference, and
equal-cohort summary `50041566` depends on successful evaluation. These jobs
are pending or running; the probe confirms API access but does not establish
that the remaining credit covers the whole experiment.

## Scatter-only scope (2026-10-28)

The user narrowed the GPT-6 Sol comparison to scatter input for both
VLM-Gate and Agent-Gate. VLM contour and density had already completed
inference; they will not be evaluated or included in the comparison. Agent
contour and density array elements `50041564[5-6]` were canceled while
inference was still running. The original six-condition evaluation `50041565`
and summary `50041566` were canceled. Agent scatter `50041564[4]` continues.

The replacement scatter-only evaluation is `50041586` (VLM scatter and Agent
scatter, depending on successful Agent scatter inference). The summary job is
`50041587`, comparing those two conditions with the saved GPT-6 Sol medium
LLM-Gate baseline. Its planned outputs are
`full_concurrency40/visual_scatter_comparison.md` and `.json`.

## Completed scatter-only comparison

Agent scatter inference `50041564[4]` finished all 11 cohorts. The two
scatter evaluations `50041586[1-2]` both completed with empty stderr logs.
The first summary `50041587` exited because of an unmatched parenthesis in
`jobs/summarize_visual_full_vlm.py`; after correcting that syntax error, job
`50041624` completed and wrote `full_concurrency40/visual_scatter_comparison.md`
and `.json`. Each row covers 11,671 test steps and weights 11 cohorts equally.

| Method | Macro F1 | Balanced accuracy | Hull IoU |
|---|---:|---:|---:|
| LLM-Gate | 0.671 | 0.747 | 0.617 |
| VLM-Gate scatter | 0.691 | 0.790 | 0.650 |
| Agent-Gate scatter | 0.691 | 0.793 | 0.646 |

The two visual methods use their respective prompts, so differences from
LLM-Gate cannot be attributed to the image alone. Agent scatter did not
improve F1 over VLM scatter at three-decimal precision and had a slightly
lower Hull IoU, despite a slightly higher balanced accuracy.
