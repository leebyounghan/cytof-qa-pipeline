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
