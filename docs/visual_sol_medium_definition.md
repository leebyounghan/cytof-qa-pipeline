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
`results/visual_sol_medium_definition/notes.yaml.used`. The final comparison
will be written to `visual_full_comparison.md` and `.json` under the output
root after evaluation succeeds. These jobs were submitted on 2026-10-28;
their status and results must be checked before interpreting the experiment.
