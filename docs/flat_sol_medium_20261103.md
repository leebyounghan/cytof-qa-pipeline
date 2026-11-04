# Flat Cascade and C2S FlowSOM, GPT-6 Sol medium

## Scope and configuration

- Test split, 17 cohort/plan cases: eight full-tree cohorts plus Acute2020, Acute2021, and Vaccine at L1/L2/L3.
- 646 sample/plan tasks. Manifest: `jobs/flat_sol_manifest_20261103.json`.
- Model `gpt-6-sol`, reasoning effort `medium`, temperature 1.0, maximum completion tokens 32768.
- C2S reuses the existing `benchmark_flat/` preparation. Its SOM uses 10x10 nodes and the plan's leaf count for metaclusters.
- Cascade uses the existing flat runner with smallest-area tiebreak and per-call inference usage, time, and estimated cost logs.
- Results: `results/flat_cascade_gpt-6-sol-medium/` and `results/flat_c2s_flowsom_gpt-6-sol-medium/`.
- Evaluation uses the revised direct multi-label leaf-set methods in `src.flat_eval` and `src.flowsom_c2s.eval`.

## Execution status, 2026-11-03

- Manifest preparation job 50073348 completed and checked prerequisites for all cases.
- Initial C2S array 50073349 failed at API authentication (HTTP 401 `invalid_api_key`).
- Initial Cascade array 50073350 failed before inference because the pricing table lacked Sol. The pricing entry was added from the official OpenAI model pricing page.
- Authentication probe jobs 50073378 and 50073379 both failed with HTTP 401 `invalid_api_key`. The probe established that the job inherits no `OPENAI_API_KEY` and reads the value from project `.env`; it did not print the key.
- No Sol predictions or evaluations have completed. The inference arrays must only be resubmitted after a successful one-call authentication probe.

## Job scripts

- `jobs/check_flat_sol_api_20261103.sh`: one-call authentication probe.
- `jobs/run_flat_sol_cascade_sample_20261103.sh`: 646-element inference array, capped at four simultaneous tasks.
- `jobs/run_flat_sol_c2s_case_20261103.sh`: 17-element inference array, capped at four simultaneous cases.
- `jobs/eval_flat_sol_cascade_case_20261103.sh` and `jobs/eval_flat_sol_c2s_case_20261103.sh`: evaluation arrays, each capped at two simultaneous cases.

Pricing source: <https://developers.openai.com/api/docs/models/gpt-6-sol>.
