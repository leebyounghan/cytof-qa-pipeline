# Final flat-versus-cascade experiment

Figure 5 of the manuscript compares whole-sample Flat C2S with the LLM-Gate cascade on the same test samples and leaf vocabulary. GPT-6 Sol medium and Qwen3.6-27B appear in the figure; Table S21 in the renumbered supplement also includes GPT-6 Luna medium.

## Cases and plans

Acute2020, Acute2021, and Vaccine share three tree depths: L1 has 16 leaves and 22 planned steps, L2 has 38 leaves and 41 steps, and L3 has 83 leaves and 62 steps. The other eight cohorts use their full native tree. This yields 17 cohort-depth settings and 646 test sample-plan pairs. Plans are stored in `benchmark_flat/<cohort>/`. The plan filename selects the slug `L1_coarse`, `L2_identity`, or `default`.

## Flat C2S

`src/flowsom_c2s/c2s.py` fits a 10×10 MiniSom to every sample's arcsinh-transformed protein channels without z-scoring, using 30,000 randomly sampled training steps. Average-linkage agglomerative clustering collapses its 100 nodes to K metaclusters, where **K equals the leaf count of the plan being evaluated**. The cluster assignment is saved as `cell2cluster.npz` under `benchmark_flat/<cohort>/<sample>/`; the rendered whole-sample prompt and `c2s.json` are also kept under `benchmark_flat/`.

The LLM returns a JSON leaf list for every cluster in one call per sample. Multiple leaves, including multiple leaves originating at one gating step, are retained. `src/flowsom_c2s/eval.py` maps each leaf set directly to member cells and compares it with GT leaf sets from the canonical parquet. It does not reconstruct Step columns or apply cascade parent masks.

## LLM-Gate cascade

`src/llm_gate_flat/runner.py` walks the expert tree in order. Every parent mask comes from earlier predicted Step columns. An empty predicted parent skips that branch while other branches continue. The runner writes `predicted_steps.parquet`, `cascade_meta.json`, per-step predictions, and `inference_calls.jsonl` with API time, token usage, and estimated cost. The smallest-area rule resolves overlapping returned gates. `src/flat_eval.py` derives leaf sets from the predicted Step columns.

Both methods use `MultiLabelBinarizer` over the same plan's leaves plus `Discard` and report multi-label F1 macro/micro/weighted, Jaccard macro/micro, Hamming loss, and subset accuracy. Per-sample metrics are averaged within each cohort-depth setting. The single primary-leaf rule used to draw UMAPs and confusion matrices does not affect these scores.

## Result locations and run definitions

| Backbone | Cascade | Flat C2S |
|---|---|---|
| GPT-6 Sol medium | `results/flat_cascade_gpt-6-sol-medium/` | `results/flat_c2s_flowsom_gpt-6-sol-medium/` |
| GPT-6 Luna medium | `results/flat_cascade_gpt-6-luna-medium/` | `results/flat_c2s_flowsom_gpt-6-luna-medium/` |

Each result root has `<plan_slug>/<cohort>/` children. The Sol manifest is `jobs/flat_sol_manifest_20261103.json`; inference and evaluation are submitted with `jobs/run_flat_sol_cascade_sample_20261103.sh`, `jobs/run_flat_sol_c2s_case_20261103.sh`, `jobs/eval_flat_sol_cascade_case_20261103.sh`, and `jobs/eval_flat_sol_c2s_case_20261103.sh`. Qwen results and exact settings are identified in their corresponding job scripts. Run all stages through `bsub` with the project `.venv/bin/python`.
