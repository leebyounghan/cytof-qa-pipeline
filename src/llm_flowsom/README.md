# LLM-C2S — clustering paradigm (FlowSOM cell-sentences)

The "clustering + LLM" rows of the paper. A fork of `src/llm` in which the
per-step cell-sentences come from **FlowSOM clusters** (SOM + hierarchical
metaclustering on the 2-D `(x, y)` plane) instead of the flowDensity grid:
one anchor cell per cluster is described to the LLM, the predicted label is
propagated to every cell in the cluster, and the result is scored by the
shared `src.eval`.

Only the sentence source differs from `src/llm` — prompts, inference
client, propagation, and metrics follow the same shapes. Module layout:

```
src/llm_flowsom/
  c2s.py                       # parquet → c2s.json (FlowSOM clusters + anchor sentences)
  c2s_prompt.py                # c2s.json → per-step prompt (c2s_{ablation}/)
  inference/run_openai.py      # OpenAI-compatible client (vLLM or API)
  postprocess/propagate.py     # cluster label → per-cell prediction.json
  scripts/run_vllm_flowsom.sh  # unified runner: serve → infer → propagate → eval
```

## Prerequisites (one-time, CPU)

```bash
python -m src.llm_flowsom.c2s --benchmark benchmark/ --data-dir data/ \
    --cluster-method flowsom --flowsom-grid 10 --flowsom-k 20
for s in benchmark/*/*/; do
    python -m src.llm_flowsom.c2s_prompt --batch "$s" --save_md --top-hvp-n 10
done
```

## Run (paper configuration)

```bash
SPLIT=test bash src/llm_flowsom/scripts/run_vllm_flowsom.sh \
    --config src/llm/configs/models/qwen3_6_27b_tb8096.yaml \
    hvp10 output/c2s_clu_qwen3_6_27b
```

The runner starts `vllm serve` from the YAML, fans out one client per
cohort, then propagates and evaluates. `SAMPLES="..."` restricts the run to
named samples (smoke tests); `GENERATION_SEED=N` threads a generation seed
(used by `scripts/run_c2s_seed_sweep.sh` for the 10-seed stability runs).

For the density-grid paradigm (flowDensity cell-sentences) and everything
shared between the two forks, see `src/llm/README.md`.
