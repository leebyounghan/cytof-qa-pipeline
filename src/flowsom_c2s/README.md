# Whole-sample FlowSOM + C2S flat annotation

The pipeline clusters each sample once, asks the LLM for a **set of leaf labels**
per cluster, and assigns that entire set to every cell in the cluster.

## Evaluation

`src.flowsom_c2s.eval` reads `pred.json` and `cell2cluster.npz`, compares each
cell's predicted leaf set with the GT leaf set from `src.flat_eval`, and writes
`eval_flat.json` per sample plus `eval_summary_<dataset>.json` per cohort/plan.
The label vocabulary includes `Discard`. Multiple leaves from one gating step
are all retained. The multi-label metrics match Cascade's definitions:
subset accuracy, Hamming loss, micro/macro/weighted F1, and micro/macro Jaccard.
There is no C2S primary-leaf metric because a predicted set need not have one
unique primary leaf.

Submit computation through LSF using the project job scripts. The full
experiment recipe and directory layout are in [README_flat.md](../../README_flat.md).
