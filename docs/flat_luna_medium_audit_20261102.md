# Flat GPT-6 Luna medium audit (2026-11-02)

The completed experiment covers 17 cohort/plan combinations and 646 test
sample/plan pairs per method. C2S evaluation scores the original cluster leaf
sets directly through `cell2cluster.npz`, retaining all selected leaves.

- Final Cascade logs contain 19,270 successful API attempts and zero recorded
  API error attempts. Completed steps have inference records with token usage,
  API time, and estimated cost.
- Final C2S predictions passed cluster response validation. An earlier repair
  job re-requested 56 invalid or missing cluster answers across five cases.
- No final Cascade sample was aborted. No sample had multi-label macro F1 exactly
  zero or was entirely predicted as `Discard`.
- The C2S model selected multiple leaves from the same Step in 442 of 20,752
  clusters, covering 23,937,110 of 197,521,044 cell/plan assignments (12.1%).
  The evaluator retains and scores every leaf.

See [the final results](../results/flat_luna_medium_summary_20261102.md) for
the canonical comparison.
