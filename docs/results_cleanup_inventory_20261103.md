# Paper-oriented results cleanup inventory (2026-11-03)

Read-only audit of `results/`: 57 top-level directories, 432.796 GiB of
regular files. Symlink targets are excluded from these sizes. The audit ran
through LSF job `50073431` using `jobs/audit_results_cleanup_20261103.py`.
No result or source directory was deleted.

## High-confidence cleanup candidates

| Path under `results/` | Size | Reason |
|---|---:|---|
| `_c2s_medium_smoke_20261027/` | <1 MiB | Smoke test; full medium runs exist. |
| `_tmp_definition_gt_4steps_20261027/`, `_tmp_definition_leak_4steps_20261027/`, `_tmp_maitnkt_gt_20261027/` | ~7 MiB combined | Four-step/plot diagnostics, not final experiments. |
| `rect2_gt_screen_aborted_49889296/` | 5 MiB regular files | Aborted screen; `rect2_gt_screen/` and `rect2_gt_min10/` are the completed alternatives. Contains 4,063 symlinks, so link targets must be checked before any deletion script follows links. |
| `gate_gpt-5.4_full_old_rect_eval/` | ~1 MiB | Historical direct rect-GT evaluation; old cell predictions are incompatible with the current benchmark. The paper comparison uses `gate_gpt-5.4_paper_gates_current_benchmark/`. |
| `rect1_joint_pilot/` | ~1 MiB | Pilot superseded by `rect1_joint_full/`. |
| `flat_cascade_gpt-6-luna-medium_before_inference_logging_20261029/` | 6.451 GiB | Superseded by the completed, timed `flat_cascade_gpt-6-luna-medium/` run used for Figure 5/Table S19. The old run's inference time and cost cannot be reconstructed. |
| `kde_otsu_rule/` | 7.565 GiB | `docs/kde_otsu_rule_20261102.md` explicitly calls this exploratory and excludes it from the direct reviewer control. Keep `kde_landmark_rule/` and `flowdensity_rule/`. |

The two large candidates above total about 14.0 GiB. The first five rows
recover little space but make the result tree clearer.

## Review or archive before deletion

| Path under `results/` | Size | Condition |
|---|---:|---|
| `visual_sol_medium_definition/` | 19.712 GiB | Contains a completed GPT-6 Sol scatter-only comparison. Current Figure 3/Table S9 show GPT-6 Luna and Qwen3.6-27B. Preserve its comparison summary and input provenance if the Sol analysis will be used later. |
| `visual_luna_medium_definition/pilot_concurrency40/`, `smoke/`, `preview/`, `vlm_scatter/` | Included in 62.347 GiB root | Pilot, smoke, preview and interrupted/early outputs coexist with the final `full_concurrency40/` run. Prune only after checking references and distinguishing shared `inputs/` from duplicate predictions. |
| `gate_gpt-6-luna-none_full/`, `gate_gpt-6-luna-low_full/`, `gate_gpt-6-luna-high_full/` | 29.383 GiB total | Effort sweep; the results appear in `docs/experiment_results.md`. Check manuscript and revision-letter usage before removing these unique API predictions. |
| `gate_gpt-6-luna_full/` and `gate_gpt-6-luna-medium_full/`; analogous `c2s_*_gpt-6-luna` and `c2s_*_gpt-6-luna-medium` roots | ~9.8 GiB each | Default and medium effort are distinct runs. Do not treat them as duplicates until every Table 2 and supplemental row is mapped to its exact source. |
| `compare_step09_gate_gpt-6-luna_vs_sol_medium/`, `compare_llm_flowdensity_valleys_20261030/`, `rect1_joint_examples/` | Small | Case/diagnostic outputs; check revision-letter references and whether analyses are still in progress. |

## Preserve for current paper or reviewer response

- `flat_cascade_gpt-6-luna-medium/` (21.300 GiB) and
  `flat_c2s_flowsom_gpt-6-luna-medium/`: final Figure 5/Table S19 comparison.
- `flat_cascade_gpt-6-luna-medium_incomplete_attempts_20261030/` (0.112 GiB):
  archived API attempts contribute to the reported Cascade time/cost. Retain
  the raw ledger unless its inclusion in final metadata has been audited.
- `gate_gpt-5.4_full_old/` (13.887 GiB): original paper GPT-5.4 result,
  despite the directory name. Also retain
  `gate_gpt-5.4_paper_gates_current_benchmark/` and the current-definition
  baseline until Table 2 provenance is fully frozen.
- Table 2 method outputs, including the C2S baselines, LLM-Gate runs,
  `flowdensity_rule/`, `unito/`, and `cymae/`.
- `visual_luna_medium_definition/full_concurrency40/`: Figure 3/Table S9.
- `gate_gpt-6-luna-medium_depletion/` and `gate_gpt-6-luna-medium_shift/`:
  Figure 4 and Tables S13/S15.
- `r3_comment4_luna_medium/` (58.883 GiB): reviewer R3 C4 sensitivity
  analysis; do not drop its raw runs while the response uses them.
- `kde_landmark_rule/`, `rect1_oracle/`, `rect1_joint_full/`,
  `rect2_gt_min10/`, `rect1_joint_letter/`, and
  `rect_proxy_bjornson_step08_approved/`: reviewer controls/proxy analyses.
- `cohort_grouped_20261101/` and `cohort_grouped_20261102/` serve different
  summaries (grouped comparison versus Qwen3.8), despite their dates.
- `_definition_rerun_20261027/`: source of the corrected-definition patch
  for full-test comparisons; verify provenance before pruning.
- `benchmark/`, `benchmark_flat/`, `data/`, and core `src/` modules. In
  particular, `src/llm_gate_flat/` and `src/flowsom_c2s/` implement Figure 5.

## Code cleanup rule

After a result directory is removed, inspect only its dedicated one-off
`jobs/` launch/plot/validation scripts for deletion. Keep shared evaluation,
inference, preprocessing, and plotting code. Existing working-tree edits in
`src/` were not changed by this audit.
