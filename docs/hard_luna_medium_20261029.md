# GPT-6 Luna medium hard perturbation run (2026-10-29)

## Design

- Setting 2 curated tasks: all 43 hard depletions and 52 hard shifts from `benchmark/hard_depletion.yaml` and `benchmark/hard_shift.yaml`.
- Nominal magnitudes only: depletion fraction 1.0 (`frac100`) and shift scale 1.0 (`scale100`).
- In-panel `test` samples at each task's target step.
- Regenerate `gate_task__<hard_id>.json` and `gate_full__<hard_id>/cells.md` per task. Prompts use `--ablation full --note-mode definition`.
- Model: `gpt-6-luna`, API reasoning effort `medium`, configuration `src/llm_gate/configs/models/gpt_6_luna.yaml` (temperature 1.0, max completion tokens 32768).
- Propagation tie break: `smallest`.
- Depletion output: `results/gate_gpt-6-luna-medium_depletion/`.
- Shift output: `results/gate_gpt-6-luna-medium_shift/`.
- Each output has `{scenario}/{dataset}_step{NN}/{frac100|scale100}/{dataset}/pred.json` and per-sample `prediction.json` files beneath the magnitude directory. `src.eval_hard` writes `eval_hard_summary.json` at each root.

## LSF jobs

- Pilot array index 1: `50044976` (`DONE`). It produced 12 prompts and 12 raw predictions for `hiv_acute2020_step29`, followed by 12 propagated `prediction.json` files.
- Remaining indices 2–95, concurrency cap 4: `50044978`, dependent on pilot success. Submitted and running. Indices 3 and 10 hit the original 16 GB LSF memory limit during task generation; pending indices 14–95 were modified to request 48 GB.
- Repair job `50044984` completed successfully. It reran array indices 3, 10, and 60. The first two had hit the original 16 GB memory limit; index 60 had exited at job startup with code 255.
- Evaluation array indices 1–2: `50044979` completed successfully for both depletion and shift.

Runner scripts: `jobs/run_gate_luna_medium_hard_20261029.sh`, `jobs/repair_gate_luna_medium_hard_20261029.sh`, and `jobs/eval_gate_luna_medium_hard_20261029.sh`. All computation runs in CPU jobs on `i2c2_normal` and uses this checkout's `.venv/bin/python`.

## Status

Full experiment and evaluation are complete. Depletion evaluation matched 1,677 of 1,677 prediction files; shift evaluation matched 1,883 of 1,883. Both reported zero size mismatches. Summary files are `results/gate_gpt-6-luna-medium_depletion/eval_hard_summary.json` and `results/gate_gpt-6-luna-medium_shift/eval_hard_summary.json`.

## Figure 4 comparison extension

The main-paper Figure 4 requires a 1% surviving depleted population and a nominal reference for the shift delta. The original nominal hard run used full depletion (`frac100`) and full calibrated shift (`scale100`), so it does not itself supply these comparisons.

- The full-run `results/gate_gpt-6-luna-medium_full/predictions` provides the zero-magnitude baseline. Its `notes.yaml.used` is byte-identical to both hard-run snapshots. LSF job `50045098` linked 1,677 depletion and 1,883 shift predictions into `frac0` and `scale0` layouts, matching the same samples and steps without another API call.
- LSF pilot `50045099` completed the first `frac99` depletion task. The remaining 42 tasks are in array `50045101` with a concurrency cap of 4; repair job `50045104` is dependent on array termination.
- Shift baseline comparison evaluation job `50045100` was submitted. Depletion `frac0` versus `frac99` comparison evaluation job `50045105` depends on repair completion. Both write `eval_hard_comparison_summary.json` in their respective result roots.
- Figure job `50045107` completed. It wrote `fig4_luna_medium_hard.pdf`, `.png`, and a per-scenario number file under `results/gate_gpt-6-luna-medium_depletion/figures/`. It uses the main Word Figure 4's two scientific quantities, with a GPT-6 Luna medium-only presentation.
- `frac99` removes 99% of matching cells. It is the paper's 1%-remaining condition. `scale100` applies each YAML-calibrated factor at its specified magnitude; it is not a universal maximal shift.

The main-paper comparison needs `frac99` and `scale0`. The supplementary three-level depletion table also uses `frac90`, which removes 90% of matching cells (10% survive).

## `frac90` depletion extension

- Same 43 curated depletion tasks, test split, definition-only prompts, `gpt-6-luna` medium effort, and propagation settings as `frac99`. The hard depletion fraction is `0.9`; outputs and task-specific prompt files use the `frac90` suffix.
- LSF pilot `50045114` (array index 1) completed successfully on `i2c2_normal`. Array `50045115` covers indices 2–43 with concurrency cap 4 and is running after pilot success.
- Repair job `50045116` checks all 43 completion markers after the array ends and reruns incomplete items. Evaluation job `50045117` depends on that repair and the `frac99` comparison evaluation (`50045105`). It evaluates `frac0`, `frac90`, and `frac99` into `results/gate_gpt-6-luna-medium_depletion/eval_hard_threelevel_summary.json`.
- The `frac90` bulk array, repair, and three-level evaluation all completed. Repair found zero incomplete tasks; all 43 `frac90` completion markers are present. Three-level evaluation matched 5,031 predictions (1,677 per magnitude) with zero size mismatches. Its summary is `results/gate_gpt-6-luna-medium_depletion/eval_hard_threelevel_summary.json`.
