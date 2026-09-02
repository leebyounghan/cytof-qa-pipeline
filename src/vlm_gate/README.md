# VLM-Gate Pipeline (`src.vlm_gate`)

Text + 2-D scatter image, **single-shot** — no tools, no loop. The third corner of the gating-method triangle:

| | `src.llm_gate` | `src.vlm_gate` | `src.agent_gate` |
|---|---|---|---|
| Inference | one chat turn | one chat turn | tool-call loop |
| Model sees | text axis distribution + categories | text + decorated scatter image | text + decorated scatter + iterative overlay tool |
| Tools | — | — | `render_gate_overlay` |
| Use case | text-only baseline | "does the image alone help?" | "does iterative visual verify help?" |

`vlm_gate` is the clean ablation between the other two: it isolates the effect of **having the scatter image as input** from the effect of **having an agentic verification loop**. `pred.json` schema and propagation rules are byte-identical to `llm_gate`'s, so all three modules produce comparable F1 numbers via the same `src.eval`.

```
parquet → gate_task.json + scatter_bg.png → gate_{ablation}/cells.md → pred.json → prediction.json → eval_summary.json
                  (Step 1)                            (Step 2)             (Step 3)        (Step 4)         (Step 5)
```

`gate_task.json`, `cells.md`, and `scatter_bg.png` are **shared with `src.agent_gate`** — same paths, same content. Either module's `task.py` / `prompt.py` produces them.

---

## Step 1 — `src.vlm_gate.task`

Same as `src.agent_gate.task`. Renders `gate_task.json` (with `scatter_bg` + `scatter_extent` fields) and the decoration-free `scatter_bg.png` per step.

```bash
python -m src.vlm_gate.task \
    --benchmark benchmark/ --data-dir data/ --n-workers 8
```

---

## Step 2 — `src.vlm_gate.prompt`

Same as `src.llm_gate.prompt` and `src.agent_gate.prompt`. Writes `cells.md` per step.

```bash
python -m src.vlm_gate.prompt \
    --batch benchmark/Acute2020/994585_Normalized --save_md
```

---

## Step 3 — `src.vlm_gate.inference.run_openai`

Async OpenAI runner. Per `(sample, step)`:

1. Render the decorated input scatter (axes + ticks + grid) by wrapping `scatter_bg.png` via `utils/scatter_render.render_input_scatter`.
2. Build a multimodal user message: `[{type:'text', text:cells.md}, {type:'image_url', image_url:{url:data:image/png;base64,...}}]`.
3. Single chat completion (no tools). Parse the JSON answer.

```bash
export OPENAI_API_KEY=sk-...

python -m src.vlm_gate.inference.run_openai \
    --dataset_path benchmark/Acute2020 \
    --model        gpt-5.4 \
    --concurrency  10 \
    --output_path  results/vlm_gpt-5.4_full/Acute2020/pred.json
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--concurrency` | `10` | Lower than `llm_gate` (10 vs 20) — each request carries an image. |
| `--max_tokens` | `8192` | Per-request cap. |
| `--temperature` | `0.0` | Greedy by default. `--top-p`, `--presence-penalty`, `--frequency-penalty` are pass-through OpenAI-native. |
| `--save-raw` | off | Persist `raw` text per step. |
| `--ablation_slug` | `full` | Reads `gate_{slug}/cells.md`. |
| `--sample`, `--max_samples`, `--seed` | — | Subsetting. |
| `--perturbation`, `--hard-depletion`, `--hard-shift` | — | Same plumbing as `llm_gate` / `agent_gate`. |

---

## Step 4 — `src.vlm_gate.postprocess.propagate`

Byte-identical to `src.llm_gate.postprocess.propagate`. Same membership rule, same `--tiebreak`, same perturbation cross-check.

```bash
python -m src.vlm_gate.postprocess.propagate \
    --eval_path  results/vlm_gpt-5.4_full/Acute2020/pred.json \
    --dataset    Acute2020 \
    --benchmark  benchmark/ --data-dir data/ \
    --method     vlm_gpt-5.4_full \
    --output_dir results/vlm_gpt-5.4_full
```

---

## Step 5 — `src.eval`

Method-agnostic.

```bash
python -m src.eval --predictions results/vlm_gpt-5.4_full \
    --benchmark benchmark/ --data-dir data/
```

---

## End-to-end recipe (Setting 1)

```bash
DS=Acute2020
SAMPLE=994585_Normalized
METHOD=vlm_gpt-5.4_full

python -m src.vlm_gate.task \
    --benchmark benchmark/ --data-dir data/ \
    --datasets $DS --samples $SAMPLE --n-workers 4

python -m src.vlm_gate.prompt \
    --batch benchmark/$DS/$SAMPLE --save_md

export OPENAI_API_KEY=sk-...
python -m src.vlm_gate.inference.run_openai \
    --dataset_path benchmark/$DS --sample $SAMPLE \
    --model gpt-5.4 \
    --output_path results/$METHOD/$DS/pred.json --concurrency 10

python -m src.vlm_gate.postprocess.propagate \
    --eval_path results/$METHOD/$DS/pred.json \
    --dataset $DS --benchmark benchmark/ --data-dir data/ \
    --method $METHOD --output_dir results/$METHOD

python -m src.eval --predictions results/$METHOD \
    --benchmark benchmark/ --data-dir data/
```

Or the unified runner:

```bash
SPLIT=test OPENAI_API_KEY=sk-... \
    bash src/vlm_gate/scripts/run_vlm_gate.sh \
        --config src/vlm_gate/configs/models/gpt_5_4.yaml \
        full results/vlm_gpt-5.4_full
```

---

## File layout

```
src/vlm_gate/
├── README.md
├── task.py                      # Step 1 (gate_task.json + scatter_bg.png; shared paths with agent_gate)
├── prompt.py                    # Step 2 (cells.md; identical to llm_gate / agent_gate)
├── inference/
│   ├── data_loader.py           # surfaces scatter_bg_path + scatter_extent per group
│   ├── output_schema.py         # gate parser + pred.json assembler
│   ├── system_prompts.py        # text+image single-shot guidance — no tool protocol
│   └── run_openai.py            # Step 3 — multimodal single-shot
├── utils/
│   ├── scatter_render.py        # render_input_scatter — full copy of agent_gate's render_bare_scatter
│   ├── axis_distribution.py
│   ├── density_clouds.py
│   ├── joint_distribution.py
│   └── task_input_adapter.py
├── postprocess/
│   └── propagate.py             # Step 4
├── configs/models/gpt_5_4.yaml
└── scripts/
    ├── _config_to_shell.py
    └── run_vlm_gate.sh          # unified runner (1→2→3→4→5)
```
