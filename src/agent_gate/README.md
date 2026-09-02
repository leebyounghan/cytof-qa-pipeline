# Agent-Gate Pipeline (`src.agent_gate`)

A close cousin of `src.llm_gate`. Same task (per-category axis-aligned rectangular gates on a 2-D `(x_marker, y_marker)` plane), same per-step `pred.json` schema, same propagation rules, same `src.eval`. The one difference is **how the model fills in the gates**:

| | `src.llm_gate` | `src.agent_gate` |
|---|---|---|
| Inference style | one chat turn → final JSON | iterative tool-call loop → final JSON |
| Model sees | text axis distribution + categories | the same text prompt, **plus** an overlay image of its own proposed gates drawn on the parent population's scatter — re-rendered every revision |
| New tool | — | `render_gate_overlay(gates)` — returns the overlay PNG + a text read-back of how propagation will interpret each gate |
| Target API | OpenAI Chat Completions (and OpenAI-compat vLLM) | **OpenAI hosted multimodal only** (GPT-5.4 etc.) — needs vision + tool calling |

If you've used `src.llm_gate`, the only commands that change are Step 3 (`run_openai` → `run_agent`) and Step 1 (`task` now also writes a background scatter PNG). Steps 2, 4, and 5 are identical in shape to llm_gate's, just renamespaced under `src.agent_gate`.

```
parquet → gate_task.json + scatter_bg.png → gate_{ablation}/cells.md → pred.json → prediction.json → eval_summary.json
                  (Step 1)                            (Step 2)             (Step 3)        (Step 4)         (Step 5)
```

---

## How the agentic Step 3 actually works

For each `(sample, step)`:

1. The runner sends `system_prompt` + `cells.md` (the same text prompt llm_gate uses) and exposes one tool — `render_gate_overlay`.
2. **Turn 1**: `tool_choice="required"` — the model **must** propose an initial set of gates and call `render_gate_overlay`. The tool draws those boxes on the pre-rendered parent scatter (`scatter_bg.png`, in **data coordinates** via `imshow(extent=…)` so a box at `x=5.2,y=2.1` lands on the cells at `x=5.2,y=2.1`) and returns:
   - a base64 PNG (the overlay), forwarded to the model in a follow-up `user` message because the OpenAI `tool` role cannot carry images;
   - a text read-back: every parsed gate's coordinates as propagation will see them, plus warnings for out-of-data-range boxes and pairwise overlaps.
3. **Turns 2..N-1**: `tool_choice="auto"` — the model inspects the overlay, decides whether any box leaks into a neighbouring cluster / misses its blob / overshoots the data, and either calls `render_gate_overlay` again with revised gates or commits.
4. **Final turn**: `tool_choice="none"` — the model is forced to output a plain-text JSON object. That text is what `extract_gates` parses, so the rest of the pipeline (`assemble_eval_output` → `propagate` → `src.eval`) is byte-identical to llm_gate's.

Tool budget knob: `--max-turns N` (default 4 = 1 forced render + up to 2 free revisions + 1 forced text turn).

Why a pre-rendered background:
`task.py` renders the (slow) 60-k-point scatter ONCE per step into `scatter_bg.png` whose pixel bounds map exactly to `scatter_extent`. The tool just `imshow`'s that PNG and overlays the boxes — cheap enough to run every revision. Coordinate mapping is verified end-to-end (a box at the empirical density-max lands on the brightest blob in the overlay).

---

## Step 1 — `src.agent_gate.task`: parquet → `gate_task.json` + `scatter_bg.png`

Same as llm_gate's task generator, plus two new artefacts per step:

- `scatter_bg.png` — a decoration-free density scatter of parent `(x, y)` cells. Axes fill the figure, ticks/labels off, no `bbox_inches='tight'` — so the image's pixels correspond one-to-one to a known `[xmin, xmax, ymin, ymax]` data rectangle.
- `scatter_extent` (in `gate_task.json`) — that rectangle. The overlay tool reads it and reuses it verbatim with `imshow(..., extent=scatter_extent, origin='upper', aspect='auto')`.

```bash
python -m src.agent_gate.task \
    --benchmark benchmark/ --data-dir data/ --n-workers 8

# Test split only
python -m src.agent_gate.task \
    --benchmark benchmark/ --data-dir data/ \
    --split benchmark/splits.json --split-set test
```

Flags inherited from llm_gate: `--modality {auto|cytof|flow}`, `--n-workers N`, `--skip-existing`, `--datasets / --samples / --split`, plus the perturbation knobs `--perturbation` / `--hard-depletion` / `--hard-shift`. A perturbed run writes `gate_task__{slug}.json` + `scatter_bg__{slug}.png` so the clean and perturbed scatters never shadow each other.

---

## Step 2 — `src.agent_gate.prompt`: `gate_task.json` → `gate_{ablation}/cells.md`

Identical to `src.llm_gate.prompt`. The agentic system prompt explains the tool-call protocol; the per-step `cells.md` only describes the categories + axis distribution, not the loop. Same `--with-joint-2d`, `--without-note`, `--without-notes-yaml`, and perturbation flags.

```bash
python -m src.agent_gate.prompt \
    --batch benchmark/Acute2020/994588_Normalized --save_md
```

---

## Step 3 — `src.agent_gate.inference.run_agent`: prompts → `pred.json`

Async OpenAI runner with the tool-call loop above. Walks `--dataset_path`, fires one agentic loop per `(sample, step)`, gathers them with a semaphore.

```bash
export OPENAI_API_KEY=sk-...

python -m src.agent_gate.inference.run_agent \
    --dataset_path benchmark/Acute2020 \
    --model        gpt-5.4 \
    --max-turns    4 \
    --concurrency  10 \
    --output_path  results/agent_gpt-5.4_full/Acute2020/pred.json
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--max-turns` | `4` | Min 2. Turn 1 forces the tool, last turn forces text. |
| `--concurrency` | `10` | Lower than llm_gate (each loop fires N turns + uploads images). |
| `--max_tokens` | `8192` | Per-turn cap. |
| `--temperature` | `0.0` | Greedy by default. `--top-p`, `--presence-penalty`, `--frequency-penalty` are pass-through OpenAI-native. |
| `--save-raw` | off | Persist `raw` (final text) AND `agent_trace = {n_turns, n_tool_calls, tool_diagnostics, finish_reason}`. **Image bytes are not saved** — only the text read-backs. |
| `--ablation_slug` | `full` | Reads `gate_{slug}/cells.md`. |
| `--sample`, `--max_samples`, `--seed` | — | Subsetting like llm_gate. |
| `--perturbation`, `--hard-depletion`, `--hard-shift` | — | Same plumbing as llm_gate. |

The output JSON has the same `{meta, samples}` schema as llm_gate's `pred.json` (categories without a valid gate stored as `null`; absent sentinels stored as `{"absent": true, "rationale": ...}`; polygon and multi-rectangle forms supported). With `--save-raw`, each step also carries `raw` and `agent_trace`.

### The tool — `render_gate_overlay`

Single argument: a `gates` JSON object keyed by category name. Each value is one of:

- `{x_min, x_max, y_min, y_max, rationale}` — single rectangle (default);
- `[{x_min, x_max, y_min, y_max, rationale}, ...]` — multi-rectangle (union — only when sub-clusters are spatially disconnected; same guardrails as llm_gate);
- `{"vertices": [[x1,y1], [x2,y2], [x3,y3], ...], "rationale": "..."}` — polygon (3+ vertices);
- `{"absent": true, "rationale": "..."}` — declared not present.

The model may pass a *partial* `gates` object during the loop (only the categories it has decided so far). The **final JSON answer** must include every named category from `[Categories]` (except `Unassigned`).

The tool returns:
- `tool` role text: a parsed read-back of every category (exactly as `propagate` would interpret it) + warnings for out-of-data-range boxes and pairwise overlaps.
- `user` role follow-up: an `image_url` block carrying the overlay PNG (data URI). The picture and the text are feedback on the **same** proposal — read both before revising.

---

## Step 4 — `src.agent_gate.postprocess.propagate`: gates → `prediction.json`

Byte-identical to `src.llm_gate.postprocess.propagate`. Same membership rule (closed boundaries; `MplPath.contains_points` for polygons), same `--tiebreak {smallest|priority}`, same perturbation cross-check.

```bash
python -m src.agent_gate.postprocess.propagate \
    --eval_path  results/agent_gpt-5.4_full/Acute2020/pred.json \
    --dataset    Acute2020 \
    --benchmark  benchmark/ --data-dir data/ \
    --method     agent_gpt-5.4_full \
    --output_dir results/agent_gpt-5.4_full
```

---

## Step 5 — `src.eval`

Method-agnostic. Reused as-is from the top-level pipeline.

```bash
python -m src.eval --predictions results/agent_gpt-5.4_full \
    --benchmark benchmark/ --data-dir data/
```

For the `--hard-depletion` / `--hard-shift` magnitude sweeps, use `src.eval_hard` exactly as documented in `src.llm_gate`.

---

## End-to-end recipe (Setting 1 — clean baseline)

```bash
DS=Acute2020
SAMPLE=994588_Normalized
METHOD=agent_gpt-5.4_full

python -m src.agent_gate.task \
    --benchmark benchmark/ --data-dir data/ \
    --datasets $DS --samples $SAMPLE --n-workers 4

python -m src.agent_gate.prompt \
    --batch benchmark/$DS/$SAMPLE --save_md

export OPENAI_API_KEY=sk-...
python -m src.agent_gate.inference.run_agent \
    --dataset_path benchmark/$DS --sample $SAMPLE \
    --model gpt-5.4 --max-turns 4 \
    --output_path results/$METHOD/$DS/pred.json --concurrency 10

python -m src.agent_gate.postprocess.propagate \
    --eval_path results/$METHOD/$DS/pred.json \
    --dataset $DS \
    --benchmark benchmark/ --data-dir data/ \
    --method $METHOD --output_dir results/$METHOD

python -m src.eval --predictions results/$METHOD \
    --benchmark benchmark/ --data-dir data/
```

Or the unified runner:

```bash
SPLIT=test OPENAI_API_KEY=sk-... \
    bash src/agent_gate/scripts/run_agent_gate.sh \
        --config src/agent_gate/configs/models/gpt_5_4.yaml \
        full results/agent_gpt-5.4_full
```

---

## Per-model YAML schema (OpenAI-only)

Two sections. Any field set to `null` (or omitted) is not passed to `run_agent`, so the runner default applies.

```yaml
model:
  openai_id: gpt-5.4               # required
  # served_name: defaults to openai_id

generation:
  max_tokens: 32768                # per-turn cap
  max_turns:  4                    # 1 forced render + up to 2 revisions + 1 text
  # temperature / top_p / presence_penalty / frequency_penalty
  # null → run_agent defaults (T=0.0, others server-side)
```

`model.hf_id` and the `server:` / `server_env:` sections from `src.llm_gate` are not supported — agent-gate is OpenAI-hosted only. For open-source vLLM models, use `src.llm_gate` directly.

Orchestration knobs (NOT in YAML):

| Env | Default | Meaning |
|---|---|---|
| `CONCURRENCY` | `10` | Per-client in-flight loops (lower than llm_gate). |
| `PARALLEL_CLIENTS` | `4` | Datasets in flight at once. |
| `SPLIT` | `(all)` | `test` / `val` / `train` filter via `splits.json`. |
| `BENCHMARK_DIR` | `benchmark` | Benchmark root. |
| `DATA_DIR` | `data` | Raw parquet root. |
| `SKIP_PROPAGATE` | `0` | `1` = stop after `pred.json`. |
| `SKIP_EVAL` | `0` | `1` = stop after `prediction.json`. |
| `TIEBREAK` | `smallest` | `smallest` \| `priority`. |
| `METHOD_TAG` | `agent_<served>_<ablation>` | Recorded in `prediction.json`. |
| `SAVE_RAW` | `0` | `1` adds `--save-raw` to each client. |

---

## File layout

```
src/agent_gate/
├── README.md
├── task.py                      # Step 1 + scatter_bg.png renderer
├── prompt.py                    # Step 2
├── inference/
│   ├── data_loader.py           # surfaces scatter_bg_path + scatter_extent per group
│   ├── output_schema.py         # gate parser + pred.json assembler (shared with llm_gate)
│   ├── system_prompts.py        # agentic system prompt
│   └── run_agent.py             # Step 3 — the tool-call loop
├── tools/
│   ├── render_overlay.py        # the one tool: gates → overlay PNG + read-back
│   └── registry.py              # OpenAI tool schema + dispatch
├── postprocess/
│   └── propagate.py             # Step 4 — gates → prediction.json
├── utils/                       # axis bins, density clouds, joint distribution (copied)
├── configs/models/gpt_5_4.yaml
└── scripts/
    ├── _config_to_shell.py      # YAML → shell
    └── run_agent_gate.sh        # unified runner (1→2→3→4→5)
```

---

## Limitations / non-goals

- **No vLLM / open-source path.** Multimodal tool-call support across open-source vision LMs is uneven; left to a follow-up.
- **No batched tool calls.** Each tool call is its own matplotlib render in a worker thread. Throughput-wise the bottleneck is the chat completions latency, not the renders.
- **No ground-truth peeking.** The overlay is coloured by density, not by GT. The agent sees the same scatter a human gating-software user would see.
- **One tool for now.** The registry is set up so adding a second tool (e.g., `summarise_region` for asking what's in a sub-rectangle) only changes `tools/registry.py` — the runner does not.
