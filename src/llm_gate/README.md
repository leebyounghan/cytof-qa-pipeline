# LLM-Gate Pipeline (`src.llm_gate`)

An alternative LLM baseline that asks the model to draw **axis-aligned rectangular gates** (or polygons) directly on the `(x_marker, y_marker)` plane — one gate per named category, per gating step. Mirrors how an expert places boxes on a biaxial plot. Independent of the C2S pipeline.

```
parquet → gate_task.json → gate_{ablation}/cells.md → pred.json → prediction.json → eval_summary.json
            (Step 1)            (Step 2)               (Step 3)      (Step 4)         (Step 5)
```

## vs C2S

| | `src.llm.c2s` | `src.llm_gate` |
|---|---|---|
| LLM input | meta-cluster cell sentences | per-axis 1-D bins (+ optional 2-D joint) |
| LLM output | `cluster_id → category` | `category → {x_min, x_max, y_min, y_max, rationale}` |
| Propagation | cluster label → cell label via `cell2cluster.npz` | gate box → cell membership on raw `(x, y)` |
| Tie-break | n/a | smallest-area / `options` priority |
| Polygon support | n/a | yes — `{vertices, rationale}` |

## Step 1 — `src.llm_gate.task`: parquet → `gate_task.json`

For every (sample, step) under `--datasets` / `--samples`, loads the parent population and writes `step_NN/gate_task.json`:

- `step`, `parent`, `x_marker`, `y_marker`, `note`, `tips`, `options`, `n_parent_cells`, `modality`, `cofactor`
- `x_summary`, `y_summary` — per-axis min/max + q05/25/50/75/95
- `axis_distribution` — per-axis 1-D 30-bin histogram
- `joint_distribution` — 2-D peak/valley info on the joint heatmap
- `density_clouds` — watershed-segmented density clouds
- `gt_category_counts` — diagnostic only (never rendered to the LLM)

```bash
python -m src.llm_gate.task \
    --benchmark benchmark/ --data-dir data/ --n-workers 8

# Test split only
python -m src.llm_gate.task \
    --benchmark benchmark/ --data-dir data/ \
    --split benchmark/splits.json --split-set test
```

Key flags: `--modality {auto|cytof|flow}` (default `auto` — detected from the cohort), `--n-workers N`, `--skip-existing`, `--perturbation` / `--hard-depletion` / `--hard-shift`. `--datasets` / `--samples` / `--split` and other loader flags inherit from `BenchmarkLoader.add_cli_args`.

## Step 2 — `src.llm_gate.prompt`: `gate_task.json` → `gate_{ablation}/cells.md`

One Markdown prompt per step. Output goes to a sibling directory of `gate_task.json` (`gate_full/cells.md`).

The prompt has five sections: `[Context]`, `[Categories]`, `[Description]` (note + Unassigned definition), `[Axis distribution]` (1-D bins; 2-D joint heatmap is opt-in), and `## Output Format` (per-category JSON schema example).

```bash
# Single step
python -m src.llm_gate.prompt \
    benchmark/Acute2020/994588_Normalized/step_09/gate_task.json --save_md

# Whole sample (default — 1-D bins only, note included)
python -m src.llm_gate.prompt \
    --batch benchmark/Acute2020/994588_Normalized --save_md

# Add the 2-D joint heatmap
python -m src.llm_gate.prompt \
    --batch benchmark/Acute2020/994588_Normalized --save_md --with-joint-2d

# Drop the gating-plan note from [Description]
python -m src.llm_gate.prompt \
    --batch benchmark/Acute2020/994588_Normalized --save_md --without-note
```

Flags:
- `--save_md` — write to file. Off → stdout.
- `--without-note` — exclude the gating-plan `note` from `[Description]`.
- `--with-joint-2d` — append the 2-D joint heatmap block after the 1-D bin lists.
- `--perturbation` / `--hard-depletion` / `--hard-shift` — match the spec used in Step 1; reads `gate_task__{slug}.json`, writes to `gate_{ablation}__{slug}/`.

Prompt ablations are selected with `--ablation {full,no_desc,no_hist,no_pv,anon}` and render into slug-named sibling directories (`gate_no_pv/cells.md`, ...), so variants coexist per step: `no_desc` drops `[Description]`, `no_hist` drops the whole `[Axis distribution]` section (histogram *and* peak/valley lines), `no_pv` keeps the histogram but strips the peak/valley (`P*`/`V*`) lines, `anon` anonymizes marker names. Per-(cohort, step) note overrides come from `benchmark/notes.yaml` (disable with `--without-notes-yaml`). `--with-joint-2d` overwrites the same `cells.md` in place, so for that comparison save runs under separate `results/` method names.

### `cells.md` layout (default — 1-D bins only)

```
[Context]
Modality: CyTOF (arcsinh cofactor: 5)
Path: Step01_CD4|CD8 == CD8 → [CCR7 vs CD45RA]
Step: 2
X-axis marker: CCR7
Y-axis marker: CD45RA
Parent cells: 63,891

[Categories]
Naive_CD8, CM_CD8, EM_CD8, EMRA_CD8, Unassigned

[Description]
CD8 memory layout on CCR7 × CD45RA. Strategy: ...
Unassigned: cells whose (x, y) position falls outside every named category's gate at this step.

---

[Axis distribution]
### CCR7 (x-axis), range [-1.23, 6.45], n=63891 parent cells
counts (30 bins, lo→hi): [...]
bin_centers:             [...]

### CD45RA (y-axis), range [0.00, 7.81], n=63891 parent cells
counts (30 bins, lo→hi): [...]
bin_centers:             [...]

---

## Output Format
Return a JSON object keyed by category name. ...
```

## Step 3 — `src.llm_gate.inference.run_openai`: prompts → `pred.json`

Async OpenAI runner. Walks `--dataset_path` for (sample, step) pairs, loads each `gate_full/cells.md` together with its `gate_task.json` (for options + axis labels), fires one chat request per step (`temperature=0.0`), and parses the per-category JSON gate.

```bash
export OPENAI_API_KEY=sk-...

python -m src.llm_gate.inference.run_openai \
    --dataset_path benchmark/Acute2020 \
    --model        gpt-5.4 \
    --output_path  results/gate_gpt-5.4_full/Acute2020/pred.json \
    --concurrency  20
```

Key flags: `--ablation_slug` (default `full`, reads `gate_{slug}/cells.md`), `--sample`, `--max_samples N --seed`, `--concurrency 20`, `--max_tokens 8192`, `--perturbation` / `--hard-depletion` / `--hard-shift` (recorded in `pred.json.meta.perturbation`).

### `pred.json` structure

```json
{
  "meta": {"model": "gpt-5.4", "ablation": "full", "dataset_path": "...",
           "n_samples": 4, "n_steps": 49, "perturbation": null},
  "samples": {
    "994588_Normalized": {
      "steps": {
        "1": {
          "x_marker": "Time", "y_marker": "Bead",
          "options": ["Cleanup1", "Bead", "Unassigned"],
          "gates": {
            "Cleanup1": {"x_min": 0.0, "x_max": 10.0,
                         "y_min": 0.0, "y_max": 4.5, "rationale": "..."},
            "Bead":     {"x_min": 0.0, "x_max": 10.0,
                         "y_min": 6.5, "y_max": 8.5, "rationale": "..."}
          }
        }
      }
    }
  }
}
```

Categories without a valid gate (parse error / missing entry / degenerate box) are stored as `null` and contribute no membership at propagation time.

### Absent categories

When the LLM judges that a named category is genuinely not present in this step's parent population (zero expected mass, no recognizable mode in the histogram), it may emit an absent sentinel instead of a gate. The key is the actual category name from the step's `options` list — the same name that would otherwise carry a gate dict. Example for a CD8 memory step where the LLM judges `EMRA_CD8` to be absent:

```json
{
  "Naive_CD8": {"x_min": 0.2, "x_max": 1.8, "y_min": 4.0, "y_max": 6.5, "rationale": "..."},
  "CM_CD8":    {"x_min": 0.2, "x_max": 1.8, "y_min": 0.0, "y_max": 4.0, "rationale": "..."},
  "EM_CD8":    {"x_min": 1.8, "x_max": 5.5, "y_min": 0.0, "y_max": 4.0, "rationale": "..."},
  "EMRA_CD8":  {"absent": true, "rationale": "no mass on the expected CD45RA+ shoulder of the CCR7- side"}
}
```

Stored verbatim as `{"absent": true, "rationale": "..."}` in `pred.json`. Cells contribute no membership — same runtime effect as `null` — but the propagation step counts these separately (`info.n_gates_absent`, `info.categories_absent`) so intentional absence is distinguishable from parse failure. The system prompt restricts `absent` to genuinely-missing populations; sparse-but-present clusters still get a tight gate.

### Polygon gates

A category may instead return a free polygon (3+ vertices), accepted by `extract_gates`:

```json
{"Naive_CD8": {"vertices": [[1.2, 3.4], [5.6, 3.4], [5.6, 6.8], [1.2, 6.8]],
               "rationale": "..."}}
```

The system prompt asks for axis-aligned boxes; polygon support is a lenient parsing path for models that prefer rotated rectangles or arbitrary shapes. Membership uses `matplotlib.path.Path.contains_points` with closed boundaries.

## Step 4 — `src.llm_gate.postprocess.propagate`: gates → `prediction.json`

For every (sample, step) in `pred.json`, rebuilds the parent population through `BenchmarkLoader`, applies each category's gate to the parent cells' `(x, y)`, resolves overlaps, and writes one `prediction.json` per step (`src.eval` compatible).

```bash
python -m src.llm_gate.postprocess.propagate \
    --eval_path  results/gate_gpt-5.4_full/Acute2020/pred.json \
    --dataset    Acute2020 \
    --benchmark  benchmark/ --data-dir data/ \
    --method     gate_gpt-5.4_full \
    --output_dir results/gate_gpt-5.4_full
```

`--eval_path` / `--dataset` are repeatable for multi-dataset runs. Key flags:
- `--tiebreak {smallest|priority}` (default `smallest`) — overlap resolution.
- `--perturbation` / `--hard-depletion` / `--hard-shift` — must match `pred.json.meta.perturbation`; mismatch fails fast.

**Membership rule.** A cell at `(x, y)` falls inside a gate when `x_min ≤ x ≤ x_max & y_min ≤ y ≤ y_max` (boundaries closed); polygon gates use `MplPath.contains_points(radius=1e-9)` so edge cells are included. Cells outside every named gate become `Unassigned`.

**Tie-break.** Gates are applied in *application order* — the last gate to write a cell wins. With `--tiebreak smallest`, gates are sorted largest-area first, so the smallest (most specific) gate overwrites larger ones in the intersection. With `--tiebreak priority`, gates are applied in reverse `options` order so the first listed category wins. The system prompt explicitly tells the LLM to avoid overlaps; tie-break is the fallback.

## Step 5 — `src.eval`

Method-agnostic. See [Evaluation Protocol in the top-level README](../../README.md#evaluation-protocol).

```bash
python -m src.eval --predictions results/gate_gpt-5.4_full \
    --benchmark benchmark/ --data-dir data/
```

## Setting 2: In-Panel Hard

Setting 2 = same panel, perturbed parent population. Two axes are pre-curated as benchmark spec — see the [main README](../../README.md#setting-2-in-panel-hard) for the scenario tables and physical rationale.

- **`--hard-depletion <hard_id>`** — pulls a (dataset, step, depletion-target) triple from `benchmark/hard_depletion.yaml` (43 ids × 9 clinical scenarios: HIV, rituximab, neutropenia, …). Removes the named GT cells from the target step's parent population. Magnitude knob: `--hard-depletion-fraction <F>` (default 1.0 = full removal, 0.0 = baseline).
- **`--hard-shift <hard_id>`** — pulls a (dataset, step, channel-scale) triple from `benchmark/hard_shift.yaml` (52 ids × 9 calibration scenarios: lineage_dim, ccr7_internalization, panel_global, …). Multiplicatively rescales raw (pre-arcsinh) signal on the listed markers; GT labels are unchanged. Magnitude knob: `--hard-shift-scale <S>` (default 1.0, 0.0 = baseline).

Both flags are mutually exclusive with each other and with `--perturbation` / `--datasets`. They must be passed **identically** to every stage (`task` → `prompt` → `run_openai` → `propagate`); the slug round-trip is what cross-checks runs at eval time.

### Hard depletion — 0% / 90% / 99% magnitude sweep

The hard-depletion scenario is interesting at the **edges of the fraction sweep** — the question is whether the model still hallucinates the depleted population once almost-but-not-quite all of it is gone. Recommended comparison: `0` (baseline, no removal — the model should label normally), `0.9` (90% removed — sparse-but-present, the trickiest regime), `0.99` (99% removed — essentially absent, model should ideally emit `{"absent": true}`).

```bash
HID=hiv_acute2020_step29     # one curated hard depletion
METHOD=gate_gpt-5.4_depletion

for FRAC in 0 0.9 0.99; do
    PCT=$(python -c "print(int(round($FRAC * 100)))")     # 0, 90, 99
    SLUG_DIR=frac${PCT}                                   # frac0 / frac90 / frac99

    # Step 1 — task
    python -m src.llm_gate.task \
        --benchmark benchmark/ --data-dir data/ \
        --hard-depletion $HID --hard-depletion-fraction $FRAC

    # Step 2 — prompt (--batch takes one SAMPLE directory; loop the cohort's samples)
    DS=$(python -c "from src.hard_depletions import get_hard_depletion as g; print(g('$HID').dataset)")
    for s in benchmark/$DS/*/; do
        python -m src.llm_gate.prompt --batch "$s" --save_md \
            --hard-depletion $HID --hard-depletion-fraction $FRAC
    done

    # Step 3 — inference (OUT_DIR layout: results/<method>/<scenario>/<ds>_step<NN>/<frac{NN}>/<ds>/pred.json)
    SCENARIO=$(python -c "from src.hard_depletions import get_hard_depletion as g; print(g('$HID').scenario)")
    DS=$(python -c "from src.hard_depletions import get_hard_depletion as g; print(g('$HID').dataset)")
    STEP=$(python -c "from src.hard_depletions import get_hard_depletion as g; print(f'{g(\"$HID\").step:02d}')")
    OUT_ROOT=results/$METHOD/$SCENARIO/${DS}_step${STEP}/$SLUG_DIR

    python -m src.llm_gate.inference.run_openai \
        --dataset_path benchmark/$DS \
        --hard-depletion $HID --hard-depletion-fraction $FRAC \
        --model gpt-5.4 \
        --output_path $OUT_ROOT/$DS/pred.json --concurrency 20

    # Step 4 — propagate
    python -m src.llm_gate.postprocess.propagate \
        --eval_path $OUT_ROOT/$DS/pred.json \
        --dataset $DS \
        --benchmark benchmark/ --data-dir data/ \
        --hard-depletion $HID --hard-depletion-fraction $FRAC \
        --method $METHOD --output_dir $OUT_ROOT
done

# Step 5 — magnitude-aware eval (auto-detects depletion mode from frac{NN} dirs)
python -m src.eval_hard \
    --root results/$METHOD \
    --magnitudes 0 0.9 0.99 \
    --plot-dir results/$METHOD/plots
```

`src.eval_hard` parses each `frac{NN}` subdirectory, re-loads GT at the matching fraction, computes per-magnitude metrics, and (with `--plot-dir`) renders a 1×3 side-by-side scatter per (hard_id, sample) — baseline | 90% | 99% — with the LLM's emitted gates overlaid as dashed boxes. The expected reading: F1 stable at `frac0`, sharpest drop at `frac90` (sparse populations seduce naive box gates), and (for a well-behaved model) recovery at `frac99` via the absent-sentinel path.

For hard shift, swap `--hard-depletion`/`--hard-depletion-fraction` for `--hard-shift`/`--hard-shift-scale` and use `scale{NN}` directories; `src.eval_hard` auto-detects the mode.

## End-to-end recipe (Setting 1 — clean baseline)

```bash
DS=Acute2020
SAMPLE=994588_Normalized
METHOD=gate_gpt-5.4_full

python -m src.llm_gate.task \
    --benchmark benchmark/ --data-dir data/ \
    --datasets $DS --samples $SAMPLE --n-workers 4

python -m src.llm_gate.prompt \
    --batch benchmark/$DS/$SAMPLE --save_md

export OPENAI_API_KEY=sk-...
python -m src.llm_gate.inference.run_openai \
    --dataset_path benchmark/$DS --sample $SAMPLE \
    --model gpt-5.4 \
    --output_path results/$METHOD/$DS/pred.json --concurrency 20

python -m src.llm_gate.postprocess.propagate \
    --eval_path results/$METHOD/$DS/pred.json \
    --dataset $DS \
    --benchmark benchmark/ --data-dir data/ \
    --method $METHOD --output_dir results/$METHOD

python -m src.eval --predictions results/$METHOD \
    --benchmark benchmark/ --data-dir data/
```

For Setting 2 (In-Panel Hard), see the section above — pass the same `--hard-depletion <HID>` / `--hard-shift <HID>` (with optional `--hard-depletion-fraction` / `--hard-shift-scale`) to every stage and evaluate with `src.eval_hard` instead of `src.eval`.

### Open-source vLLM runner (YAML config)

Replaces the manual Step 3 / 4 / 5 chain when running an open-source
model via local `vllm serve` (Qwen, Gemma, …). One YAML per model
holds the full server + generation spec; a unified runner brings up
the server, fans datasets out as concurrent OpenAI-compatible clients
sharing the same engine, then runs propagate + eval.

```bash
# Qwen3.6-27B (dense), test split, full ablation
SPLIT=test bash src/llm_gate/scripts/run_vllm_gate.sh \
    --config src/llm_gate/configs/models/qwen3_6_27b_tb8096.yaml \
    full output/eval_gate_qwen3_6_27b

# Gemma 4 26B-A4B (MoE) — same runner, different YAML
SPLIT=test bash src/llm_gate/scripts/run_vllm_gate.sh \
    --config src/llm_gate/configs/models/gemma_4_26b_a4b_it.yaml \
    full output/eval_gate_gemma_4_26b_a4b

# Gemma 4 31B, subset of datasets
SPLIT=test bash src/llm_gate/scripts/run_vllm_gate.sh \
    --config src/llm_gate/configs/models/gemma_4_31b_it.yaml \
    full output/eval_gate_gemma_4_31b \
    Acute2020 Bjornson
```

Layout:

```
src/llm_gate/
├── configs/models/
│   ├── qwen3_6_27b_tb8096.yaml     # dense reasoning (tb8096 = paper setting)
│   ├── gemma_4_26b_a4b_it.yaml     # MoE (--enable-expert-parallel)
│   └── gemma_4_31b_it.yaml         # dense
└── scripts/
    ├── _config_to_shell.py      # YAML → shell-array converter (shlex-safe)
    └── run_vllm_gate.sh         # unified runner (positional: ABLATION OUT_DIR DATASETS…)
```

**Per-model YAML schema.** Four sections. Any field set to `null` (or
omitted) is *not* passed to `vllm`/`run_openai`, so the underlying
default applies — this is how e.g. an unset `temperature` falls
through to `run_openai`'s default rather than being passed as a
literal `"None"`.

```yaml
model:
  hf_id: Qwen/Qwen3.6-27B            # required: HF id or local path
  served_name: Qwen3.6-27B           # optional; defaults to basename(hf_id)

server:                              # → vllm serve flags
  data_parallel_size: 4
  tensor_parallel_size: 2
  max_model_len: 32768
  gpu_memory_utilization: 0.90
  enable_prefix_caching: true
  trust_remote_code: true
  enable_expert_parallel: false
  reasoning_parser: qwen3            # null = no flag
  kv_cache_dtype: null               # e.g. "fp8" for DeepSeek-V4
  block_size: null                   # e.g. 256
  tokenizer_mode: null               # e.g. "deepseek_v4"
  compilation_config: null           # YAML mapping → serialized as one JSON arg

server_env:                          # exported before `vllm serve` (optional)
  VLLM_DISABLE_COMPILE_CACHE: 1
  TILELANG_CLEANUP_TEMP_FILES: 1

generation:                          # → run_openai flags
  max_tokens: 20000
  temperature: null                  # null = run_openai default (0.0)
  top_p: null
  top_k: null
  min_p: null
  presence_penalty: null
  frequency_penalty: null
  repetition_penalty: null
  thinking: null                     # on | off | null
  reasoning_effort: null             # high | max | null  (DeepSeek)
  strip_think: true                  # robust JSON parser: slice past
                                     # </think>, prefer the last fenced
                                     # block over earlier drafts.
                                     # Default off; turn ON for any
                                     # open-source thinking model.
```

Orchestration knobs (NOT in YAML — they vary per experiment, set via env):

| Env | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | vLLM serve port |
| `CONCURRENCY` | `64` | Per-client in-flight requests |
| `PARALLEL_CLIENTS` | `4` | Datasets in flight at once |
| `READY_TIMEOUT` | `900` | Seconds to wait for server `/health` |
| `SPLIT` | `(all)` | `test` / `val` / `train` filter via `splits.json` |
| `BENCHMARK_DIR` | `benchmark` | Benchmark root |
| `DATA_DIR` | `data` | Raw parquet root |
| `SKIP_PROPAGATE` | `0` | `1` = stop after `pred.json` |
| `SKIP_EVAL` | `0` | `1` = stop after `prediction.json` |
| `TIEBREAK` | `smallest` | `smallest` \| `priority` |
| `METHOD_TAG` | `gate_<served>_<ablation>` | Recorded in `prediction.json` |
| `SAVE_RAW` | `0` | `1` adds `--save-raw` to each client |
| `THINKING` | — | If set, overrides YAML `generation.thinking` per run |

**Adding a new model.** Drop a new `configs/models/<name>.yaml` — no
script edits required. Copy one of the existing files as a template;
keep only the keys your model needs and leave the rest unset/`null`.

**Serve-only mode.** For interactive debugging (no clients, no eval):

```bash
bash src/llm_gate/scripts/run_vllm_gate.sh \
    --config src/llm_gate/configs/models/gemma_4_31b_it.yaml \
    --serve-only
```

### OpenAI API runner (YAML config)

Same YAML schema, sibling runner — for hosted OpenAI models (GPT-5.4
etc.) where there's no local server lifecycle. The YAML uses
`model.openai_id` instead of `model.hf_id` and omits the `server` /
`server_env` sections; the runner skips Phase 1 (vllm serve +
health-check) and fires clients straight at `api.openai.com` using the
ambient `OPENAI_API_KEY`.

```bash
# GPT-5.4, hvp10 ablation, test split
SPLIT=test OPENAI_API_KEY=sk-... \
    bash src/llm_gate/scripts/run_openai_gate.sh \
    --config src/llm_gate/configs/models/gpt_5_4.yaml \
    hvp10 results/gate_gpt-5.4_hvp10
```

YAML (`gpt_5_4.yaml`):

```yaml
model:
  openai_id: gpt-5.4               # → run_openai --model gpt-5.4
  # served_name: defaults to "gpt-5.4"

generation:
  max_tokens: 8192
  # temperature/top_p/etc null → run_openai defaults (T=0.0)
  # strip_think: false (GPT-5.4 single-shot)
```

Differences from `run_vllm_gate.sh`:

| | vLLM runner | OpenAI runner |
|---|---|---|
| YAML key  | `model.hf_id`        | `model.openai_id` |
| Phase 1   | `vllm serve` + wait  | (skipped) |
| Default `CONCURRENCY` | `64` | `20` (rate-limit safer) |
| `OPENAI_BASE_URL` | overridden to `localhost:$PORT` | left unset (SDK default) |
| `OPENAI_API_KEY`  | hard-coded `EMPTY`              | required from environment |
| `--serve-only`    | supported            | not applicable (errors) |

The two runners share `_config_to_shell.py` for YAML loading. Passing
an `openai_id` config to `run_vllm_gate.sh` (or vice versa) errors out
early with a clear message rather than silently misbehaving.

