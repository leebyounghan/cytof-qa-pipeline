#!/usr/bin/env bash
# OpenAI-API Setting 1 sweep — Stage 1 (LLM gen) only.
#
# Counterpart to run_setting1_all_models.sh's vllm-served sweep, with the
# same bottleneck-mitigation pattern as run_hard_openai.sh: no vllm
# lifecycle, no inline propagate/eval, per-cohort resume, dry-run support.
#
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline split
# ─────────────────────────────────────────────────────────────────────────────
#   Stage 0  build_all_prereqs.py (or standard preprocess pipeline)
#            → benchmark/<DS>/<sample>/step_NN/{gate_task.json, gate_<ABL>/cells.md}
#   Stage 1  ⟵ THIS SCRIPT  — OpenAI LLM gen → pred.json per cohort.
#            Resumable. `.llm_done` marker when all cohorts land.
#   Stage 2  run_propagate_all.sh  — cell-level predictions (CPU).
#   Stage 3  run_eval_all.sh       — per-step metrics (CPU).
#
# ─────────────────────────────────────────────────────────────────────────────
# Bottlenecks this script avoids
# ─────────────────────────────────────────────────────────────────────────────
#   1. No vllm/GPU lifecycle. Clients hit OpenAI directly via OPENAI_BASE_URL.
#   2. No inline propagate/eval. The API queue is the shared resource and any
#      seconds spent on local CPU is API throughput we never get back.
#   3. Per-cohort resume. Each cohort worker checks `[[ -f $out ]]`; existing
#      pred.json is skipped → re-running the script after a crash only hits
#      OpenAI for the missing cohorts.
#   4. `.llm_done` marker. When all configured cohorts wrote pred.json, mark
#      the model "done" so a downstream multi-model wrapper can skip cleanly.
#   5. PARALLEL_DS × CONCURRENCY in-flight. Default 11 cohorts × 20 = 240
#      in-flight requests; raise to your OpenAI Tier ceiling, lower on 429.
#   6. Pre-flight prereq auto-detect. Counts gate_task.json vs gate_<ABL>/cells.md
#      under each cohort's test samples; logs which are short, lets the user
#      run build_all_prereqs.py before paying for API calls. A missing prereq
#      doesn't block the run (run_openai itself errors clearly), but you'll
#      want to know upfront.
#
# ─────────────────────────────────────────────────────────────────────────────
# Usage
# ─────────────────────────────────────────────────────────────────────────────
#   OPENAI_API_KEY=sk-... \
#       bash run_setting1_openai.sh \
#            --config src/llm_gate/configs/models/gpt_5_4.yaml \
#            [--cohorts ds1 ds2 ...]   # subset of the default 12-cohort list
#            [--dry-run]               # validate orchestration; never call OpenAI
#
# Examples
#   # Default 11 cohorts, GPT-5.4 with the model's yaml-defined sampling.
#   bash run_setting1_openai.sh --config src/llm_gate/configs/models/gpt_5_4.yaml
#
#   # 3 small cohorts only (fast smoke test before paying for the full sweep)
#   PARALLEL_DS=3 \
#       bash run_setting1_openai.sh \
#            --config src/llm_gate/configs/models/gpt_5_4.yaml \
#            --cohorts FR-FCM-Z74D_hc Lyoplate_DC Lyoplate_treg
#
#   # Dry-run — confirms config + sample resolution, never touches OpenAI
#   bash run_setting1_openai.sh \
#       --config src/llm_gate/configs/models/gpt_5_4.yaml --dry-run
#
# ─────────────────────────────────────────────────────────────────────────────
# Env (orchestration only — model id + sampling come from yaml)
# ─────────────────────────────────────────────────────────────────────────────
#   PARALLEL_DS=12          cohorts in flight at once (xargs -P)
#   CONCURRENCY=20          in-flight HTTP per cohort (asyncio inside run_openai)
#   ABLATION=full
#   SPLIT=test
#   ROOT=results/sweep
#   BENCHMARK_DIR=benchmark
#   DATA_DIR=data
#   OPENAI_BASE_URL=        optional override (proxy / Azure / vllm-on-localhost)
#
# Output (same layout as run_setting1_all_models.sh, so Stage 2/3 just work)
#   $ROOT/setting1/<served_name>/<cohort>_<served>_<abl>__<split>.json
#   $ROOT/setting1/<served_name>/.llm_done   (when all cohorts complete)

set -uo pipefail
cd "$(dirname "$0")"

# ── CLI ──────────────────────────────────────────────────────────────────────
CONFIG=""
COHORTS_FILTER=()
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)    CONFIG="${2:?--config requires path}"; shift 2 ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --cohorts)   shift
                     while [[ $# -gt 0 && "$1" != --* ]]; do COHORTS_FILTER+=("$1"); shift; done ;;
        -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
        *)           echo "ERROR: unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ -z "$CONFIG" || ! -f "$CONFIG" ]] && { echo "ERROR: --config <yaml> required" >&2; exit 2; }

# --dry-run bypasses the OPENAI_API_KEY gate; never reaches OpenAI either way.
if [[ $DRY_RUN -eq 1 ]]; then
    : "${OPENAI_API_KEY:=__dry_run__}"
fi

# ── Env (orchestration only — generation comes from yaml) ────────────────────
PARALLEL_DS="${PARALLEL_DS:-12}"
CONCURRENCY="${CONCURRENCY:-20}"
ABLATION="${ABLATION:-full}"
SPLIT="${SPLIT:-test}"
ROOT="${ROOT:-results/sweep}"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark}"
DATA_DIR="${DATA_DIR:-data}"

# Default 12-cohort list (same order as run_setting1_all_models.sh).
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
if [[ ${#COHORTS_FILTER[@]} -gt 0 ]]; then
    DATASETS=("${COHORTS_FILTER[@]}")
fi

# ── Load yaml → MODEL, SERVED_NAME, OPENAI_ARGS[] ────────────────────────────
eval "$(python3 -m src.llm_gate.scripts._config_to_shell --config "$CONFIG")" || {
    echo "ERROR: failed to parse $CONFIG" >&2; exit 2; }
[[ "${OPENAI_MODE:-0}" != "1" ]] && {
    echo "ERROR: $CONFIG is not an OpenAI API config (model.openai_id missing)." >&2; exit 2; }
[[ -z "${OPENAI_API_KEY:-}" ]] && { echo "ERROR: OPENAI_API_KEY not set" >&2; exit 2; }

OUT_DIR="$ROOT/setting1/${SERVED_NAME}"
LOG_DIR="$ROOT/_logs/setting1_openai_${SERVED_NAME}"
mkdir -p "$OUT_DIR" "$LOG_DIR" /tmp/c2s_logs
MAIN_LOG="$LOG_DIR/main.log"
LLM_DONE_MARKER="$OUT_DIR/.llm_done"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }

# Pre-flight: whole-model skip if .llm_done already exists (and not dry-run).
if [[ -f "$LLM_DONE_MARKER" && $DRY_RUN -eq 0 ]]; then
    log "[skip-all] $SERVED_NAME — .llm_done already present ($LLM_DONE_MARKER)"
    log "  remove the marker to force re-run; otherwise this is a no-op."
    exit 0
fi

log "================================================================"
log "OpenAI Setting 1 sweep — Stage 1 (LLM gen)"
log "  served_name : $SERVED_NAME"
log "  cohorts     : ${#DATASETS[@]}   (${DATASETS[*]})"
log "  parallel    : $PARALLEL_DS cohorts × concurrency=$CONCURRENCY (= $((PARALLEL_DS*CONCURRENCY)) in-flight)"
log "  ablation    : $ABLATION    split: $SPLIT"
log "  output      : $OUT_DIR"
log "  dry-run     : $([[ $DRY_RUN -eq 1 ]] && echo YES || echo no)"
log "================================================================"

# ── Pre-flight prereq audit (cheap; reports per-cohort prereq health) ───────
# Setting 1 prereqs are usually built once via build_all_prereqs.py and
# remain stable. We don't auto-build here (Stage 0 is a separate concern);
# we just *report* if cells.md is missing for any test sample so the user
# can build before paying for API calls.
audit_prereqs() {
    local ds=$1
    local n_test n_md
    n_test=$(python3 -m src.llm.scripts.list_split_samples \
                --splits "$BENCHMARK_DIR/splits.json" --dataset "$ds" --split "$SPLIT" \
                2>/dev/null | wc -l)
    # cells.md count under gate_<ABL>/ for this cohort. We don't filter by
    # split in the find (cells.md exist for all samples, not just test);
    # the comparison is "do we have at least one cells.md per test sample's
    # step_NN folder" — we approximate with total cells.md count.
    n_md=$(find "$BENCHMARK_DIR/$ds" -maxdepth 4 -type f \
                -path "*gate_${ABLATION}/cells.md" 2>/dev/null | wc -l)
    echo "$n_test $n_md"
}

log "Prereq audit (test samples × steps vs gate_${ABLATION}/cells.md count):"
SHORT=0
for ds in "${DATASETS[@]}"; do
    read -r n_test n_md <<< "$(audit_prereqs "$ds")"
    # We can't trivially know n_steps per sample without parsing task.json.
    # Use n_md > 0 + n_test > 0 as a sanity floor; deeper validation belongs
    # to build_all_prereqs.py. Flag obvious mis-builds.
    if [[ $n_test -eq 0 ]]; then
        log "  [skip-cohort] $ds — no $SPLIT samples"
    elif [[ $n_md -lt $n_test ]]; then
        log "  [WARN] $ds — only $n_md cells.md present, $n_test test samples (run build_all_prereqs.py?)"
        SHORT=$((SHORT+1))
    else
        log "  [ok]   $ds — $n_md cells.md, $n_test test samples"
    fi
done
if (( SHORT > 0 )); then
    log "  ($SHORT cohorts under-provisioned; run_openai will error out on those.)"
fi

# ── Worker (one cohort) ──────────────────────────────────────────────────────
run_one_cohort() {
    local ds=$1
    local out="$OUT_DIR/${ds}_${SERVED_NAME}_${ABLATION}__${SPLIT}.json"
    if [[ -f "$out" ]]; then
        log "  [skip]  $ds — pred.json exists"
        return 0
    fi

    local samples
    if ! mapfile -t samples < <(
        python3 -m src.llm.scripts.list_split_samples \
            --splits "$BENCHMARK_DIR/splits.json" --dataset "$ds" --split "$SPLIT" \
            2> "/tmp/c2s_logs/openai_split_${ds}.log"
    ); then
        log "  [ERROR] $ds — list_split_samples failed"
        return 1
    fi
    if [[ ${#samples[@]} -eq 0 ]]; then
        log "  [skip]  $ds — no $SPLIT samples"
        return 0
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] $ds (n=${#samples[@]}) — would write $out"
        return 0
    fi

    log "  [start] $ds (n=${#samples[@]})"
    python3 -m src.llm_gate.inference.run_openai \
        --dataset_path "$BENCHMARK_DIR/$ds" \
        --model "$SERVED_NAME" \
        --ablation_slug "$ABLATION" \
        --concurrency "$CONCURRENCY" \
        --output_path "$out" \
        --sample "${samples[@]}" \
        --save-raw \
        "${OPENAI_ARGS[@]}" \
        > "$LOG_DIR/${ds}_client.log" 2>&1
    local rc=$?
    log "  [done]  $ds (rc=$rc)"
    return $rc
}

# ── xargs wrapper: rebuild OPENAI_ARGS array from exported string ────────────
xargs_worker() {
    local ds=$1
    # shellcheck disable=SC2206
    OPENAI_ARGS=($OPENAI_ARGS_STR)
    run_one_cohort "$ds"
}

export -f log run_one_cohort xargs_worker
export SERVED_NAME ABLATION CONCURRENCY ROOT OUT_DIR LOG_DIR MAIN_LOG \
       BENCHMARK_DIR DATA_DIR SPLIT DRY_RUN
OPENAI_ARGS_STR="${OPENAI_ARGS[*]}"
export OPENAI_ARGS_STR

# ── Run ──────────────────────────────────────────────────────────────────────
T0=$(date +%s)
printf '%s\n' "${DATASETS[@]}" | xargs -P "$PARALLEL_DS" -I {} bash -c 'xargs_worker "$@"' _ "{}"
T1=$(date +%s)

# ── Summary + .llm_done marker ───────────────────────────────────────────────
DONE=0; MISS=0
for ds in "${DATASETS[@]}"; do
    out="$OUT_DIR/${ds}_${SERVED_NAME}_${ABLATION}__${SPLIT}.json"
    [[ -f "$out" ]] && DONE=$((DONE+1)) || MISS=$((MISS+1))
done

log "================================================================"
if [[ $DRY_RUN -eq 1 ]]; then
    log "Stage 1 dry-run finished in $((T1-T0))s — ${#DATASETS[@]} cohorts validated, NO pred.json written"
    log "  (exit 0; orchestration only — OpenAI was never contacted)"
    log "================================================================"
    exit 0
fi
log "Stage 1 finished in $((T1-T0))s — done=$DONE  missing=$MISS  total=${#DATASETS[@]}"
if (( DONE == ${#DATASETS[@]} )); then
    touch "$LLM_DONE_MARKER"
    log "  .llm_done marked: $LLM_DONE_MARKER"
fi
log "Next:"
log "  Stage 2:  bash run_propagate_all.sh   ($SERVED_NAME)"
log "  Stage 3:  bash run_eval_all.sh        ($SERVED_NAME)"
log "================================================================"
exit $MISS
