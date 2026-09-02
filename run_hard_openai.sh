#!/usr/bin/env bash
# OpenAI-API hard sweep (Setting 2) — Stage 1 (LLM gen) only.
#
# OpenAI-API counterpart of the vLLM Setting-2 path. The vLLM runner does
# bring-up-vllm → per-HID (task.py + prompt.py + run_openai + propagate) → tear-down,
# all inline. Inlining propagate is fine when the GPU is the bottleneck (vllm
# saturates a single H200 anyway), but it kills throughput when clients hit a
# remote OpenAI endpoint: every minute of CPU propagate is a minute of API
# slots sitting idle. So this script is a deliberate fork of that pattern,
# tuned for the API-bound case.
#
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline split (each stage = independent script; no stage starves another)
# ─────────────────────────────────────────────────────────────────────────────
#   Stage 0  build_all_prereqs.py / build_depletion_magnitude_prereqs.py
#            (one-time, 96-way ProcessPoolExecutor) → cells.md per (HID, frac).
#   Stage 1  ⟵ THIS SCRIPT  — OpenAI LLM gen → pred.json. Resumable per job.
#   Stage 2  run_propagate_all.sh  — cell-level predictions (CPU).
#   Stage 3  run_eval_all.sh       — per-step metrics (CPU).
#
# ─────────────────────────────────────────────────────────────────────────────
# Bottlenecks this script avoids
# ─────────────────────────────────────────────────────────────────────────────
#   1. No vllm/GPU lifecycle. Clients hit OpenAI directly via OPENAI_BASE_URL.
#      Removes ~80s warm-up + tear-down per model that sweep_overnight needs.
#
#   2. No inline propagate/eval. Critical for OpenAI: the API queue is the
#      shared resource, and any seconds spent on local CPU is API throughput
#      we never get back. Stage 2/3 run separately AFTER all LLM jobs land.
#
#   3. Per-(HID, fraction, cohort) resume. Each worker checks `[[ -f $out ]]`
#      and returns 0 if pred.json already exists. Crash mid-sweep → just
#      re-run the script; only the missing jobs hit the API again. This is
#      the same pattern that turned today's Qwen3.5-4B re-run into a no-op
#      for the 3 cohorts that had already succeeded in the broken sweep.
#
#   4. Prereqs idempotent + auto-skip (per-spot, handles partial). Decision
#      matrix vs n_expected = #samples in $SPLIT for this HID's dataset:
#        --no-prereq                          → skip task + prompt (explicit).
#        n_task ≥ expected, n_md ≥ expected   → skip task + prompt (fully built).
#                                               HID cost = 2 × find ≈ 50 ms.
#        n_task ≥ expected, n_md  < expected  → skip task.py, run prompt.py
#                                               (task.py with --skip-existing
#                                                would just re-skip everything,
#                                                burning ~3-4 s on Python +
#                                                BenchmarkLoader init for nothing).
#        else                                  → run both. task.py uses
#                                                --skip-existing so already-built
#                                                tasks aren't redone; prompt.py
#                                                is naturally idempotent.
#      Net: a fully-prebuilt 138-job magnitude sweep validates in seconds
#      instead of minutes, while partial states still get correctly filled.
#
#   5. PARALLEL_HIDS × CONCURRENCY. xargs spawns PARALLEL_HIDS workers; each
#      worker's run_openai uses asyncio with CONCURRENCY in-flight HTTPs.
#      Default 8 × 20 = 160 in-flight, well within OpenAI Tier-3 RPM. Raise
#      both for higher tiers; lower if you start seeing 429s.
#
# ─────────────────────────────────────────────────────────────────────────────
# Usage
# ─────────────────────────────────────────────────────────────────────────────
#   OPENAI_API_KEY=sk-... \
#       bash run_hard_openai.sh \
#            --config src/llm_gate/configs/models/gpt_5_4.yaml \
#            --mode   depletion       # depletion | shift | depletion-magnitude | all
#            [--hids  hid1 hid2 ...]  # optional subset (bare HIDs OR HID_fracNN slugs)
#            [--no-prereq]            # skip task+prompt build (Stage 0 already ran)
#
# Examples
#   # Full hard depletion + shift (95 HIDs) for GPT-5.4
#   bash run_hard_openai.sh --config src/llm_gate/configs/models/gpt_5_4.yaml --mode all
#
#   # Magnitude sweep (138 (HID, frac) combos), prereqs already built
#   PARALLEL_HIDS=12 CONCURRENCY=24 \
#       bash run_hard_openai.sh \
#            --config src/llm_gate/configs/models/gpt_5_4.yaml \
#            --mode depletion-magnitude --no-prereq
#
#   # Re-run two specific HIDs only
#   bash run_hard_openai.sh \
#       --config src/llm_gate/configs/models/gpt_5_4.yaml \
#       --mode depletion \
#       --hids b_cell_aplasia_acute2020_step12 hiv_acute2021_step08
#
# ─────────────────────────────────────────────────────────────────────────────
# Env (orchestration only — model id + sampling come from the yaml config)
# ─────────────────────────────────────────────────────────────────────────────
#   PARALLEL_HIDS=8        HIDs in flight at once (xargs -P)
#   CONCURRENCY=20         in-flight HTTP per client (asyncio inside run_openai)
#   ABLATION=full
#   SPLIT=test
#   FRACS="0.0 0.9 0.99"   only used by depletion-magnitude
#   ROOT=results/sweep
#   BENCHMARK_DIR=benchmark
#   DATA_DIR=data
#   OPENAI_BASE_URL=       optional override (proxy / Azure / vllm-on-localhost)
#
# Output (same layout as the vLLM sweep, so Stage 2/3 just work)
#   $ROOT/setting2_<mode>/<served_name>/<hid_or_hid_fracNN>/<cohort>_<served>_<abl>.json
#
# ─────────────────────────────────────────────────────────────────────────────
# Implementation note: passing OPENAI_ARGS through xargs
# ─────────────────────────────────────────────────────────────────────────────
# Bash arrays cannot be exported, so we serialize OPENAI_ARGS into a single
# space-separated string (OPENAI_ARGS_STR) and re-split inside the xargs
# worker via word-splitting. Safe because every token from _config_to_shell.py
# is whitespace-free (--max_tokens, 32768, --temperature, 1.0, ...). If a
# yaml ever introduces a value with embedded whitespace this contract breaks
# — switch to `printf '%s\0'` + xargs -0 at that point.

set -uo pipefail
cd "$(dirname "$0")"

# ── CLI ──────────────────────────────────────────────────────────────────────
CONFIG=""
MODE=""
HIDS_FILTER=()
NO_PREREQ=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     CONFIG="${2:?--config requires path}"; shift 2 ;;
        --mode)       MODE="${2:?--mode requires value}";    shift 2 ;;
        --no-prereq)  NO_PREREQ=1; shift ;;
        --dry-run)    DRY_RUN=1;   shift ;;
        --hids)       shift
                      while [[ $# -gt 0 && "$1" != --* ]]; do HIDS_FILTER+=("$1"); shift; done ;;
        -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
        *)            echo "ERROR: unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ -z "$CONFIG" || ! -f "$CONFIG" ]] && { echo "ERROR: --config <yaml> required" >&2; exit 2; }
case "$MODE" in
    depletion|shift|depletion-magnitude|all) ;;
    *) echo "ERROR: --mode must be depletion|shift|depletion-magnitude|all" >&2; exit 2 ;;
esac

# --dry-run: validate orchestration without ever invoking run_openai. Useful
# to confirm prereq auto-skip, sample resolution, and pred.json layout
# without sending a single request to OpenAI. Bypasses the OPENAI_API_KEY
# check too, since no API call will be made.
if [[ $DRY_RUN -eq 1 ]]; then
    : "${OPENAI_API_KEY:=__dry_run__}"
fi

# ── Env (orchestration only — generation comes from yaml) ────────────────────
PARALLEL_HIDS="${PARALLEL_HIDS:-8}"
CONCURRENCY="${CONCURRENCY:-20}"
ABLATION="${ABLATION:-full}"
SPLIT="${SPLIT:-test}"
FRACS_STR="${FRACS:-0.0 0.9 0.99}"
ROOT="${ROOT:-results/sweep}"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark}"
DATA_DIR="${DATA_DIR:-data}"

# ── Load yaml → MODEL, SERVED_NAME, OPENAI_ARGS[] ────────────────────────────
eval "$(python3 -m src.llm_gate.scripts._config_to_shell --config "$CONFIG")" || {
    echo "ERROR: failed to parse $CONFIG" >&2; exit 2; }
[[ "${OPENAI_MODE:-0}" != "1" ]] && {
    echo "ERROR: $CONFIG is not an OpenAI API config (model.openai_id missing)." >&2; exit 2; }
[[ -z "${OPENAI_API_KEY:-}" ]] && { echo "ERROR: OPENAI_API_KEY not set" >&2; exit 2; }

LOG_DIR="$ROOT/_logs/hard_openai_${SERVED_NAME}"
mkdir -p "$LOG_DIR" /tmp/c2s_logs
MAIN_LOG="$LOG_DIR/main.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }

# ── Build job list (mode|hid|frac, with frac="" for nominal) ─────────────────
list_hids() {
    python3 -c "
import yaml,sys
with open('$1') as f: d=yaml.safe_load(f)
for t in d['tasks']: print(t['hard_id'])
"
}
DEP_HIDS=$(list_hids "$BENCHMARK_DIR/hard_depletion.yaml")
SHIFT_HIDS=$(list_hids "$BENCHMARK_DIR/hard_shift.yaml")

JOBS=()
add_dep_nominal() { for h in $DEP_HIDS; do JOBS+=("depletion|$h|"); done; }
add_shift()       { for h in $SHIFT_HIDS; do JOBS+=("shift|$h|"); done; }
add_magnitude()   { for h in $DEP_HIDS; do for f in $FRACS_STR; do JOBS+=("depletion|$h|$f"); done; done; }
case "$MODE" in
    depletion)            add_dep_nominal ;;
    shift)                add_shift ;;
    depletion-magnitude)  add_magnitude ;;
    all)                  add_dep_nominal; add_shift; add_magnitude ;;
esac

# Optional --hids filter (matches bare HID OR fully qualified HID_fracNN slug).
if [[ ${#HIDS_FILTER[@]} -gt 0 ]]; then
    NEW=()
    for j in "${JOBS[@]}"; do
        IFS='|' read -r _m h f <<< "$j"
        # Compute slug: bare HID or HID_fracNN.
        local_slug="$h"
        if [[ -n "$f" ]]; then
            case "$f" in
                0.0|0) fn=0 ;; 0.9) fn=90 ;; 0.99) fn=99 ;;
                *) fn=$(python3 -c "print(int(float('$f')*100))") ;;
            esac
            local_slug="${h}_frac${fn}"
        fi
        for k in "${HIDS_FILTER[@]}"; do
            [[ "$h" == "$k" || "$local_slug" == "$k" ]] && { NEW+=("$j"); break; }
        done
    done
    JOBS=("${NEW[@]}")
fi

log "================================================================"
log "OpenAI hard sweep — Stage 1 (LLM gen)"
log "  served_name : $SERVED_NAME"
log "  mode        : $MODE   jobs=${#JOBS[@]}"
log "  parallel    : ${PARALLEL_HIDS} HIDs × concurrency=${CONCURRENCY}"
log "  prereq      : $([[ $NO_PREREQ -eq 1 ]] && echo SKIP || echo build-on-demand)"
log "  output root : $ROOT/setting2_<mode>/$SERVED_NAME/"
log "================================================================"

# ── Worker (one job = one (mode, hid, frac)) ─────────────────────────────────
run_one_job() {
    local mode=$1 hid=$2 frac=$3
    local flag_name; [[ "$mode" == depletion ]] && flag_name=--hard-depletion || flag_name=--hard-shift

    # slug: bare HID for nominal, HID_fracNN for magnitude.
    local slug="$hid"
    local frac_args=()
    if [[ -n "$frac" ]]; then
        local fn
        case "$frac" in
            0.0|0) fn=0 ;; 0.9) fn=90 ;; 0.99) fn=99 ;;
            *)     fn=$(python3 -c "print(int(float('$frac')*100))") ;;
        esac
        slug="${hid}_frac${fn}"
        frac_args+=(--hard-depletion-fraction "$frac")
    fi

    local hid_dir="$ROOT/setting2_${mode}/${SERVED_NAME}/${slug}"
    mkdir -p "$hid_dir"

    # Resolve dataset from HID.
    local ds
    if [[ "$mode" == depletion ]]; then
        ds=$(python3 -c "from src.hard_depletions import get_hard_depletion as g; print(g('$hid').dataset)" 2>/dev/null)
    else
        ds=$(python3 -c "from src.hard_shifts import get_hard_shift as g; print(g('$hid').dataset)" 2>/dev/null)
    fi
    [[ -z "$ds" ]] && { log "  [ERROR] $mode/$slug — could not resolve dataset"; return 1; }

    local out="$hid_dir/${ds}_${SERVED_NAME}_${ABLATION}.json"
    if [[ -f "$out" ]]; then
        log "  [skip]  $mode/$slug — pred.json exists"
        return 0
    fi

    # ── Sample list FIRST so we know the expected prereq count. ─────────────
    local samples
    if ! mapfile -t samples < <(
        python3 -m src.llm.scripts.list_split_samples \
            --splits "$BENCHMARK_DIR/splits.json" --dataset "$ds" --split "$SPLIT" \
            2> "/tmp/c2s_logs/openai_split_${slug}.log"
    ); then
        log "  [ERROR] $mode/$slug — list_split_samples failed"; return 1
    fi
    if [[ ${#samples[@]} -eq 0 ]]; then
        log "  [skip]  $mode/$slug — no $SPLIT samples"; return 0
    fi
    local n_expected=${#samples[@]}

    # ── Stage 0 prereqs — per-spot skip (handles partial cleanly). ──────────
    # task.py creates step_NN/gate_task__<slug>.json (1 per sample).
    # prompt.py creates step_NN/gate_${ABL}__<slug>/cells.md from each task.json.
    # Decision matrix vs $n_expected (test split sample count):
    #   --no-prereq                            → skip task + prompt (caller's promise)
    #   n_task ≥ expected, n_md ≥ expected     → skip task + prompt (fully built)
    #   n_task ≥ expected, n_md  < expected    → skip task, run prompt (task.py would
    #                                            just re-skip; saves ~3-4s/HID startup)
    #   else                                   → run both (handles partial via
    #                                            --skip-existing on task.py side)
    # Net effect: a fully-prebuilt sweep skips both Python invocations entirely
    # (HID-level cost ≈ 2 × find ≈ 50 ms instead of ≈ 3-4 s for task.py startup).
    if [[ $NO_PREREQ -eq 0 ]]; then
        local n_task n_md
        n_task=$(find "$BENCHMARK_DIR/$ds" -maxdepth 4 -type f \
                      -name "gate_task__${slug}.json" 2>/dev/null | wc -l)
        n_md=$(find "$BENCHMARK_DIR/$ds" -maxdepth 4 -type f \
                    -path "*gate_${ABLATION}__${slug}/cells.md" 2>/dev/null | wc -l)

        if [[ $n_task -ge $n_expected && $n_md -ge $n_expected ]]; then
            log "  [auto-skip-prereq] $mode/$slug — fully built (task=$n_task md=$n_md, expected≥$n_expected)"
        elif [[ $n_task -ge $n_expected && $n_md -lt $n_expected ]]; then
            log "  [auto-skip-task] $mode/$slug — task.py done ($n_task), running prompt.py for missing cells.md ($n_md/$n_expected)"
            : > "$LOG_DIR/${slug}_prompt.log"
            local pf=0
            for sd in "$BENCHMARK_DIR/$ds"/*/; do
                [[ -d "$sd" ]] || continue
                python3 -m src.llm_gate.prompt --batch "${sd%/}" --save_md \
                    $flag_name "$hid" "${frac_args[@]}" \
                    >> "$LOG_DIR/${slug}_prompt.log" 2>&1 || pf=$((pf+1))
            done
            [[ $pf -gt 0 ]] && log "  [WARN]  $mode/$slug — prompt.py failed for $pf samples"
        else
            log "  [build-prereq] $mode/$slug — task=$n_task md=$n_md expected=$n_expected; running task.py + prompt.py"
            python3 -m src.llm_gate.task \
                --benchmark "$BENCHMARK_DIR" --data-dir "$DATA_DIR" \
                --skip-existing \
                $flag_name "$hid" "${frac_args[@]}" \
                > "$LOG_DIR/${slug}_task.log" 2>&1 || {
                    log "  [ERROR] $mode/$slug — task.py failed"; return 1; }
            : > "$LOG_DIR/${slug}_prompt.log"
            local pf=0
            for sd in "$BENCHMARK_DIR/$ds"/*/; do
                [[ -d "$sd" ]] || continue
                python3 -m src.llm_gate.prompt --batch "${sd%/}" --save_md \
                    $flag_name "$hid" "${frac_args[@]}" \
                    >> "$LOG_DIR/${slug}_prompt.log" 2>&1 || pf=$((pf+1))
            done
            [[ $pf -gt 0 ]] && log "  [WARN]  $mode/$slug — prompt.py failed for $pf samples"
        fi
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] $mode/$slug (ds=$ds, n=${#samples[@]}) — would write $out"
        return 0
    fi

    log "  [start] $mode/$slug (ds=$ds, n=${#samples[@]})"
    # OPENAI_ARGS is rebuilt from OPENAI_ARGS_STR inside the xargs subshell.
    python3 -m src.llm_gate.inference.run_openai \
        --dataset_path "$BENCHMARK_DIR/$ds" \
        --model "$SERVED_NAME" \
        --ablation_slug "$ABLATION" \
        --concurrency "$CONCURRENCY" \
        --output_path "$out" \
        --sample "${samples[@]}" \
        $flag_name "$hid" "${frac_args[@]}" \
        --save-raw \
        "${OPENAI_ARGS[@]}" \
        > "$LOG_DIR/${slug}_client.log" 2>&1
    local rc=$?
    log "  [done]  $mode/$slug (rc=$rc)"
    return $rc
}

# ── xargs wrapper: rebuild OPENAI_ARGS array from the exported string ────────
xargs_worker() {
    local job=$1
    IFS='|' read -r m h f <<< "$job"
    # shellcheck disable=SC2206  # word-splitting intentional (no spaces in args)
    OPENAI_ARGS=($OPENAI_ARGS_STR)
    run_one_job "$m" "$h" "$f"
}

export -f log run_one_job xargs_worker
export SERVED_NAME ABLATION CONCURRENCY ROOT LOG_DIR MAIN_LOG \
       BENCHMARK_DIR DATA_DIR SPLIT NO_PREREQ DRY_RUN
# Bash arrays don't survive `export`; serialize OPENAI_ARGS as a string and
# re-split inside xargs_worker (safe — args from yaml never contain spaces).
OPENAI_ARGS_STR="${OPENAI_ARGS[*]}"
export OPENAI_ARGS_STR

# ── Run ──────────────────────────────────────────────────────────────────────
T0=$(date +%s)
printf '%s\n' "${JOBS[@]}" | xargs -P "$PARALLEL_HIDS" -I {} bash -c 'xargs_worker "$@"' _ "{}"
T1=$(date +%s)

# ── Summary ──────────────────────────────────────────────────────────────────
DONE=0; MISS=0
for j in "${JOBS[@]}"; do
    IFS='|' read -r m h f <<< "$j"
    slug="$h"; if [[ -n "$f" ]]; then
        case "$f" in 0.0|0) fn=0;; 0.9) fn=90;; 0.99) fn=99;; *) fn=$(python3 -c "print(int(float('$f')*100))");; esac
        slug="${h}_frac${fn}"
    fi
    ds=$(python3 -c "
from src.hard_depletions import get_hard_depletion as gd
from src.hard_shifts import get_hard_shift as gs
print((gd if '$m'=='depletion' else gs)('$h').dataset)
" 2>/dev/null)
    out="$ROOT/setting2_${m}/${SERVED_NAME}/${slug}/${ds}_${SERVED_NAME}_${ABLATION}.json"
    if [[ -f "$out" ]]; then DONE=$((DONE+1)); else MISS=$((MISS+1)); fi
done
log "================================================================"
if [[ $DRY_RUN -eq 1 ]]; then
    log "Stage 1 dry-run finished in $((T1-T0))s — ${#JOBS[@]} jobs validated, NO pred.json written"
    log "  (exit 0; orchestration only — OpenAI was never contacted)"
    log "================================================================"
    exit 0
fi
log "Stage 1 finished in $((T1-T0))s — done=$DONE  missing=$MISS  total=${#JOBS[@]}"
log "Next:"
log "  Stage 2:  bash run_propagate_all.sh   ($SERVED_NAME)"
log "  Stage 3:  bash run_eval_all.sh        ($SERVED_NAME)"
log "================================================================"
exit $MISS
