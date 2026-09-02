#!/usr/bin/env bash
# CPU-only propagate stage. Reads pred.json, writes predictions/ dirs.
# Skips HIDs whose .propagate_done marker exists.
#
# Usage:
#   N_PARALLEL=96 SERVED_NAME=Qwen3.6-27B bash run_propagate_all.sh

set -uo pipefail
cd "$(dirname "$0")"

SERVED_NAME=${SERVED_NAME:-Qwen3.6-27B}
ABLATION=${ABLATION:-full}
METHOD_TAG="gate_${SERVED_NAME}_${ABLATION}"
N_PARALLEL=${N_PARALLEL:-96}

ROOT=results/sweep
LOG_DIR="$ROOT/_logs/propagate"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/main.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }

JOBS_FILE=$(mktemp)
trap "rm -f $JOBS_FILE" EXIT

log "collecting propagate jobs (skip if .propagate_done exists)"
python3 - >> "$JOBS_FILE" <<'PY'
import yaml
from pathlib import Path
import os

SERVED = os.environ.get('SERVED_NAME', 'Qwen3.6-27B')
ABL    = os.environ.get('ABLATION', 'full')
ROOT_S = Path('results/sweep/setting2_shift')    / SERVED
ROOT_D = Path('results/sweep/setting2_depletion') / SERVED

# shift (no fraction)
with open('benchmark/hard_shift.yaml') as f: sd = yaml.safe_load(f)
for t in sd['tasks']:
    hid = t['hard_id']; ds = t['dataset']
    hid_dir = ROOT_S / hid
    if (hid_dir / '.propagate_done').exists(): continue
    pred_json = hid_dir / f'{ds}_{SERVED}_{ABL}.json'
    if not pred_json.exists(): continue
    print(f'shift\t{hid}\t\t{ds}\t{pred_json}\t{hid_dir}')

# depletion default + magnitude
with open('benchmark/hard_depletion.yaml') as f: dd = yaml.safe_load(f)
for t in dd['tasks']:
    hid = t['hard_id']; ds = t['dataset']
    # default (no suffix)
    hid_dir = ROOT_D / hid
    pred_json = hid_dir / f'{ds}_{SERVED}_{ABL}.json'
    if pred_json.exists() and not (hid_dir / '.propagate_done').exists():
        print(f'depletion\t{hid}\t\t{ds}\t{pred_json}\t{hid_dir}')
    # magnitude
    for frac in [0.0, 0.9, 0.99]:
        pct = int(round(frac*100))
        slug = f'{hid}_frac{pct}'
        hd = ROOT_D / slug
        pj = hd / f'{ds}_{SERVED}_{ABL}.json'
        if pj.exists() and not (hd / '.propagate_done').exists():
            print(f'depletion\t{hid}\t{frac}\t{ds}\t{pj}\t{hd}')
PY

N=$(wc -l < "$JOBS_FILE")
log "Total propagate jobs: $N (parallel=$N_PARALLEL)"
[[ "$N" == "0" ]] && { log "nothing to do"; exit 0; }

run_one_propagate() {
    # tab-separated: mode hid frac ds pred_json hid_dir
    local mode=$1; local hid=$2; local frac=$3; local ds=$4
    local pred_json=$5; local hid_dir=$6
    local pred_dir="$hid_dir/predictions"
    local flag_name suffix frac_args=()
    [[ "$mode" == "depletion" ]] && flag_name=--hard-depletion || flag_name=--hard-shift
    if [[ -n "$frac" && "$frac" != "1.0" ]]; then
        local pct=$(python3 -c "print(int(round(float('$frac')*100)))")
        suffix="_frac${pct}"
        frac_args=(--hard-depletion-fraction "$frac")
    fi
    local tag="${mode}_${hid}${suffix}"

    if python3 -m src.llm_gate.postprocess.propagate \
        --eval_path "$pred_json" --dataset "$ds" \
        --benchmark benchmark/ --data-dir data/ \
        --method "$METHOD_TAG" --output_dir "$pred_dir" \
        "$flag_name" "$hid" "${frac_args[@]}" \
        > "$LOG_DIR/${tag}.log" 2>&1; then
        touch "$hid_dir/.propagate_done"
        echo "[done] $tag"
    else
        echo "[ERROR] $tag rc=$? (see $LOG_DIR/${tag}.log)"
    fi
}
export -f run_one_propagate
export METHOD_TAG LOG_DIR

cat "$JOBS_FILE" | xargs -P "$N_PARALLEL" -I {} bash -c '
    line="{}"
    mode=$(echo "$line" | cut -f1); hid=$(echo "$line" | cut -f2)
    frac=$(echo "$line" | cut -f3); ds=$(echo "$line" | cut -f4)
    pj=$(echo "$line" | cut -f5);   hd=$(echo "$line" | cut -f6)
    out=$(run_one_propagate "$mode" "$hid" "$frac" "$ds" "$pj" "$hd")
    echo "[$(date +%H:%M:%S)] $out" >> "'"$MAIN_LOG"'"
'

log "all propagate jobs done"
