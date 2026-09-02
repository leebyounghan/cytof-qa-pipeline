#!/usr/bin/env bash
# Parallel CPU eval for all (model × setting × HID) combinations produced by
# the sweep launchers. Skips entries that already have eval_summary.json.
# After all evals, aggregates into results/sweep/sweep_results.csv.

set -uo pipefail
cd "$(dirname "$0")"

ROOT=results/sweep
LOG_DIR="$ROOT/_logs"
mkdir -p "$LOG_DIR"
N_PARALLEL="${N_PARALLEL:-$(nproc)}"

echo "[eval_all] N_PARALLEL=$N_PARALLEL"

JOBS_FILE=$(mktemp)
trap "rm -f $JOBS_FILE" EXIT

emit_job() {
    # Each line: <log_path>\t<eval cmd>
    local log=$1; shift
    printf '%s\t%s\n' "$log" "$*" >> "$JOBS_FILE"
}

# ── Setting 1 (per model) ─────────────────────────────────────────────────────
for pred_dir in "$ROOT"/setting1/*/predictions; do
    [[ -d "$pred_dir" ]] || continue
    [[ -f "$pred_dir/eval_summary.json" ]] && continue
    model=$(basename "$(dirname "$pred_dir")")
    emit_job "$LOG_DIR/eval_setting1_${model}.log" \
        python3 -m src.eval --predictions "$pred_dir" \
        --benchmark benchmark/ --data-dir data/
done

# ── Setting 2 depletion (per model × HID, with optional _fracNN suffix) ──────
for pred_dir in "$ROOT"/setting2_depletion/*/*/predictions; do
    [[ -d "$pred_dir" ]] || continue
    [[ -f "$pred_dir/eval_summary.json" ]] && continue
    slug=$(basename "$(dirname "$pred_dir")")
    model=$(basename "$(dirname "$(dirname "$pred_dir")")")
    # Split _fracNN suffix → bare HID + fraction
    if [[ "$slug" =~ ^(.*)_frac([0-9]+)$ ]]; then
        hid="${BASH_REMATCH[1]}"
        pct="${BASH_REMATCH[2]}"
        frac=$(python3 -c "print(int('$pct')/100)")
        emit_job "$LOG_DIR/eval_depletion_${model}_${slug}.log" \
            python3 -m src.eval --predictions "$pred_dir" \
            --benchmark benchmark/ --data-dir data/ \
            --hard-depletion "$hid" --hard-depletion-fraction "$frac"
    else
        emit_job "$LOG_DIR/eval_depletion_${model}_${slug}.log" \
            python3 -m src.eval --predictions "$pred_dir" \
            --benchmark benchmark/ --data-dir data/ \
            --hard-depletion "$slug"
    fi
done

# ── Setting 2 shift (per model × HID) ─────────────────────────────────────────
for pred_dir in "$ROOT"/setting2_shift/*/*/predictions; do
    [[ -d "$pred_dir" ]] || continue
    [[ -f "$pred_dir/eval_summary.json" ]] && continue
    hid=$(basename "$(dirname "$pred_dir")")
    model=$(basename "$(dirname "$(dirname "$pred_dir")")")
    emit_job "$LOG_DIR/eval_shift_${model}_${hid}.log" \
        python3 -m src.eval --predictions "$pred_dir" \
        --benchmark benchmark/ --data-dir data/ \
        --hard-shift "$hid"
done

N_JOBS=$(wc -l < "$JOBS_FILE")
echo "[eval_all] $N_JOBS pending eval jobs"
if [[ "$N_JOBS" == "0" ]]; then
    echo "[eval_all] nothing to do"
else
    # Run jobs in parallel. Each line: log<TAB>cmd...
    # xargs -P N runs N at once; bash -c executes per line with full quoting.
    awk -F'\t' '{print NR "\t" $0}' "$JOBS_FILE" | \
        xargs -P "$N_PARALLEL" -I{} bash -c '
            line="{}"
            id=$(echo "$line" | cut -f1)
            log=$(echo "$line" | cut -f2)
            cmd=$(echo "$line" | cut -f3-)
            echo "[$(date +%H:%M:%S)] [$id] start: $cmd" >> "'"$LOG_DIR"'/eval_all_main.log"
            eval "$cmd" > "$log" 2>&1
            rc=$?
            echo "[$(date +%H:%M:%S)] [$id] rc=$rc" >> "'"$LOG_DIR"'/eval_all_main.log"
        '
fi

# ── Aggregate ────────────────────────────────────────────────────────────────
echo "[eval_all] aggregating into $ROOT/sweep_results.csv"
python3 - <<'PY'
import json, sys, csv
from pathlib import Path

ROOT = Path('results/sweep')
rows = []

def walk(path: Path, setting: str, model: str, hid: str | None):
    overall = path / 'eval_summary.json'
    if not overall.exists(): return
    try:
        d = json.loads(overall.read_text())
    except Exception as e:
        print(f'  parse-fail {overall}: {e}', file=sys.stderr); return
    for ds, methods in (d.items() if isinstance(d, dict) else []):
        if not isinstance(methods, dict): continue
        for method, m in methods.items():
            if not isinstance(m, dict): continue
            rows.append({
                'setting': setting, 'model': model, 'hid': hid or '',
                'dataset': ds, 'method': method,
                'n_tasks': m.get('n_tasks') or m.get('n', ''),
                'f1_macro':     m.get('f1_macro_mean')                    or m.get('f1_macro', ''),
                'f1_weighted':  m.get('f1_weighted_mean')                 or m.get('f1_weighted', ''),
                'bal_acc':      m.get('balanced_accuracy_mean')           or m.get('balanced_accuracy', ''),
                'pop_prop_err': m.get('population_proportion_error_mean') or m.get('population_proportion_error', ''),
                'hull_iou':     m.get('hull_iou_mean')                    or m.get('hull_iou', ''),
            })

for s1m in sorted((ROOT / 'setting1').glob('*/predictions')):
    walk(s1m, 'setting1', s1m.parent.name, None)
for hd in sorted((ROOT / 'setting2_depletion').glob('*/*/predictions')):
    walk(hd, 'setting2_depletion', hd.parent.parent.name, hd.parent.name)
for hd in sorted((ROOT / 'setting2_shift').glob('*/*/predictions')):
    walk(hd, 'setting2_shift', hd.parent.parent.name, hd.parent.name)

out = ROOT / 'sweep_results.csv'
if rows:
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {len(rows)} rows -> {out}')
else:
    print('No eval_summary.json files found')
PY
echo "[eval_all] DONE"
