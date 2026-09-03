#!/bin/bash
# Lyoplate pipeline: GatingSetList -> per-sample CSV -> Parquet, per panel.
#
# Run from the Lyoplate data directory (the one holding gslist-bcell/, etc.):
#   bash pipeline/run_pipeline.sh                 # all panels
#   bash pipeline/run_pipeline.sh bcell           # single panel
#   bash pipeline/run_pipeline.sh --test 2        # all panels, 2 samples each
#   bash pipeline/run_pipeline.sh bcell --test 2  # single panel, 2 samples
#
# Panel names are case-sensitive: bcell tcell DC treg
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Parse panel argument (default: all panels)
PANELS="bcell tcell DC treg"
if [[ "${1:-}" != "" && "${1:-}" != "--test" ]]; then
    PANELS="$1"
    shift || true
fi

MAX_ARGS=""
if [[ "${1:-}" == "--test" ]]; then
    MAX_ARGS="--max_samples ${2:-2}"
    echo "=== TEST MODE: ${2:-2} samples ==="
    shift 2 || true
fi

cd "$BASE_DIR"

for PANEL in $PANELS; do
    echo ""
    echo "========================================="
    echo "Panel: $PANEL"
    echo "========================================="

    echo "Step 1: Extract from GatingSet -> CSV"
    Rscript "$SCRIPT_DIR/extract_from_gs.R" \
        --panel "$PANEL" \
        $MAX_ARGS

    echo ""
    echo "Step 2: CSV -> Parquet"
    mkdir -p "$SCRIPT_DIR/$PANEL/parquet"
    python3 "$SCRIPT_DIR/convert_to_parquet.py" \
        --csv_dir "$SCRIPT_DIR/$PANEL/csv_tmp" \
        --out_dir "$SCRIPT_DIR/$PANEL/parquet" \
        --gating_plan "$SCRIPT_DIR/$PANEL/gating_plan.json" \
        --config "$SCRIPT_DIR/$PANEL/dataset_config.yaml" \
        $MAX_ARGS
done

echo ""
echo "========================================="
echo "Pipeline complete!"
for PANEL in $PANELS; do
    N=$(ls "$SCRIPT_DIR/$PANEL/parquet/"*.parquet 2>/dev/null | wc -l | tr -d ' ')
    echo "  $PANEL: $N parquet files"
done
echo "========================================="
