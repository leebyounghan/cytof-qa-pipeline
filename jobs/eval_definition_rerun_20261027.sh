#!/usr/bin/env bash
#BSUB -J "evaldef[1-11]%2"
#BSUB -q i2c2_normal
#BSUB -n 2
#BSUB -M 40000
#BSUB -R "rusage[mem=40000] span[hosts=1]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/evaldef.%J.%I.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/evaldef.%J.%I.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=.venv/bin/python
RUN=$($PY - "$LSB_JOBINDEX" <<'PY'
import json, sys
reports = json.load(open('jobs/definition_rerun_audit_20261027.json'))
print(sorted(reports)[int(sys.argv[1]) - 1])
PY
)
echo "Merging and evaluating $RUN"
DATASETS=(Acute2020 Acute2021 Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue)
BACKUP=results/_definition_rerun_20261027/backup/$RUN/eval
mkdir -p "$BACKUP"
for DS in "${DATASETS[@]}"; do
  for NAME in eval_summary.json eval_summary__gt_rect1_oracle.json; do
    [[ -f "results/$RUN/predictions/$DS/$NAME" ]] || { echo "ERROR: missing prior $RUN/$DS/$NAME" >&2; exit 2; }
    [[ -f "$BACKUP/${DS}__$NAME" ]] || cp "results/$RUN/predictions/$DS/$NAME" "$BACKUP/${DS}__$NAME"
  done
done
$PY jobs/merge_definition_rerun_20261027.py "$RUN"
for DS in "${DATASETS[@]}"; do
  mapfile -t MATCHES < <(compgen -G "results/$RUN/${DS}_*.json")
  (( ${#MATCHES[@]} == 1 )) || { echo "ERROR: expected one output for $RUN/$DS" >&2; exit 2; }
  case "$RUN" in
    c2s_flowdensity_*)
      SERVED=${RUN#c2s_flowdensity_}
      $PY -m src.llm.postprocess.propagate --eval_path "${MATCHES[0]}" --dataset "$DS" \
        --benchmark benchmark --data-dir data --method "c2s_grid_${SERVED}_hvp10" \
        --output_dir "results/$RUN/predictions" --mapping-filename cell2grid.npz
      ;;
    c2s_flowsom_*)
      SERVED=${RUN#c2s_flowsom_}
      $PY -m src.llm_flowsom.postprocess.propagate --eval_path "${MATCHES[0]}" --dataset "$DS" \
        --benchmark benchmark --data-dir data --method "c2s_flowsom_${SERVED}_hvp10" \
        --output_dir "results/$RUN/predictions"
      ;;
    gate_*)
      $PY -m src.llm_gate.postprocess.propagate --eval_path "${MATCHES[0]}" --dataset "$DS" \
        --benchmark benchmark --data-dir data --method "$RUN" \
        --output_dir "results/$RUN/predictions" --tiebreak smallest
      ;;
  esac
  $PY -m src.eval --predictions "results/$RUN/predictions" --benchmark benchmark --data-dir data \
    --datasets "$DS" --split benchmark/splits.json --split-set test
  $PY -m src.eval --predictions "results/$RUN/predictions" --gt-labels results/rect1_oracle \
    --benchmark benchmark --data-dir data --datasets "$DS" \
    --split benchmark/splits.json --split-set test
  echo "Evaluated $RUN/$DS"
done
cp benchmark/notes.yaml "results/$RUN/notes.yaml.used"
echo "DONE: $RUN"
