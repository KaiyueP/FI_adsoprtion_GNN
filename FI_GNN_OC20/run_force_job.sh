#!/bin/bash
#SBATCH --job-name=fire-force-inputs
#SBATCH --account=m4868
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mail-user=kaiyue_peng@berkeley.edu
#SBATCH --mail-type=ALL

# Invoke with bash, not sbatch: create the run folder before Slurm opens its logs.
set -euo pipefail
# Slurm executes a spool copy: BASH_SOURCE is NOT the source checkout.
REPO=${FIRE_GNN_REPO:-/pscratch/sd/k/kaiyuep/ChemE239/Project/project/FIRE-GNN}
REPO=$(cd -- "$REPO" && pwd)
export FIRE_GNN_REPO="$REPO"
SCRIPT="$REPO/FIRE/run_force_job.sh"
[[ -f "$REPO/FIRE/Dataset/adsorption_records.py" ]] || { echo "Invalid FIRE_GNN_REPO: $REPO" >&2; exit 2; }
cd "$REPO"
MLIP=${MLIP:-orb}
FORCE_MODE=${FORCE_MODE:-all}
POOLING=${POOLING:-adsorbate_mean}
SEED=${SEED:-13}
case "$MLIP" in orb|uma) ;; *) echo 'Supported MLIPs: orb or uma' >&2; exit 2;; esac
case "$FORCE_MODE" in none|adsorbate|adsorbate_nearby|all) ;; *) echo 'Invalid FORCE_MODE' >&2; exit 2;; esac
case "$POOLING" in mean|adsorbate_mean) ;; *) echo 'POOLING must be mean or adsorbate_mean' >&2; exit 2;; esac
if [[ -z ${SLURM_JOB_ID:-} && ${PREPARE_ONLY:-0} != 1 ]]; then
  RUN_DIR=$(mktemp -d "$REPO/adsorption_runs/force_${MLIP}_${FORCE_MODE}_${POOLING}_seed${SEED}_XXXXXXXX")
  export FIRE_FORCE_RUN_DIR="$RUN_DIR"
  sbatch --export=ALL --chdir="$REPO" --output="$RUN_DIR/job-%j.out" --error="$RUN_DIR/job-%j.err" \
    --time="${WALLTIME:-06:00:00}" "$SCRIPT"
  printf 'Results and Slurm logs: %s\n' "$RUN_DIR"
  exit 0
fi
# Presets can also be submitted directly with sbatch. Their temporary Slurm
# streams go to /dev/null; redirect immediately into the assigned result folder.
if [[ -z ${FIRE_FORCE_RUN_DIR:-} ]]; then
  export FIRE_FORCE_RUN_DIR=$(mktemp -d "$REPO/adsorption_runs/force_${MLIP}_${FORCE_MODE}_${POOLING}_seed${SEED}_XXXXXXXX")
  exec >"$FIRE_FORCE_RUN_DIR/job-${SLURM_JOB_ID:-prepare}.out" 2>"$FIRE_FORCE_RUN_DIR/job-${SLURM_JOB_ID:-prepare}.err"
fi
RUN_DIR=$FIRE_FORCE_RUN_DIR
module load conda
GNN_ENV=${GNN_ENV:-fire_gnn}
conda activate "$GNN_ENV"
printf 'GNN environment: %s; project: %s\n' "$CONDA_DEFAULT_ENV" "$REPO"
CALCULATOR_KWARGS=${CALCULATOR_KWARGS:-'{}'}
FORCE_RADIUS=${FORCE_RADIUS:-3.0}
FORCE_SCALE=${FORCE_SCALE:-1.0}
LOSS=${LOSS:-mse}
EPOCHS=${EPOCHS:-1000}
PATIENCE=${PATIENCE:-30}
BATCH_SIZE=${BATCH_SIZE:-16}
WORKERS=${WORKERS:-4}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$RUN_DIR/data"
# Persist the original inputs and all chosen controls under this run's folder.
python - "$RUN_DIR" "$MLIP" "$FORCE_MODE" "$POOLING" "$SEED" "$LOSS" "$FORCE_RADIUS" "$FORCE_SCALE" "$EPOCHS" "$BATCH_SIZE" <<'PY'
import json, os, shutil, sys
from pathlib import Path
repo = Path(os.environ['FIRE_GNN_REPO'])
sys.path.insert(0, str(repo / 'FIRE'))
from Dataset.adsorption_records import load_records, check_splits, normalize_force_config
root=Path(sys.argv[1]); shared=Path(os.environ.get('SHARED_DATA_DIR', str(repo/'adsorption_runs/shared_data')))
records=[load_records(shared/f'{s}.json') for s in ('train','validation','test')]
check_splits(*records)
for split in ('train','validation','test'):
    shutil.copyfile(shared/f'{split}.json',root/'data'/f'{split}.json')
c=json.loads((shared/'config.json').read_text())
c.update(force_mode=sys.argv[3],pooling=sys.argv[4],seed=int(sys.argv[5]),loss=sys.argv[6],
         force_radius=float(sys.argv[7]),force_scale=float(sys.argv[8]),
         epochs=int(sys.argv[9]),batch_size=int(sys.argv[10]))
c=normalize_force_config(c)
(root/'config.json').write_text(json.dumps(c,indent=2))
(root/'experiment.json').write_text(json.dumps(dict(mlip=sys.argv[2],force_mode=sys.argv[3],pooling=sys.argv[4],
                                                   seed=int(sys.argv[5]),relaxation=False, data_source=str(shared),
                                                   train_count=len(records[0]),validation_count=len(records[1]),test_count=len(records[2])),indent=2))
PY
if [[ ${PREPARE_ONLY:-0} == 1 ]]; then
  printf 'Preparation verified: %s (no force inference or training)\n' "$RUN_DIR"
  exit 0
fi
if [[ "$FORCE_MODE" != none ]]; then
  for split in train validation test; do
    if [[ -n ${FORCE_DIR:-} ]]; then
      # Reuse only force data matching the exact original records and source family.
      python - "$RUN_DIR" "$FORCE_DIR" "$split" "$MLIP" <<'PY'
import json,os,shutil,sys
from pathlib import Path
repo = Path(os.environ['FIRE_GNN_REPO'])
sys.path.insert(0,str(repo / 'FIRE'))
from Dataset.adsorption_records import load_records,validate_forces
root,source,split,mlip=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4]
path=source/f'{split}_forces.json';records=load_records(path)
stripped=[{k:v for k,v in r.items() if k not in ('mlip_forces','mlip_force_metadata')} for r in records]
assert stripped==load_records(root/'data'/f'{split}.json'),'Precomputed forces do not match input records'
for r in records:
    validate_forces(r,required=True)
    assert mlip in r['mlip_force_metadata']['source'].lower(),'FORCE_DIR source disagrees with MLIP'
shutil.copyfile(path,root/'data'/f'{split}_forces.json')
PY
    elif [[ "$MLIP" == uma ]]; then
      # Only UMA inference switches environments. Keep GNN/Orb dependencies isolated.
      conda activate "${UMA_ENV:-gnn_uma}"
      srun --ntasks=1 --cpus-per-task=16 --cpu-bind=cores \
        python -u "$REPO/FIRE/add_mlip_forces.py" \
        "$RUN_DIR/data/$split.json" "$RUN_DIR/data/${split}_forces.json" \
        --mlip uma --device cpu --calculator-kwargs "$CALCULATOR_KWARGS"
      conda activate "$GNN_ENV"
    else
      srun --ntasks=1 --cpus-per-task=16 --cpu-bind=cores \
        python -u "$REPO/FIRE/add_mlip_forces.py" \
        "$RUN_DIR/data/$split.json" "$RUN_DIR/data/${split}_forces.json" \
        --mlip "$MLIP" --device cpu --slab-vacuum 15 \
        --calculator-kwargs "$CALCULATOR_KWARGS"
    fi
  done
  suffix=_forces
else
  suffix=
fi
srun --ntasks=1 --cpus-per-task=16 --cpu-bind=cores \
  python -u "$REPO/FIRE/run_adsorption.py" train \
  --config "$RUN_DIR/config.json" --train "$RUN_DIR/data/train${suffix}.json" \
  --validation "$RUN_DIR/data/validation${suffix}.json" --test "$RUN_DIR/data/test${suffix}.json" \
  --output "$RUN_DIR/model" --device cpu --force-mode "$FORCE_MODE" --pooling "$POOLING" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --patience "$PATIENCE" --num-workers "$WORKERS" \
  --cache-dir "$RUN_DIR/graph_cache"
