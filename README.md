# CHEMENG 239: adsorption-energy prediction

This is the single guide for the CHEMENG 239 adsorption project, including
setup, force inputs, z-axis options, pooling, saved experiments and attribution.
Run commands from the FIRE-GNN repository root, not from inside FIRE.

## Scope and attribution

The project adapts the upstream [FIRE-GNN](https://github.com/d2r2group/FIRE-GNN)
code to OC20 initial-structure-to-relaxed-energy prediction (IS2RE). Its active
architecture is **SEMPNN/SEGNN**: elemental features and an adsorbate indicator,
equivariant message passing, and a scalar energy output. Optional MLIP force
vectors from Orb-v3, localized force features, a fixed-z reference and
adsorbate-only readout are supported together.
No new DFT labels, force prediction or geometry relaxation are performed.

The model is adapted from [Steerable-E3-GNN](https://github.com/RobDHess/Steerable-E3-GNN),
not an original architecture of this course project. Cite:

- Hsu et al., [FIRE-GNN: Force-informed, Relaxed Equivariance Graph Neural Network
  for Rapid and Accurate Prediction of Surface Properties](https://doi.org/10.1002/aidi.202500162).
- Brandstetter, Hesselink, van der Pol, Bekkers and Welling,
  [Geometric and Physical Quantities Improve E(3) Equivariant Message Passing](https://arxiv.org/abs/2110.02905).
- Chanussot et al., [The Open Catalyst 2020 Dataset and Community Challenges](https://doi.org/10.1021/acscatal.0c04525).

The original upstream tree also contained an SE(3)-Transformer adaptation from
[e3nn's transformer example](https://docs.e3nn.org/en/stable/guide/transformer.html),
crediting Fuchs et al., [SE(3)-Transformers](https://arxiv.org/abs/2006.10503).
That unused alternative and generic relaxed-convolution models are removed from
this focused adsorption checkout. The current fixed-z option is **not** the
published FIRE-GNN learned relaxed-equivariance mechanism. Original surface
work-function weights are incompatible with the adsorption input/output layout.
Keep [LICENSE.md](LICENSE.md) and [CITATION.cff](CITATION.cff) with the code.
LICENSE.md preserves both the original root MIT notice and the distinct notice
previously stored in FIRE/LICENSE. CITATION.cff retains the upstream software
citation metadata; it does not attribute this course-project adaptation to the
upstream authors.

## Files kept

| File or directory | Purpose |
| --- | --- |
| FIRE/run_adsorption.py | Train a model or predict from a saved checkpoint |
| FIRE/prepare_oc20.py | Convert official OC20 LMDB subsets to JSON |
| FIRE/prepare_force_comparison.py | Validate and copy the fixed 1000/200/200 splits |
| FIRE/add_mlip_forces.py | Add initial-geometry force arrays with an ASE calculator |
| FIRE/mlip_calculators.py | Lazy Orb-v3 force calculator |
| FIRE/summarize_z_axis.py | Summarize four saved z-reference variants |
| FIRE/summarize_adsorbate_pooling.py | Summarize four saved pooling variants |
| FIRE/Dataset/ | Record checks, graph construction and elemental features |
| FIRE/Model/ | Active SEGNN/SEMPNN architecture and required layers |
| FIRE/configs/adsorption.json | General model defaults; not the completed-run settings |
| FIRE/tests/ | Unit regressions and small real-data integration checks |

The old Mac-trial helper is obsolete because its runs were removed and shared
inputs now have a canonical location. Slurm templates are stored under adsorption_runs/. The new configurable
submit_force_inputs.sbatch uses the current source layout; see the force-choice
section below. Other retained launchers may contain historical paths.

## Environment

The configured Perlmutter environment is `fire_gnn`:

```bash
module load conda
conda activate fire_gnn
python FIRE/run_adsorption.py train --help
```

Core dependencies are PyTorch, torch_geometric, torch_scatter, e3nn and pymatgen;
LMDB conversion additionally needs lmdb. See FIRE/requirements.txt. Its existing Orb force
inference stack uses: PyTorch 2.6.0,
orb-models 0.5.5, pymatgen 2024.5.1 and NumPy below 2. torch_scatter must match
the installed PyTorch/CUDA build. The training path can read precomputed forces
without importing Orb. Setup, force generation and GNN training all use fire_gnn. The result notebook also needs pandas and matplotlib.

## Data and splits

Canonical inputs live in `adsorption_runs/shared_data/`: train.json,
validation.json, test.json and config.json. source_manifest.json records their
original paths and SHA256 hashes. The local split has 1000/200/200 structures;
validation and development-test data come from OC20 in-domain validation records,
not the hidden official benchmark. Overlapping sample/trajectory IDs are rejected.

Each record contains `id`, optional `group_id`, `material` (a pymatgen Structure),
a binary `adsorbate_mask` with one value per atom, and `adsorption_energy` in eV
(omitted for unlabeled prediction). OC20 tag 2 marks adsorbate atoms. IS2RE uses
initial `pos` and the supplied `y_relaxed` energy label; relaxed geometry/forces
must not become model inputs. Conversion preserves provided reference energies.
For S2EF energy-only data, use `--target y --referenced-energy`; for unlabeled
records, use `--target none`. This is not a full S2EF force-learning pipeline.

```bash
python FIRE/prepare_oc20.py /path/to/train/data.lmdb oc20_train.json \
  --target y_relaxed --limit 1000
python FIRE/prepare_oc20.py /path/to/val_id/data.lmdb oc20_val.json \
  --target y_relaxed --limit 200
```

The converter samples reproducibly (default seed 13). Keep official partitions
and OOD categories separate. JSON is suitable for small subsets; full OC20
training would require a scalable loader. The converter consumes trusted OC20
LMDB payloads containing Python pickles.

## Training, loss and checkpoint selection

Prepare a **fresh** comparison directory using the canonical splits:

```bash
python FIRE/prepare_force_comparison.py adsorption_runs/new_mse_comparison --loss mse
```

This prepares data/configuration only. Its base config retains historical small
trial defaults (five epochs, batch four); override those explicitly for training.
The general configs/adsorption.json has different defaults (100 epochs, width 64,
three layers, angular order two). Completed-run checkpoint configs are authoritative.

For a matched pair using the existing precomputed force data:

```bash
FORCES=adsorption_runs/cpu_force_full_60641594
for mode in baseline forces; do
  extra=()
  if [[ "$mode" == forces ]]; then extra=(--use-mlip-forces --force-scale 1.0); fi
  python FIRE/run_adsorption.py train \
    --config adsorption_runs/new_mse_comparison/config.json \
    --train "$FORCES/train_forces.json" \
    --validation "$FORCES/validation_forces.json" \
    --test "$FORCES/test_forces.json" \
    --output "adsorption_runs/new_mse_comparison/$mode" --device cpu \
    --epochs 1000 --batch-size 16 --patience 30 --num-workers 4 \
    --cache-dir adsorption_runs/new_mse_comparison/graph_cache "${extra[@]}"
done
```

The baseline ignores force arrays. To change loss, prepare with `--loss mae` or
set `loss` to `mae`/`mse` in the config. MSE is the current preferred objective,
but a config without a loss field defaults to MAE. Training normalizes targets
using training statistics only. MSE runs select checkpoints and early-stop on
validation MSE; MAE runs use validation MAE. Patience zero disables early stopping.
The best checkpoint is evaluated on the optional labeled test split. Existing
nonempty output directories are rejected.

Each trained model saves best.pt, history.json and, when test data are supplied,
test_metrics.json. MAE and RMSE are in eV; MSE is in eV squared. Training history
metrics accumulate during updates; validation/test evaluate fixed parameters.

## MLIP force inputs

The current factory chooses Orb-v3 direct OMat checkpoint dated 20250404 through
the pinned orb-models 0.5.5 API. The FIRE-GNN paper names Orb-v3 but does not
identify this exact variant; this is a project choice, not an exact reproduction.
Its accuracy on adsorbate/slab systems must be evaluated rather than assumed.

Generate each split once on an allocated compute node:

```bash
python FIRE/add_mlip_forces.py adsorption_runs/shared_data/train.json train_forces.json \
  --calculator-factory mlip_calculators:orb_v3 \
  --calculator-kwargs '{"device":"cpu"}' --slab-vacuum 15 \
  --source 'orb-models-0.5.5/orb-v3-direct-inf-omat-20250404'
```

Repeat for validation/test, preserving IDs and order. Forces are unconstrained,
initial-geometry vectors in eV/Angstrom; no energy labels are sent to Orb and no
relaxation is performed. For xy-periodic slabs, the calculator uses full PBC with
an adjusted perpendicular cell and a 15-Angstrom vacuum gap, retaining relative
coordinates. Original GNN geometry is unchanged; the calculator cell is recorded
in metadata. Orb 0.5.5 does not support partial PBC. Slab atoms receive tag 0 and
adsorbates tag 2; the binary mask cannot reconstruct subsurface tags.

The graph uses the same [y,z,x] permutation for force vectors and spherical
harmonics. `--force-scale` is a positive scalar divisor (default 1 eV/Angstrom),
not an independent per-component normalization. Estimate any scale on training
records only. All records must have finite N-by-3 force arrays with consistent
source and initial-geometry metadata. Existing force output files are rejected;
an interrupted partial JSON must not be used for training.

The factory uses a process-local /tmp cache to avoid shared-filesystem locking
errors. ORB_CACHE_DIR can override it. ORB_WEIGHTS_PATH, or the factory's
`weights_path` argument, reuses a locally downloaded checkpoint. Ensure weights
are available before a compute job when external network access is restricted.
Include force-generation cost in end-to-end comparisons even when arrays are
reused by subsequent experiments.

## Fixed-z reference

Set `"z_symmetry_breaking": true` in the config or pass
`--z-symmetry-breaking` to the training command. The default is off. A fixed
Cartesian +z polar vector is appended per atom, represented as [0,1,0] in the
model's permuted frame. Force and normal vectors remain separate irreps.

The option preserves rotations about z and translation invariance, but allows
sensitivity to tilts and reversal of z. It adds input parameters. Some lattice
a-b normals are tilted relative to Cartesian z: minimum absolute normal/z cosines
are approximately 0.724/0.742/0.797 for train/validation/test. A coordinate-axis
benefit is not automatically a physical surface-normal effect. Any alignment
must rotate geometry, cell and force vectors consistently.

For a four-way ablation, train baseline, forces, baseline_z and forces_z with
otherwise matched settings. Summarize a completed root with:

```bash
python FIRE/summarize_z_axis.py adsorption_runs/z_axis_mae_seed13_60660414
```

This writes z_comparison.json. It does not launch training.

## Adsorbate-focused pooling

Use `--pooling adsorbate_mean` or `"pooling": "adsorbate_mean"`. All atoms and
edges remain in message passing and prepool layers. The readout averages only
selected adsorbate embeddings, dividing by the adsorbate count in each graph.
Surface atoms still affect predictions through messages. Batch normalization is
unchanged. Each graph must contain at least one adsorbate atom.

This changes neither parameter count nor geometric symmetries. Other supported
readouts are mean/add/max; the supplied configs use mean. Z inputs and force
inputs can be enabled independently or together with adsorbate pooling. Compare
baseline/forces with mean pooling against baseline_adsorbate/forces_adsorbate,
keeping all other settings matched. Summarize a completed four-way root with:

```bash
python FIRE/summarize_adsorbate_pooling.py \
  adsorption_runs/adsorbate_pooling_mae_seed13_60660713
```

This writes pooling_comparison.json and pooling_comparison.csv. Positive error
reduction favors the modification. Current saved architecture comparisons used
MAE; reporting MSE as an evaluation metric does not make them MSE-trained.

## Prediction and existing checkpoints

```bash
python FIRE/run_adsorption.py predict \
  --checkpoint adsorption_runs/adsorbate_pooling_mae_seed13_60660713/baseline_adsorbate/best.pt \
  --data adsorption_runs/shared_data/test.json \
  --output adsorption_runs/pooling_predictions.json --device cpu
```

Supply force-enriched records when the checkpoint expects force inputs. Prediction
reconstructs the graph and architecture from saved config/normalization, including
z and pooling flags. Existing baseline, force, z and pooling checkpoints remain
compatible; no original experiment source copies are required.

## Running on Perlmutter

Use a compute allocation for full training/force generation. The CPU comparisons
used 16 allocated cores, eight PyTorch/OMP/MKL threads and four graph workers:

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
```

Use `--device cuda` for one GPU. Multi-GPU training uses torchrun with
`--device cuda` and NCCL. Batch size is per rank and global batch scales with rank
count. Evaluation does not pad duplicate samples. CPU/GPU and distributed batch
normalization need not yield identical trajectories; runtime comparisons require
matched resources, cache conditions and data. First epochs include startup and
graph building, so report later-epoch medians and full wall time separately.

## Validation and caches

```bash
python -m unittest discover -s FIRE/tests -v
python FIRE/tests/smoke_z_axis.py
python FIRE/tests/smoke_adsorbate_pooling.py
```

Unit checks cover records/splits, graph features, force preprocessing, pooling,
rotational/axial behavior, loader workers, checkpoint reload and early stopping.
Real-data smoke scripts use four training, two validation and two test structures
for one epoch; their JSON evidence is not an accuracy benchmark. The pooling
smoke also covers both pooling and z together.

Graph caches store tensors/scalars and are versioned by featurization settings.
They can be regenerated and are not needed to read results. Clear caches only
after users of those paths have finished. Preserve checkpoints, metrics,
histories, data, provenance and the comparison notebook. Use
adsorption_runs/mlip_gnn_comparison.ipynb for current results; the saved sweeps
and architecture tests are mostly single-seed development experiments. Choose
settings on validation data, repeat paired seeds, and reserve a fresh holdout for
final reporting. The current development-test set has been inspected repeatedly.

## Saved experiments and analysis

| Repository path | Purpose |
| --- | --- |
| adsorption_runs/cpu_force_full_60641594/ | Matched MSE baseline / MLIP |
| adsorption_runs/cpu_force_mae_60660105/ | Matched MAE baseline / MLIP |
| adsorption_runs/sweep_*/ | Parameter-sweep baseline / MLIP pairs |
| adsorption_runs/z_axis_mae_seed13_60660414/ | Completed fixed-z ablation |
| adsorption_runs/adsorbate_pooling_mae_seed13_60660713/ | Completed pooling ablation |
| adsorption_runs/shared_data/ | Canonical inputs and original provenance |
| [Result notebook](adsorption_runs/mlip_gnn_comparison.ipynb) | Current tables and plots |
| adsorption_runs/mlip_analysis_figures/ | Exported analysis artifacts |

The old original/non-MLIP trials, mini and smoke run directories were removed.
The non-force baseline inside each matched comparison is retained because it is
needed to measure the force effect. Source smoke-test utilities remain useful.
Checkpoint, history, metric and input files are preserved. The cleanup manifests
in adsorption_runs record removed paths and retained-file hashes. Original split
manifests preserve historical paths as provenance, while current preparation and
notebook checks use shared_data/. Regenerable graph and Python caches were cleared
after the architecture jobs completed.

## Submission templates and parameter sweeps

Existing submission templates are stored in adsorption_runs/*.sbatch and
adsorption_runs/sweeps/. They were relocated from FIRE; some retain references
to the previous layout or obsolete helpers. Check those paths before submission.
The explicit training commands earlier in this guide use the retained Python
entry points and do not depend on the old launcher layout.

Each sweep is intended to train a matched baseline/force pair. The saved sweep
settings are:

| Variant | Change from control |
| --- | --- |
| control | Learning rate 0.001, weight decay 0.00001, width 32, two graph layers |
| lr_low | Learning rate 0.0003 |
| lr_high | Learning rate 0.003 |
| wd_strong | Weight decay 0.001 |
| width16 / width64 | Hidden features 16 / 64 |
| depth3 | Three graph layers |

The historical sweep defaults were MAE loss, batch 16, seed 13, force scale 1,
1000 maximum epochs, patience 30, four workers, 16 CPU cores and six hours for
the whole pair. LOSS, SEED, EPOCHS, PATIENCE, BATCH_SIZE, WORKERS, FORCE_SCALE and
FORCE_DIR were environment overrides. The completed control/learning-rate/
weight-decay sweeps used MAE, despite MSE being the preferred next objective.
The baseline and force model should share data and all non-force settings.

Precomputed forces can be validated and reused across configurations; their
original generation time remains part of an end-to-end cost assessment.
Run directories preserve config.json, optional sweep.json/selection.json,
model subdirectories and timing/summary JSONs when available. Use a common
validation metric across losses and compare it at the checkpoint actually
selected by each objective. Repeat finalists with paired seeds 13, 29 and 47;
never select settings by the smallest observed test error.

## Force locality and pooling controls

The built-in force generator is **Orb-v3 direct OMat**, using the existing
orb-models 0.5.5 installation in fire_gnn. It supplies force vectors to your
SEMPNN/SEGNN; it does not replace the GNN or perform geometry relaxation.
The custom ASE `--calculator-factory module:function --source ...` interface
remains available. No additional MLIP environment is needed.

### Generate forces once, on the full initial structure

```bash
module load conda
conda activate fire_gnn
python FIRE/add_mlip_forces.py adsorption_runs/shared_data/train.json \
  adsorption_runs/new_orb_forces/train_forces.json \
  --mlip orb --device cpu --slab-vacuum 15
```

Repeat for validation/test when generating a new source. Built-in Orb provenance
is recorded automatically; a custom weights_path or ORB_WEIGHTS_PATH is recorded
with its file hash. Existing arrays under cpu_force_full_60641594 can be reused.
Generation evaluates the full slab-plus-adsorbate geometry, preserving atom order
and input records. Force localization happens later in GNN featurization.

### Choose one force-input mode

| `--force-mode` | Input to the GNN |
| --- | --- |
| none | No force-vector channels or force inference |
| adsorbate | Adsorbate force vectors; other atoms' force features set to zero |
| adsorbate_nearby | Adsorbate plus substrate atoms within the specified radius |
| all | Force vectors on every atom |

`--force-radius` defaults to 3 Angstrom. Distances honor original lattice/PBC
and do not depend on graph-neighbor truncation. Nearby substrate means atoms
within that distance of any adsorbate atom, not a separately tagged top layer.
No atoms or edges are removed. Messages can propagate beyond the input region.
`--force-scale` is a positive scalar divisor, default 1 eV/Angstrom.

The same fields can be set in JSON as force_mode, force_radius and force_scale.
Explicit force_mode takes precedence over the old use_mlip_forces flag.
`--use-mlip-forces` remains an alias for `--force-mode all`. Old checkpoints retain
their original force/no-force interpretation. Checkpoints save these settings;
prediction applies them automatically. Graph cache keys distinguish modes/radii.

### Turn adsorbate pooling on or off

Use `--adsorbate-pooling` or `--no-adsorbate-pooling`. These select adsorbate mean
or global mean, independently of the force-input region and fixed-z option.
The existing `--pooling mean|add|max|adsorbate_mean` and JSON pooling field remain
supported. Select one pooling CLI interface per command; without an override,
the config determines pooling.

## Short Orb experiment submissions

There are **eight configurations** in adsorption_runs/mlip_sbatch:

- none_pool_on.sbatch and none_pool_off.sbatch;
- orb_adsorbate_pool_on.sbatch and orb_adsorbate_pool_off.sbatch;
- orb_adsorbate_nearby_pool_on.sbatch and orb_adsorbate_nearby_pool_off.sbatch;
- orb_all_pool_on.sbatch and orb_all_pool_off.sbatch.

Each file runs only its selected configuration. Adsorbate forces do not launch
an additional no-force model. All new runs default to MSE, seed 13, batch 16,
1000 maximum epochs, patience 30, four workers, 16 CPU cores and six hours.
Nearby radius defaults to 3 Angstrom. Pooling on means adsorbate mean; off means
global mean. Notifications use kaiyue_peng@berkeley.edu and mail-type ALL.

```bash
# Recommended: check for reusable completed results before queueing:
bash adsorption_runs/mlip_sbatch/orb_adsorbate_pool_on.sbatch
# Submit/check all eight:
for file in adsorption_runs/mlip_sbatch/*.sbatch; do bash "$file"; done
# Direct Slurm submission also works:
sbatch adsorption_runs/mlip_sbatch/orb_adsorbate_pool_on.sbatch
```

The default no-force/global-mean and all-force/global-mean MSE cases already
have complete results in cpu_force_full_60641594/baseline and forces. They are
reused if the recorded code/input hashes and settings match, leaving **six new
training runs**. Changing seed, loss or other relevant controls disables that
reuse. REUSE_COMPLETED=0 deliberately requests a fresh run. cases.json lists
the eight cases and the two verified reuse candidates; it is not an exhaustive
search of every prior experiment.

run_case.sh fixes each preset's choices and reuses existing Orb arrays by default.
FIRE/run_force_job.sh holds shared setup; submit_force_inputs.sbatch is a short
generic entry point. The shared runner explicitly loads conda and activates
`${GNN_ENV:-fire_gnn}`. FIRE_GNN_REPO identifies the source checkout independently
of Slurm's spool-copy location. Python paths and Dataset imports use that root.
Orb preprocessing stays in `fire_gnn`; optional UMA preprocessing uses the separate
`${UMA_ENV:-gnn_uma}` environment before returning to `fire_gnn` for training.

Both bash and direct sbatch assign a unique folder under adsorption_runs named
force_orb_<mode>_<pooling>_seed<seed>_<unique-id>. It contains job-<jobid>.out/.err,
config.json, experiment.json, copied data/, model/ with best.pt/history/test
metrics, and an optional graph_cache/. With direct sbatch, the shared runner
redirects streams into that folder before setup; with bash, Slurm opens them
there. Reused completed cases create no new training outputs.

Environment overrides: GNN_ENV, LOSS, SEED, EPOCHS, BATCH_SIZE, PATIENCE, WORKERS,
FORCE_RADIUS, FORCE_SCALE, FORCE_DIR, CALCULATOR_KWARGS, WALLTIME and
REUSE_COMPLETED. To use custom Orb weights, supply calculator kwargs or
ORB_WEIGHTS_PATH and the desired matching FORCE_DIR, if any. A supplied force
directory is validated against original records and the Orb source identifier.
Otherwise generation runs once per split inside that job. Force generation is
skipped in none mode. Tests cover periodic localization, cache separation,
legacy checkpoints, pooling/z combinations and offline Orb API wiring.

PREPARE_ONLY=1 verifies data/config setup without inference or training. Supply
an explicit FIRE_FORCE_RUN_DIR for that check outside Slurm. Historical failed
job folders and their logs are retained as evidence; removing an unsupported
backend from source does not delete old run records.

## More-data experiments

Six training jobs: 3,000 / 10,000 training structures × none / adsorbate / all forces.
All use adsorbate mean pooling, MSE, seed 13, batch 16, learning rate 0.001,
hidden size 32, two graph layers, 1,000 maximum epochs and early stopping patience 30.
The existing validation/test sets (200 each) stay fixed; they are development splits,
not an independent official OC20 test benchmark. Initial geometries predict y_relaxed.

Submit all six training jobs and one shared force-preparation job:

```bash
bash /pscratch/sd/k/kaiyuep/ChemE239/Project/project/FIRE-GNN/adsorption_runs/more_data_sbatch/submit_all.sh
```

Alternatively submit any `n*.sbatch` with sbatch. An individual Orb job will build
or wait for the same shared cache automatically. Do not submit both ways, which
would repeat training. No baseline is automatically trained by an Orb job.

Each job activates `fire_gnn` via `module load conda; conda activate fire_gnn`.
Requests: shared CPU, 16 cores, 24 hours, account m4868, email notifications ALL.
A 10k run may require more time; review logs before resubmitting interrupted work.

Results are in `adsorption_runs/more_n<SIZE>_<MODE>_adsorbate_mean_seed13_<unique>/`.
Inside: `model/`, `config.json`, `experiment.json`, `data/`, `graph_cache/`,
`job-<JOBID>.out`, `job-<JOBID>.err`. Preparation logs are in
`adsorption_runs/more_orb_preparation_<unique>/`. Slurm initially uses /dev/null;
the scripts redirect logs into the run directory immediately.

Shared datasets live in `adsorption_runs/more_data/{3000,10000}/` and provenance
is recorded in `more_data/manifest.json`. The original 1k is preserved exactly;
3k is a subset of 10k. `prepare_data.py` regenerates and verifies these datasets.
Orb-v3 forces are calculated once per new structure with the existing calculator,
15 Å slab vacuum and no relaxation. Existing exact-match 1k/validation/test forces
are reused. Restartable 100-record batches are in `more_data/orb_cache/`.
Four force-enabled models reuse identical arrays; masks are applied by the GNN.

## 10k no-force architecture tests

Three isolated tests are configured in `adsorption_runs/adsorption_runs_extend_sbatch/`:

| Preset | Pooling | Graph layers | Hidden features |
| --- | --- | --- | --- |
| `local_pooling.sbatch` | Mean over adsorbate + atoms within 3 Å of any adsorbate atom | 2 | 32 |
| `deeper_4layers.sbatch` | Adsorbate mean | 4 | 32 |
| `wider_64features.sbatch` | Adsorbate mean | 2 | 64 |

Local pooling uses initial-geometry periodic distances, counts each unique atom
once and retains the full message-passing graph. It is selected with
`pooling: local_mean` and `local_pool_radius: 3.0` in the JSON configuration.
The original `adsorbate_mean` remains available. Pooling masks are cached
separately from the unchanged baseline graphs, so old checkpoints remain usable.

All three use the same nested 10k training set, fixed 200/200 development splits,
MSE, no forces, seed 13, batch 16, learning rate 0.001, weight decay 1e-5,
1,000 maximum epochs and early stopping patience 30. No relaxation or force
preprocessing is run. Compare validation MSE against the existing 10k no-force
baseline (0.955165 eV²), rather than choosing configurations by test performance.

Submit once using:

```bash
bash /pscratch/sd/k/kaiyuep/ChemE239/Project/project/FIRE-GNN/adsorption_runs/adsorption_runs_extend_sbatch/submit_all.sh
```

Individual `.sbatch` files also accept `sbatch`. Both submission paths activate
`fire_gnn`, allocate 16 CPU cores and a 24-hour walltime, and enable ALL email
notifications. Eight PyTorch threads and four graph workers retain the proven
CPU configuration. Runtime depends on early stopping and model size; a
24-hour allocation is a limit, not a predicted duration or a forced minimum.

Each run creates `adsorption_runs/extend_<CASE>_n10000_seed13_<unique>/`, containing
`data/`, `config.json`, `experiment.json`, `model/`, and `job-<JOBID>.out/.err`.
The scripts reuse the existing no-force baseline's on-disk graph cache;
local-pooling graphs receive separate cache keys in that directory.
`PREPARE_ONLY=1 EXTEND_RUN_DIR=<empty-directory> bash
adsorption_runs/adsorption_runs_extend_sbatch/run_case.sh <CASE>` validates data/configuration
without training. Submitting both individual presets and submit_all repeats jobs.

## Optional UMA force inputs

`FIRE/add_mlip_forces.py --mlip uma` uses UMA-S `uma-s-1p2p1` with task `oc20`.
The default inference setting is `batch`, suitable for changing structures and
compositions without a fixed-composition compiled model. Override with
`--calculator-kwargs '{"model_name":"uma-s-1p2p1","task_name":"oc20","inference_settings":"batch"}'`.
The adapter lazily imports FAIRChem; Orb remains the default and its pinned
packages and existing presets are unchanged. No FAIRChem dependency was added
to `FIRE/requirements.txt` or installed in `fire_gnn`.

Use the existing separately installed `gnn_uma` environment for force generation:

```bash
module load conda
conda activate gnn_uma
python FIRE/add_mlip_forces.py INPUT.json OUTPUT_forces.json --mlip uma --device cpu
```

Run real inference on an allocated compute node. CPU and CUDA can be selected
in this preprocessing command. This path reads the existing ordered structure
JSON directly using ASE and NumPy, so it does not require pymatgen or PyG in
`gnn_uma`. UMA retains the original lattice, positions, atom order and partial
PBC; unlike the Orb-specific runner path, it does not apply a vacuum-cell
conversion. No relaxation is performed and labels are never passed to UMA.
All forces are saved, with package/model/task/inference provenance; GNN
`force_mode=adsorbate` zeros only substrate force features during training.

The shared CPU launcher accepts `MLIP=uma` and switches environments explicitly:

```bash
# Template for a full new 10k run; do not run until a small inference check passes.
MLIP=uma UMA_ENV=gnn_uma GNN_ENV=fire_gnn FORCE_MODE=adsorbate \
POOLING=adsorbate_mean LOSS=mse WALLTIME=24:00:00 \
SHARED_DATA_DIR="$PWD/adsorption_runs/more_data/10000" \
bash FIRE/run_force_job.sh
```

Old Orb-only preset directories remain Orb-only. Use the generic launcher for
UMA. `FORCE_DIR` may supply previously generated UMA arrays only if their
geometries, labels and sample IDs exactly match the selected inputs. Run results
and logs go into `adsorption_runs/force_uma_<mode>_<pooling>_seed<seed>_<unique>/`.

The UMA weights require access to <https://huggingface.co/facebook/UMA> and Hugging
Face authentication in your account. Environment/API/offline tests do not verify
checkpoint access or numerical inference accuracy. Do not supply tokens in chat.
Installation and inference API references:
<https://github.com/facebookresearch/fairchem/blob/main/docs/core/install.md> and
<https://facebookresearch.github.io/fairchem/ase-calculator/>.

### UMA submission stages

`adsorption_runs/uma_sbatch/` contains three CPU presets:

| File | Work | Walltime |
| --- | --- | --- |
| `01_check_uma.sbatch` | Real UMA forces for two training structures | 1 hour |
| `02_prepare_uma_forces.sbatch` | UMA forces for 10,000 train + 200 validation + 200 test structures | 24 hours |
| `03_train_uma_adsorbate_10k.sbatch` | Existing GNN, adsorbate-only UMA force inputs, adsorbate pooling, MSE | 24 hours |

Each allocates 16 CPUs and sends ALL notifications to kaiyue_peng@berkeley.edu.
Force generation uses `gnn_uma`; GNN training uses `fire_gnn`. The training
architecture, seed 13, batch 16, 1,000-epoch cap and patience 30 match the earlier
10k force comparison. No no-force baseline is trained automatically.

Before submitting, obtain access to https://huggingface.co/facebook/UMA and run
`module load conda`, `conda activate gnn_uma`, then `hf auth login` interactively.
An HTTP 401 access check means real inference has not yet been verified. Never
put tokens in these scripts or pass them in chat. Existing account authentication
is used; checkpoint files are cached on scratch in `adsorption_runs/uma_data/model_cache`.

Submit each stage with `sbatch` when ready, or explicitly run:

```bash
bash adsorption_runs/uma_sbatch/submit_pipeline.sh
```

The optional launcher submits three jobs with afterok dependencies: check →
preprocessing → training. Do not also submit the same stages separately.
Preprocessing saves restartable 100-record batches and reuses the two successful
check predictions. A time-limited preprocessing job may be resubmitted; rerun
failed/dependency-cancelled training only once preprocessing completes.

Forces are in `adsorption_runs/uma_data/forces/{train,validation,test}_forces.json`,
with a completion manifest containing input/output hashes and model provenance.
Training verifies these hashes and matching structures before use. Each stage
creates `adsorption_runs/uma_<check|prepare|train>_10k_seed13_<unique>/` with
`job-<JOBID>.out/.err`; check results and timing are in the check folder, while
model checkpoints/history/metrics and copied inputs are in the training folder.
Walltime limits do not guarantee actual completion times. Preparation can resume
completed batches, but GNN training currently starts fresh when resubmitted.

`PREPARE_ONLY=1 UMA_STAGE_RUN_DIR=<empty-directory> bash
adsorption_runs/uma_sbatch/run_stage.sh <check|prepare|train>` validates setup
without inference or training. Offline tests use mock forces and do not establish
UMA numerical accuracy or replace the real two-structure inference check.

UMA named-checkpoint downloads require `FAIRCHEM_CACHE_DIR` to be set before
importing FAIRChem (2.25.0 does not use the predictor's cache_dir argument for
that download). The UMA stage launcher explicitly sets it to the scratch
model cache to avoid home-filesystem flock errors (Errno 524). Hugging Face
authentication remains in its existing location; HF_HOME is not changed.

## Complete neighbor shells

Graph construction now treats `max_neighbors` as a target count: after finding
that neighbor's distance, it includes the entire shell within
`neighbor_tie_tolerance` (default `1e-5` Å). The radius cutoff remains unchanged.
This avoids selecting different members of a crystalline shell after rotation,
translation, or atom reordering. The actual neighbor count may exceed the target.
The cache version is `complete-neighbor-shells-v2`, and the tolerance is part of
the cache key and newly trained checkpoint configuration.

The matched baseline/ORB/UMA retraining launchers are in
`adsorption_runs/neighbor_shell_sbatch/`; prepared runs and job manifests are under
`adsorption_runs/neighbor_shell_retrain/`. Each run uses a frozen FIRE source
snapshot and its own fresh graph cache. Existing saved metrics remain historical
results from the old strict-neighbor selection. Re-evaluating an old checkpoint
with the corrected graph builder changes its input graph and is a diagnostic,
not a reproduction of those historical metrics. Corrected symmetry evidence is
in `adsorption_runs/setup_audit/corrected_verification/`.

## Force-aware residual architecture pilot

Set `architecture` to `force_residual_v1` to use the new radial geometry model
with an explicit force-dependent energy correction. It adds radial distance
features and invariant force/edge projections while retaining equivariant polar
force vectors. The correction is initialized to zero and vanishes exactly for
zero force inputs. Newly prepared controlled pilots use a training-only force
RMS scale and an independent loader RNG; legacy configurations still select
SEMPNN. The model predicts relaxed energy; it is not a conservative force field.

Implementation: `FIRE/Model/ForceResidualGNN.py`. Design details, verification,
job IDs, and the six matched validation-only pilots are documented in
[force-residual pilot instructions](adsorption_runs/force_residual_sbatch/README.md).
The pilot snapshots are frozen separately from the active source tree.
