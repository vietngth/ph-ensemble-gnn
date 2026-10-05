# FLAIRS baselines: TISER-GCN SFA and TISER-TGCN

Schwenke et al. 2023 (FLAIRS-36), "Identifying Informative Nodes in Attributed Spatial Sensor Networks using
Attention for Symbolic Abstraction in a GNN-based Modeling Approach". We use the authors' code
[lschwenke/GraphNodeAttention](https://github.com/lschwenke/GraphNodeAttention) at commit
`650079bdf506fd37f478ed91f10d9999f35f2e23` (MIT license, `LICENSE-GraphNodeAttention`). Their code is
used unmodified. This folder holds a logging patch, flattened copies of their configurations and our run
scripts.

| File | Purpose |
|---|---|
| `eqModelTrain_perim.patch` | Unified diff `eqModelTrain.py` -> `eqModelTrain_perim.py`. It changes logging only: each fold's per-IM test MSE/RMSE/MAE goes into a `perim` field of the results pickle, and the pickle is written to `results_perim/` instead of `results/`. Training, evaluation and SFA are unchanged. |
| `configs/tgcn_e200.yaml` | TISER-TGCN at 10 s. |
| `configs/gcn_sfa_e200.yaml` | TISER-GCN SFA at 10 s. |
| `configs/{tgcn,gcn_sfa}_e200_il{400..900}.yaml` | The same two models with an input window of 4-9 s. They differ from the 10 s configs only in `data.iLen`. |
| `run_one.sh` | One run with the authors' `eqModelTrain.py`: `run_one.sh <config> <network> <seed>`. Results go to `results/`. |
| `run_one_perim.sh` | The same run with `eqModelTrain_perim.py`. Results go to `results_perim/`. |
| `flairs.sbatch` | SLURM wrapper that runs a run list `LANES` at a time on one GPU, with requeue on preemption (`USR1` 15 min before the end). |
| `runs_perim_e200_{1,2}.txt` | The 40 runs at 10 s: 2 models x 2 networks x seeds 1-10. |
| `runs_windows_e200_{1,2}.txt` | The 240 window runs: 2 models x 6 windows x 2 networks x seeds 1-10. |
| `LICENSE-GraphNodeAttention` | The upstream MIT license, which covers the patch context lines and the configuration values derived from their files. |

`network1` = Central Italy (CI), `network2` = Central-West Italy (CW).

**Configurations.** Each config is a single-run version of one of the authors' seml files, passed to their
sacred script with `with configs/<name>.yaml`. TGCN comes from `eqModelTransformer.yaml` (`modelN: transformer`)
and GCN SFA from `eqModelCnn.yaml` (`modelN: bloem`). The fixed values are the authors': `seed_value 1`,
`patience 15`, `epochs 200`, the dropouts, the filters and the kernel. The grid choices are
`doStations: true` for TGCN, `nbins 6` and `ncoef 125`. The network and the seed come from the command line.

## Setup

```bash
git clone https://github.com/lschwenke/GraphNodeAttention.git
cd GraphNodeAttention && git checkout 650079b
md5sum eqModelTrain.py          # must be d048a65e69ec6b6b648ac3ae186e3519 (unmodified authors' file)
cp eqModelTrain.py eqModelTrain_perim.py
patch eqModelTrain_perim.py < $REPO/baselines/flairs/eqModelTrain_perim.patch
md5sum eqModelTrain_perim.py    # 42444c056dd3a64d8c46e673ca175ad9 (the file that produced the paper numbers)
mkdir -p configs results_perim
cp $REPO/baselines/flairs/configs/*.yaml configs/
cp $REPO/baselines/flairs/{run_one.sh,run_one_perim.sh,flairs.sbatch,runs_perim_e200_*.txt,runs_windows_e200_*.txt} .
```

`$REPO` is your checkout of this repository. The clone is the work directory, as in our runs.

**Data.** Prepare it as upstream does. The clone's `data/` already holds `targets.npy`, `station_coords.npy`,
`minmax_normalized_laplacian.npy`, `meta.npy` and `othernetwork/...`. Add the waveforms from
[zenodo.org/record/5767221](https://zenodo.org/record/5767221) as `data/inputs_ci.npy` and
`data/othernetwork/inputs_cw.npy`. Expected md5: `inputs_ci.npy` 541ce1b536a5add6681c8e1f091e2b46 and
`inputs_cw.npy` da6da738000384ef9591d2ee0b28d1be. These are the same files that our models and JOZ-CNN read.
The SFA transform is computed inside `eqModelTrain.py` and fitted on the training part of each split.

**Environment.** Use `flairs-env`: Python 3.9.13, TensorFlow 2.6.2 (conda-forge, CUDA 11.0), numpy 1.19.5,
scikit-learn 1.2.1, pyts 0.12.0, scipy 1.9.1, spektral 1.2.0, sacred 0.8.4 and seml 0.3.7. See
`../env/flairs-env.yml`; the full package list is in `../env/flairs-env.conda-export.txt`. These are the
upstream README versions, plus scikit-learn 1.2.1, which upstream does not pin. The same environment
(`experiments/build_sfa_inputs.py`) builds the SFA inputs for our own SFA arms.

## Run

Environment variables: `FLAIRS_PYTHON` (python of flairs-env, required), and for the sbatch `LIST`, `LANES`,
`RUNNER` (`run_one.sh` by default) and optionally `FLAIRS_WORK` (default: the submit directory).

These are the commands used for the paper (d3 cluster; `-p main` and the resource flags are site-specific).
Run them from the clone:

```bash
mkdir -p slurm
# 10 s, per-IM metrics (40 runs) -> results_perim/
sbatch -J flairs_perim --array=1-2 -p main -c 12 --mem=64G --time=2-00:00:00 -o slurm/%x-%A_%a.out \
  --export=ALL,LIST=runs_perim_e200_{task}.txt,LANES=5,RUNNER=run_one_perim.sh,FLAIRS_PYTHON=$FLAIRS_ENV/bin/python flairs.sbatch
# windows 4-9 s (240 runs) -> results/
sbatch -J flairs_windows --array=1-2%1 -p main -c 12 --mem=64G --time=3-00:00:00 -o slurm/%x-%A_%a.out \
  --export=ALL,LIST=runs_windows_e200_{task}.txt,LANES=5,FLAIRS_PYTHON=$FLAIRS_ENV/bin/python flairs.sbatch
```

To run without SLURM: `FLAIRS_PYTHON=... xargs -P 5 -L 1 bash run_one_perim.sh < runs_perim_e200_1.txt`.

**Checkpoint clean-up.** `run_one.sh` and `run_one_perim.sh` end with `rm -f models/*.n<network>.rs<seed>.*`.
The authors' script saves the best model of each fold to `models/*.h5` with a Keras `ModelCheckpoint`, but it
never reads these files back. It predicts the test set with the in-memory model, and the only loading path
(`saves/test<k>`, `useSaves`) is disabled (`useSaves: false`). At about 3.4 GB per run, the checkpoints filled
the disk, so each run's checkpoints are deleted when the run ends. The pattern also matches the checkpoints of
a concurrent run with the same network and seed but another config. This is harmless for the same reason: the
files are written only, never read. Keep the clean-up when re-running.

## Outputs and how they map to the paper

Each run writes one pickle (written with `np.save`; load it with `np.load(f, allow_pickle=True)`). The pickle
holds the per-fold lists `mse`, `rmse` and `mae`, each averaged over the five IMs. With the patch it also holds
`perim`: 5 dicts `{fold, mse, rmse, mae}` with one value per IM. File names:
`results.n<network>.rs<seed>.ftTrue.s6.co125...stTrue.il<len>.pkl` for TISER-TGCN and
`results.mbloem.n<network>.rs<seed>.ftTrue.s6.co125...il<len>.pkl` for TISER-GCN SFA, where `<network>` is
`network1` or `network2`.

| Output | Paper |
|---|---|
| `results_perim/*.il1000.pkl` (field `perim`) | TISER-GCN SFA and TISER-TGCN rows of Table 3 (averages) and Tables 4, 6, 7 (per-IM MAE, MSE, RMSE); the TISER-GCN SFA column of Table 5; the 10 s points in Fig. 7; the 10 s MSE quoted in Appendix C. |
| `results/*.il{400..900}.pkl` (field `mae`, fold mean) | 4-9 s points of TISER-GCN SFA and TISER-TGCN in Fig. 7. |


**Splits.** `eqModelTrain.py` calls `train_test_split(..., test_size=0.2, random_state=seed)` with the same
inputs as our models, so the test events of each seed are identical to ours (183 CI and 54 CW events). The
validation folds use the TSER-GCN fold procedure (`k_fold_split`, seeded with `init.seed_value: 1`), as in
JOZ-CNN and `phgnn.data.tser_fold_indices`.

## Caveats

- `run_one.sh` with `runs_perim_e200_*.txt` writes the same runs to `results/`. Our first 10 s runs were made
  this way (identical lists `runs_repro_e200_*.txt`, not included). The paper uses only the `results_perim/`
  re-runs, which add per-IM metrics with unchanged training.
- 35 of the 80 `tgcn_e200_il{400..700}` window results come from an earlier, cancelled array job that used
  the same config files. Their logs were overwritten by skip-reruns (the authors' script returns "Already done"
  when the pickle exists). `logs/finished.txt` shows rc=0 for all of them.
- TF on GPU is not deterministic; the authors' config notes this too ("still something makes the run not
  deterministic"). Re-runs reproduce the numbers statistically, not bit for bit.
- The scripts differ from the files that were run only in their paths: the work directory and the python are
  now set through `FLAIRS_WORK`/`FLAIRS_PYTHON`, and `logs/` and `results_perim/` are created if they are
  missing. The arguments, the clean-up and the skip logic are unchanged.
- The PH-FLAIRS experiments (`ph_*` files in our working copy) are not in the paper and are not included.
