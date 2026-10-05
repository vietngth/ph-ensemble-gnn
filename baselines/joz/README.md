# JOZ-CNN baseline

JOZ-CNN (Jozinovic et al.) is run with the implementation of the TSER-GCN authors:
`main_cnn.py` of [StefanBloemheuvel/GCNTimeseriesRegression](https://github.com/StefanBloemheuvel/GCNTimeseriesRegression)
at commit `3da6cb6bc445e71440b0c460f3bf43bbe49dfe6b`. That repository has **no license**, so its code is not
copied here. This folder holds only our patch and our run scripts.

| File | Purpose |
|---|---|
| `main_cnn_window.patch` | Unified diff `main_cnn.py` -> `main_cnn_window.py`. It makes two changes: the input length comes from the `JOZ_WINDOW` env var (default 1000 samples = 10 s at 100 Hz, as upstream), and each fold's per-IM test metrics are appended to `per_im.csv`. Training and evaluation are unchanged. |
| `run_one_joz.sh` | One run: `run_one_joz.sh <network> <window> <seed>`, in its own directory `runs/<prefix>_<network>_w<window>_rs<seed>/`. A run that already has `githubresults.csv` is skipped. |
| `joz.sbatch` | SLURM wrapper that runs a run list `LANES` at a time on one GPU, with `--requeue`. |
| `runs_joz_perim.txt` | The 20 runs at 10 s (2 networks x seeds 1-10) used with `JOZ_PREFIX=joz_perim`. |
| `runs_joz_windows_{1,2}.txt` | The 140 window runs (2 networks x windows 400-1000 x seeds 1-10), split into two array tasks. |

`network1` = Central Italy (CI), `network2` = Central-West Italy (CW).

## Setup

```bash
git clone https://github.com/StefanBloemheuvel/GCNTimeseriesRegression.git
cd GCNTimeseriesRegression && git checkout 3da6cb6
cp main_cnn.py main_cnn_window.py
patch main_cnn_window.py < $REPO/baselines/joz/main_cnn_window.patch
md5sum main_cnn_window.py        # 41b3e7e852d1e1fd8f700375728d1ebf (the file that produced the paper numbers)
cp $REPO/baselines/joz/{run_one_joz.sh,joz.sbatch,runs_joz_*.txt} .
```

`$REPO` is your checkout of this repository. The clone is the work directory: `main_cnn_window.py`, the scripts,
the run lists, `logs/` and `runs/` all live side by side, as in our runs.

**Data.** This uses the upstream layout. The clone's `data/` already holds `targets.npy`, `station_coords.npy`
and `othernetwork/{targets.npy,station_coords.npy}`. Add the waveforms from
[zenodo.org/record/5767221](https://zenodo.org/record/5767221) as `data/inputs_ci.npy` and
`data/othernetwork/inputs_cw.npy`. The upstream README says `input_ci.npy`, but the code loads `inputs_*.npy`.
To read the data from somewhere else, for example the FLAIRS clone's `data/` (same files), set
`JOZ_DATA=/path/to/data`. Expected md5: `inputs_ci.npy` 541ce1b536a5add6681c8e1f091e2b46,
`inputs_cw.npy` da6da738000384ef9591d2ee0b28d1be, `targets.npy` 53edef0aeed08e24ff6c9167704951b8 (CI) /
b0ed16ef1680e77590fb5cb5bfb64ecb (CW). These are the same files that our models read.

**Environment.** Use `tser-env` = `flairs-env` plus matplotlib (`main_cnn.py` imports it): Python 3.9.13,
TensorFlow 2.6.2 (conda-forge, CUDA 11.0), numpy 1.19.5, scikit-learn 1.2.1, matplotlib 3.5.3. See
`../env/flairs-env.yml` and `../env/tser-env-extra.txt`. Upstream lists TF 2.8 / numpy 1.22; we used the FLAIRS
environment for both TF baselines.

## Run

Environment variables: `JOZ_PYTHON` (python of tser-env, required), `JOZ_PREFIX` (`joz` by default, or
`joz_perim`), `JOZ_DATA` (default `./data`), and for the sbatch `LIST`, `LANES` and optionally `JOZ_WORK`
(default: the submit directory).

These are the commands used for the paper (d3 cluster; `-p main` and the resource flags are site-specific).
Run them from the clone:

```bash
mkdir -p slurm
# 10 s, per-IM metrics (20 runs)
sbatch -J joz_perim -p main --gres=gpu:1 -c 12 --mem=64G --time=12:00:00 -o slurm/%x-%j.out \
  --export=ALL,LIST=runs_joz_perim.txt,LANES=5,JOZ_PREFIX=joz_perim,JOZ_PYTHON=$TSER_ENV/bin/python joz.sbatch
# window study (140 runs, two array tasks)
sbatch -J joz_windows --array=1-2 -p main -c 12 --mem=64G --time=3-00:00:00 -o slurm/%x-%A_%a.out \
  --export=ALL,LIST=runs_joz_windows_{task}.txt,LANES=5,JOZ_PYTHON=$TSER_ENV/bin/python joz.sbatch
```

To run without SLURM: `JOZ_PYTHON=... JOZ_PREFIX=joz_perim xargs -P 5 -L 1 bash run_one_joz.sh < runs_joz_perim.txt`.

## Outputs and how they map to the paper

Each run directory contains `githubresults.csv`, the upstream output. It has 5 rows, one per fold, with
mse, rmse and mae averaged over the five IMs. Upstream labels the rows PGV, PGA, ...; these labels are row names
only. The fold mean is the mean of the 5 rows. With the patch, the run directory also contains `per_im.csv`,
with rows `fold, im, mse, rmse, mae`. `im` 0-4 is the index of the last axis of `targets.npy`.

| Runs | Paper |
|---|---|
| `runs/joz_perim_network{1,2}_w1000_rs{1..10}/per_im.csv` | JOZ-CNN row of Table 3 (averages) and Tables 4, 6, 7 (per-IM MAE, MSE, RMSE). The generator averages the 5 folds per IM and needs exactly 25 rows. |
| `runs/joz_perim_network{1,2}_w1000_rs{1..10}/githubresults.csv` | 10 s point of JOZ-CNN in Fig. 7, and the 10 s MAE quoted in Appendix C. |
| `runs/joz_network{1,2}_w{400..900}_rs{1..10}/githubresults.csv` | 4-9 s points of JOZ-CNN in Fig. 7. |
| `runs/joz_network{1,2}_w1000_rs*` (in the window lists) | Not used: at 10 s the per-IM runs above are used everywhere. |


## Splits

The split for each seed is the same as for our models. `main_cnn.py` calls
`train_test_split(inputs, graph_features, targets, test_size=0.2, random_state=seed)`. `phgnn/data.py` calls
`train_test_split(inputs, targets, test_size=1-0.8, random_state=data_seed)`. Both give the same test size
(183 CI, 54 CW events) and the same shuffle, so the test events are identical. The validation folds come from
upstream `k_fold_split`, which applies `np.random.seed(1)`, one permutation of the training part and 5
contiguous chunks. `phgnn.data.tser_fold_indices(seed=1)` reimplements exactly this. The input files are the
same files (md5 above).

## Caveats

- The w < 1000 window runs were produced on 2026-10-03 with an earlier `main_cnn_window.py`, which had the
  `JOZ_WINDOW` change but not the per-IM logging. No copy of that version was kept. The logging lines do not
  touch training.
- `per_im.csv` is appended after each fold, but `githubresults.csv` is written only at the end of a run. A run
  that is killed part-way therefore leaves a partial `per_im.csv` and is re-run from scratch on requeue, which
  appends duplicate rows. Delete such a run directory before re-running it. None of the paper runs was affected: all 20 files have
  exactly 25 rows (5 folds x 5 IMs).
- TF on GPU is not deterministic: re-runs reproduce the numbers statistically, not bit for bit.
- `run_one_joz.sh` and `joz.sbatch` differ from the files that were run only in their paths: the work
  directory, the data link and the python are now set through `JOZ_WORK`/`JOZ_DATA`/`JOZ_PYTHON`
  (`JOZ_PYTHON` was called `FLAIRS_PYTHON`), and `logs/` is created if it is missing. The
  arguments, the run naming and the skip logic are unchanged.
