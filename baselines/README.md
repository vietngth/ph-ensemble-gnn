# Baselines

This folder makes the paper's baseline numbers reproducible: Tables 3-7, Fig. 7 (input window 4-10 s) and
Appendix C ("Reproduction of the baselines"). All baselines use the splits of our models: seed k fixes the
80/20 split, and the training part is divided into the same five folds (TSER-GCN fold procedure, fold seed 1).
Each runs with seeds 1-10, and none of them is tuned.

| Baseline | Where | Code |
|---|---|---|
| JOZ-CNN | [`joz/`](joz/README.md) | `main_cnn.py` of the TSER-GCN authors (GCNTimeseriesRegression @3da6cb6). That repo has **no license**, so we ship only a patch (`JOZ_WINDOW` input length and per-IM logging), our run scripts and the run lists. |
| TISER-GCN SFA, TISER-TGCN | [`flairs/`](flairs/README.md) | Schwenke et al. 2023 (GraphNodeAttention @650079b, MIT). The authors' `eqModelTrain.py` is used unmodified. We ship a logging-only patch (`eqModelTrain_perim.py`), the configs we used (200 epochs, patience 15; 10 s and 4-9 s windows), the run scripts, the run lists and the upstream license. |
| TSER-GCN | main package | PyTorch reproduction: `phgnn/models/baseline.py`, `configs/fast.yml` + `configs/experiments/repro/baseline.yml` |
| KIM-GNN | main package | Model copied from Kim et al.'s code with the TSER-GCN adaptations: `phgnn/models/kim_reference.py`, `configs/fast.yml` + `configs/experiments/repro/kim_gnn.yml` |

`env/` holds the TensorFlow environment that both TF baselines used:
- `flairs-env.yml`: Python 3.9.13, TF 2.6.2 (CUDA 11.0), numpy 1.19.5, scikit-learn 1.2.1, pyts 0.12.0, scipy 1.9.1.
- `flairs-env.conda-export.txt`: the full package list.
- `tser-env-extra.txt`: matplotlib 3.5.3 on top of flairs-env, for JOZ-CNN.

The PyTorch models (ours, TSER-GCN, KIM-GNN) use the main package environment (`pyproject.toml`).

## Quick path

1. Create the environments (`env/`).
2. Clone each upstream repo at the pinned commit and apply the patch. Each README checks the result by md5.
3. Copy the scripts, configs and run lists into the clone.
4. Add the zenodo waveforms (`inputs_ci.npy`, `inputs_cw.npy`) to `data/` as upstream describes.
5. Run the listed `sbatch` commands, or the `xargs` one-liners without SLURM.

Paths are set through environment variables:

| Variable | Used by | Meaning |
|---|---|---|
| `JOZ_PYTHON`, `FLAIRS_PYTHON` | run scripts, sbatch | Python of the TF environment (required). |
| `JOZ_WORK`, `FLAIRS_WORK` | sbatch | Work directory (the patched clone). Default: the submit directory. |
| `JOZ_DATA` | `run_one_joz.sh` | Data folder. Default: `./data`. |
| `JOZ_PREFIX` | `run_one_joz.sh` | `joz` (window study) or `joz_perim` (10 s per-IM runs). |
| `LIST`, `LANES`, `RUNNER` | sbatch | Run list (`{task}` = array task id), parallel runs per GPU, FLAIRS runner script. |

## Output -> paper

| Output | Paper |
|---|---|
| JOZ `runs/joz_perim_*_w1000_rs*/per_im.csv` | JOZ-CNN row, Tables 3, 4, 6, 7 |
| JOZ `runs/joz_perim_*` and `runs/joz_*_w{400..900}_rs*` `githubresults.csv` | JOZ-CNN in Fig. 7 (10 s and 4-9 s) |
| FLAIRS `results_perim/*.il1000.pkl` | TISER rows, Tables 3, 4, 6, 7; TISER-GCN SFA column, Table 5; 10 s points, Fig. 7 |
| FLAIRS `results/*.il{400..900}.pkl` | TISER models in Fig. 7 (4-9 s) |

GPU training in TF 2.6 and in our bf16 PyTorch runs is not deterministic. Re-runs therefore reproduce the
numbers within seed variation, not bit for bit.
