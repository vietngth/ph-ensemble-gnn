# Repository of Persistent homology-induced graph ensembles for seismic intensity regression

## Layout

```
phgnn/            package: config, data (splits, folds, graph families), metrics, lr range test, models
  models/         ensemble.py (PH-TSER-Att), baseline.py (TSER-GCN), kim_reference.py (KIM-GNN)
experiments/      train.py, predict.py, build_*.py (derived data)
baselines/        JOZ-CNN and the two FLAIRS models, run with their authors' code (patches, configs, run lists)
tests/            unit tests, graph-construction checks
```

## Setup

```
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[test,ph]"
uv pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

The `ph` extra (gudhi, geopy) is only needed to rebuild the PH graphs. The FLAIRS baselines and the SFA inputs need a
separate TensorFlow environment (`baselines/env/`).

## Data

The datasets and checkpoints are available on HuggingFace:

```
huggingface-cli download vietngth/ph-ensemble-gnn-data ph-gnn-data.zip --repo-type dataset --local-dir .
python -m zipfile -e ph-gnn-data.zip .     # creates data/central_it (CI) and data/central_west_it (CW)
```

Or download manually from https://huggingface.co/datasets/vietngth/ph-ensemble-gnn-data and extract `ph-gnn-data.zip`
in the repository root (it creates `data/`).

Further descriptions of the data is available in data/README.md.

The derived files can be rebuilt from the station coordinates, in this order:

```
python experiments/build_ph_graphs.py --data_root data --out_root data    # distances, H0/H1 death times, G0, G1, G0-1 
python experiments/build_clean_graphs.py --data_root data                 # unweighted threshold graphs + one-graph control
python experiments/build_hypothesis_graphs.py --data_root data            # graph families of the analysis 
python experiments/build_single_scale_graphs.py --data_root data          # 38 single-scale controls
python experiments/build_gcn_operators.py --data_root data                # GCN operator of the TSER-GCN graph
<flairs-env>/bin/python experiments/build_sfa_inputs.py --data_root data  # SFA inputs
```

## **Training**

```
# untuned (published TSER-GCN settings)
python experiments/train.py --config configs/fast.yml configs/experiments/clean/shared_g0.yml --set dataset=ci data_seed=1 --data_root data
python experiments/train.py --config configs/fast.yml configs/experiments/clean/shared_g0.yml --set dataset=cw data_seed=1 --data_root data

# tuned (AdamW, one-cycle, learning rate from a range test)
python experiments/train.py --config configs/fast.yml configs/experiments/clean/shared_g0.yml configs/experiments/optim/fastai.yml --set dataset=ci data_seed=1 --data_root data
python experiments/train.py --config configs/fast.yml configs/experiments/clean/shared_g0.yml configs/experiments/optim/fastai.yml --set dataset=cw data_seed=1 --data_root data
```

`data_seed` (1-10) fixes the train/test split; results go to `logs/runs/`.

## **Inference**

Download the checkpoints into `data/checkpoints/` (or manually from https://huggingface.co/vietngth/ph-ensemble-gnn):

```
huggingface-cli download vietngth/ph-ensemble-gnn --local-dir data/checkpoints
python experiments/predict.py --checkpoint data/checkpoints/ci_tuned_seed1/best.ckpt --results data/checkpoints/ci_tuned_seed1/results.json --data_root data
```

Folders: `{ci,cw}_{tuned,untuned}_seed1/`, each with `best.ckpt` and `results.json`.
