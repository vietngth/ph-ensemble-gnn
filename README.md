# Persistent Homology-induced Graph Ensembles for Time Series Regressions

This is the code implementation for paper "Persistent Homology-induced Graph Ensembles for Time Series Regressions". 
Preprint: [[link]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5521531).

## Use Cases
We experimented all two applications:
1. Time-series Extrinsic Regression (TSER) on 2 seismic earthquake datasets: Central-West Italy (CW) and Central Italy (CI).
2. Traffic speed forecasting on PEMS-BAY and METR-LA.

## How To Run
Our model implementations are provided in `{PROJECT_ROOT}/src`, and the experiment scripts are stored in `{PROJECT_ROOT}/experiments`. 

### 1. Download datasets
Due to large datasets, we cannot store them in the repository. Please download the dataset via this [link](https://drive.google.com/file/d/16IiHnW9_fhh5hMCODyCm6HPUKqiNERv0/view?usp=sharing). Then, place all the contents in `data/*` of the zip file into `{PROJECT_ROOT}/data` folder.

### 2. Create a WanDB account
We track all the logging information via `WanDB` together with `Torch Lightning`. You only need to create an account on this [platform](https://wandb.ai/site) and retrieve the Token-ID in the account. When first calling the script, the user is prompted to give the Token-ID input. Then, the experiments will run automatically. Optionally, the script also provides options to create a `WanDB` account directly in the terminal or to run the script without visualizations.

### 3. Environment setup
We use [uv](https://docs.astral.sh/uv/) to manage the Python environment. In the project root:

1. Run `uv sync`. This step will install all necessary dependencies declared in `pyproject.toml`.
2. Run `source .venv/bin/activate`.
3. Install `torch_geometric`: run `uv pip install torch_geometric`.
4. Install dependencies: run `uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html`
5. Run `pip install -e .`

Note that we use `PyTorch 2.5.1` with `CUDA 12.4`. You may want to install all packages manually with `uv` if you want to use a different PyTorch version.

### 4. Example of running scripts
One can run the scripts in `experiments` folder with manual input. We provide all the `yml` configurations in the `configs` folder for ease of reproducibility. 

### 4.1. Earthquake Regressions
Run attention-based model `PH-TSER-Att_0` on CI dataset:
```python
python3 experiments/train_regression.py --config configs/earthquake_regression/central_it/weighted/weighted_0.yml
```
Run attention-based model `PH-TSER-Att_0` on CW dataset:
```python
python3 experiments/train_regression.py --config configs/earthquake_regression/central_west_it/weighted/weighted_0.yml
```
Run window reduction on attention-based model `PH-TSER-Att_0` on CW dataset with `W=4s`:
```python
python3 experiments/train_regression.py --config configs/earthquake_regression/central_west_it/windows/windows_400.yml
```

#### 4.2. Traffic Forecasting
Run attention-based model `PH-TSER-Att_0` on METR-LA dataset:
```python
python3 experiments/train_forecasting.py --config configs/traffic/metr_la/weighted_0.yml
```
Run attention-based model `PH-TSER-Att_0` on PEMS-BAY dataset:
```python
python3 experiments/train_forecasting.py --config configs/traffic/metr_la/weighted_0_k150.yml
```

### 5. Results
All results are stored in:
- WanDB: all logging metrics, training and validation losses.
- Local: `{PROJECT_ROOT}/logs/{task_name}/{experiment}/*`