# Persistent Homology-induced Graph Ensembles for Time Series Regressions

This is the code implementation for paper "Persistent Homology-induced Graph Ensembles for Time Series Regressions". 

## Use Cases
We experimented all two applications:
1. Time-series Extrinsic Regression (TSER) on 2 seismic earthquake datasets: Central-West Italy (CW) and Central Italy (CI).
2. Traffic speed forecasting on PEMS-BAY and METR-LA.

## How To Run
Our model implementations are provided in `{PROJECT_ROOT}/src`, and the experiment scripts are stored in `{PROJECT_ROOT}/experiments`. 

### 1. Download datasets
Due to large datasets, we cannot store them in the repository. Please download the dataset via this [link](https://drive.google.com/file/d/16IiHnW9_fhh5hMCODyCm6HPUKqiNERv0/view?usp=sharing). Then, place all the contents in `data/*` of the zip file into `{PROJECT_ROOT}/data` folder.

### 2. Create a WanDB account
We track all the logging information via `WanDB` together with `Torch Lightning`. You only need to create an account on this [platform](https://wandb.ai/site) and retrieve the Token-ID in the account. When first calling the script, user is prompted to give the Token-ID input. Then, the experiments will run automatically. Optionally, the script also provides an option to create a `WanDB` account directly in the terminal.

### 3. Environment setup
We use [Conda](https://anaconda.org/anaconda/conda) to manage the python environment:
1. In the project root, run `pip install -e .`
2. Run `conda create -n phgnn python=3.11`
3. Install [PyTorch 2.5.1](https://pytorch.org/get-started/locally/). If you have CUDA 12.x GPU on a Linux machine, run `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`.
4. Install [PyTorch Lightning 2.5.0](https://lightning.ai/docs/pytorch/stable/): `conda install lightning -c conda-forge`
5. Install [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html): `conda install pyg -c pyg`
6. Install [scikit-learn](https://scikit-learn.org/1.6/install.html): `conda install -c conda-forge scikit-learn`
7. Install [WanDB](https://docs.wandb.ai/support/anaconda_package/): `conda install -c conda-forge wandb`
8. Install [einops](https://anaconda.org/conda-forge/einops): `conda install conda-forge::einops`

### 4. Example of running scripts
One can run the scripts in `experiments` folder with manual input. We provide all the `yml` configurations in the `configs` folder for ease of reproducability. 

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