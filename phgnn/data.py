"""Seismic datasets: file layout, train/validation/test splits and graph families."""
import os

import lightning as L
import numpy as np
import torch
from sklearn import model_selection
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import to_undirected

FOLDERS = dict(ci="central_it", cw="central_west_it")


def dataset_root(cfg, data_root):
    return os.path.join(data_root, FOLDERS[cfg["dataset"]])


def data_file(cfg, key, data_root):
    """Return the configured file for `key`; relative paths are taken relative to data_root and may
    contain {dataset}, {folder} and {seed} (the data seed) placeholders."""
    if not cfg.get(key):
        return None
    return os.path.join(data_root, cfg[key].format(dataset=cfg["dataset"], folder=FOLDERS[cfg["dataset"]],
                                                   seed=cfg["data_seed"]))


def input_paths(cfg, data_root):
    root = dataset_root(cfg, data_root)
    inputs = data_file(cfg, "inputs_path", data_root) or os.path.join(root, f"inputs_{cfg['dataset']}.npy")
    coords_dir = os.path.dirname(baseline_graph_path(cfg, data_root)) if cfg["exp"] == "baseline" else root
    return dict(inputs=inputs, targets=os.path.join(root, "targets.npy"),
                coords=os.path.join(coords_dir, "station_coords.npy"))


def baseline_graph_path(cfg, data_root):
    return data_file(cfg, "graph_path", data_root) or os.path.join(
        dataset_root(cfg, data_root), "baseline", "minmax_normalized_laplacian.npy")


def load_baseline_operator(cfg, data_root):
    return torch.as_tensor(np.load(baseline_graph_path(cfg, data_root)), dtype=torch.float32)


def order_folder(graph_order):
    if isinstance(graph_order, list):
        return "order_" + "-".join(map(str, graph_order))
    return f"order_{graph_order}"


def load_graph_family(cfg, data_root):
    """Return (edge_indices, edge_weights) of the ensemble's graphs.

    Stored PH graphs keep each edge once (i < j); they are mirrored so that messages pass both ways.
    """
    path = data_file(cfg, "graph_family_path", data_root)
    if path is None:
        raise ValueError("graph_family_path is not set")
    graphs = torch.load(path, weights_only=False)
    pairs = [to_undirected(g.edge_index, g.edge_attr, num_nodes=g.num_nodes, reduce="mean") for g in graphs]
    return [ei for ei, _ in pairs], [ew for _, ew in pairs]


def normalize_events(inputs):
    """Scale every event by its maximum absolute amplitude over stations, samples and channels."""
    return np.array([event / np.maximum(np.max(np.abs(event)), 1e-8) for event in inputs])


def tser_fold_indices(n, seed, n_splits, k):
    """Return (train, validation) indices of fold k, as in the TSER-GCN reference code.

    The reference permutes once and cuts contiguous chunks (remainder to the last fold); its
    extra np.random.permutation(seed) call is kept so the random stream matches.
    """
    np.random.seed(seed)
    np.random.permutation(seed)
    order = np.random.permutation(n)
    size = n // n_splits
    chunks = [order[i * size:(i + 1) * size] for i in range(n_splits - 1)] + [order[(n_splits - 1) * size:]]
    train = np.concatenate([c for i, c in enumerate(chunks) if i != k])
    return train.tolist(), chunks[k].tolist()


class EarthquakeKFoldDataModule(L.LightningDataModule):
    """80/20 train/test split of events; fold k of the training part is the validation set."""

    def __init__(self, k, cfg, data_root):
        super().__init__()
        self.k, self.cfg = k, cfg
        self.paths = input_paths(cfg, data_root)
        self.data_train = self.data_val = self.data_test = None

    def setup(self, stage=None):
        if self.data_train is not None:
            return
        cfg = self.cfg
        inputs = np.load(self.paths["inputs"], allow_pickle=True)[:, :, :cfg["window"], :]
        targets = np.load(self.paths["targets"], allow_pickle=True)
        coords = torch.from_numpy(np.load(self.paths["coords"], allow_pickle=True)).float()
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            inputs, targets, test_size=1 - cfg["train_ratio"], random_state=cfg["data_seed"])
        if cfg["normalize_inputs"]:
            x_train, x_test = normalize_events(x_train), normalize_events(x_test)
        train_idx, val_idx = tser_fold_indices(len(x_train), cfg["fold_seed"], cfg["k_fold"], self.k)
        self.data_train = self._dataset(x_train[train_idx], y_train[train_idx], coords)
        self.data_val = self._dataset(x_train[val_idx], y_train[val_idx], coords)
        self.data_test = self._dataset(x_test, y_test, coords)

    def _dataset(self, x, y, coords):
        """(waveforms [E, N, T, C], station coordinates [E, N, 2], targets [E, N, 5])."""
        return TensorDataset(torch.from_numpy(x).float(), coords.unsqueeze(0).expand(len(x), *coords.shape),
                             torch.from_numpy(y).float())

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.cfg["batch_size"], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.cfg["batch_size"])

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=len(self.data_test))
