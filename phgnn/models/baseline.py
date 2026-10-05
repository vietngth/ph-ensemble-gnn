"""Single-graph TSER-GCN baseline."""
import torch
import torch.nn.functional as F
from torch import nn

from phgnn.models import base


class BaselineModel(base.SeismicRegressor):
    """CNN encoder (+ station coordinates), two graph convolutions A X W with a stored graph matrix A, five heads."""

    def __init__(self, size_config, arch_config, train_config, operator):
        super().__init__(train_config)
        self.save_hyperparameters(ignore=["operator"])
        hidden, nodes = size_config["hidden_size"], size_config["num_nodes"]
        self.conv1, self.conv2 = base.conv_encoder(
            size_config["in_size"], hidden, size_config["kernel_size"], size_config["stride"])
        features = base.encoded_size(self.conv1, self.conv2, arch_config["window"], size_config["in_size"]) + 2
        self.graph_conv1 = nn.Linear(features, hidden * 2, bias=False)
        self.graph_conv2 = nn.Linear(hidden * 2, hidden * 2, bias=False)
        self.fc = nn.Linear(hidden * 2 * nodes, hidden * 4)
        self.output_layers = nn.ModuleList([nn.Linear(hidden * 4, nodes) for _ in range(5)])
        self.dropout = nn.Dropout(arch_config["dropout"])
        self.register_buffer("operator", operator.float())

    def encode(self, x):
        batch, nodes, window, channels = x.shape
        h = x.permute(0, 1, 3, 2).reshape(batch * nodes, channels, window)
        h = F.relu(self.conv2(F.relu(self.conv1(h))))
        return h.view(batch, nodes, -1)

    def propagate(self, h):
        h = F.relu(self.operator @ self.graph_conv1(h))
        return torch.tanh(self.operator @ self.graph_conv2(h))

    def predict(self, x, coords):
        h = torch.cat([self.encode(x), coords], dim=2)
        h = self.fc(self.dropout(self.propagate(h).reshape(len(x), -1)))
        pred = torch.stack([head(h) for head in self.output_layers], dim=2)
        l2 = base.l2_penalty(self.reg_const, [self.conv1.weight, self.conv2.weight],
                             [self.graph_conv1, self.graph_conv2])
        return pred, l2
