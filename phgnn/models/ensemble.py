"""PH-TSER-Att: a shared CNN encoder and one GCN sub-network per graph of a (persistent-homology) graph family."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from phgnn.models import base


def encode_waveforms(conv1, conv2, x):
    """Run the CNN encoder on x [B, N, T, C]; return features [B, N, F]."""
    batch, nodes, window, channels = x.shape
    h = x.permute(0, 1, 3, 2).reshape(batch * nodes, channels, window)
    h = F.relu(conv2(F.relu(conv1(h))))
    return h.view(batch, nodes, -1)


class GraphExpert(nn.Module):
    """One sub-network: two GCN layers on the encoded waveforms and station coordinates, five linear heads."""

    def __init__(self, size_config, arch_config, train_config):
        super().__init__()
        hidden, nodes = size_config["hidden_size"], size_config["num_nodes"]
        self.reg_const = train_config["reg_const"]
        # An encoder is built here only for its output size. Building it keeps the random stream, and so the initial
        # weights, of the reported runs.
        conv1, conv2 = base.conv_encoder(size_config["in_size"], hidden, size_config["kernel_size"], size_config["stride"])
        features = base.encoded_size(conv1, conv2, arch_config["window"], size_config["in_size"]) + 2  # + coordinates
        self.graph_conv1 = GCNConv(features, hidden * 2, bias=False)
        self.graph_conv2 = GCNConv(hidden * 2, hidden * 2, bias=False)
        self.fc1 = nn.Linear(hidden * 2 * nodes, hidden * 4)
        for i in range(1, 6):
            setattr(self, f"fc2{i}", nn.Linear(hidden * 4, nodes))
        self.dropout = nn.Dropout(arch_config["dropout"])

    def heads(self):
        return [getattr(self, f"fc2{i}") for i in range(1, 6)]

    def forward(self, encoded, coords, edge_index, edge_weight):
        """Return (predictions [5, B, N], node representations [B, N, 2h])."""
        h = torch.cat([encoded, coords], dim=2)
        h = torch.tanh(self.graph_conv2(F.relu(self.graph_conv1(h, edge_index, edge_weight)), edge_index, edge_weight))
        hidden = self.fc1(self.dropout(h.reshape(len(h), -1)))
        return torch.stack([head(hidden) for head in self.heads()]), h

    def l2_regularization(self):
        return base.l2_penalty(self.reg_const, [], [self.graph_conv1, self.graph_conv2])


class MultiGraphsAggregator(base.SeismicRegressor):
    """Combine the sub-networks' predictions by a softmax gate.

    The gate scores every sub-network from its own node representations; weights are per node. The loss acts on the
    blended prediction (mixture_loss 'blended', the cooperative error of Jacobs et al. 1991, Eq. 1.1) or on the
    gate-weighted squared errors of the sub-networks (mixture_loss 'competitive', their Eq. 1.2); predictions are the
    blended ones in both cases. The L2 penalty covers the kernels of the shared encoder and of every GCN layer.
    """

    def __init__(self, size_config, arch_config, train_config):
        super().__init__(train_config)
        self.save_hyperparameters()
        hidden = size_config["hidden_size"]
        self.mixture_loss = arch_config["mixture_loss"]
        if self.mixture_loss not in ("blended", "competitive"):
            raise ValueError(f"mixture_loss must be blended or competitive, got {self.mixture_loss!r}")
        self.edge_indices = arch_config["edge_indices"]
        self.edge_weights = arch_config["edge_weights"]
        self.experts = nn.ModuleList([GraphExpert(size_config, arch_config, train_config) for _ in self.edge_indices])
        self.gate_proj = nn.Linear(hidden * 2, 1)
        self.shared_conv1, self.shared_conv2 = base.conv_encoder(
            size_config["in_size"], hidden, size_config["kernel_size"], size_config["stride"])
        self.gate_weights = None

    def gate(self, representations):
        """Return softmax weights [B, K, N, 1] over the K sub-networks."""
        return torch.softmax(torch.stack([self.gate_proj(r) for r in representations], dim=1), dim=1)

    def predict_parts(self, x, coords):
        """Return (sub-network predictions [B, K, N, 5], per-node weights [B, K, N, 1], l2)."""
        encoded = encode_waveforms(self.shared_conv1, self.shared_conv2, x)
        outputs, representations = [], []
        for expert, edge_index, edge_weight in zip(self.experts, self.edge_indices, self.edge_weights):
            out, rep = expert(encoded, coords, edge_index.to(x.device), edge_weight.to(x.device))
            outputs.append(out)
            representations.append(rep)
        stacked = torch.stack(outputs).permute(2, 0, 3, 1)
        weights = self.gate(representations)
        self.gate_weights = weights.detach().float().mean(dim=(0, 2, 3))
        l2 = sum(expert.l2_regularization() for expert in self.experts)
        l2 = l2 + base.l2_penalty(self.reg_const, [self.shared_conv1.weight, self.shared_conv2.weight], [])
        return stacked, weights, l2

    def predict(self, x, coords):
        stacked, weights, l2 = self.predict_parts(x, coords)
        return (stacked * weights).sum(dim=1), l2

    def training_step(self, batch, batch_idx):
        if self.mixture_loss == "blended":
            return super().training_step(batch, batch_idx)
        x, coords, y = self.split_batch(batch)
        stacked, weights, l2 = self.predict_parts(x, coords)
        errors = (stacked - y[:, None]) ** 2
        loss = (weights * errors).sum(dim=1).mean(dim=(0, 1)).sum() + l2
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
