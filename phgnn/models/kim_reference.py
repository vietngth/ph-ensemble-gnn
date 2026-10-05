"""KIM-GNN: the GCN of Kim et al. (IEEE GRSL 2022), copied from the authors' code and adapted to IM regression.

Source: github.com/GT-KIM/GCN_seismic_event_classification @84b9116, train.py (MIT License, Copyright (c) 2021 GTKim).
FeatureExtraction, NodeApplyModule, GCN and the Classifier's layers are verbatim. The only change inside them: their
DGL message passing (update_all(copy_src, sum), i.e. h_i = sum over the edges j -> i of h_j) is written as a dense
product with the weighted adjacency, h = A h. The adaptations are the three that the TSER-GCN paper (Bloemheuvel et
al. 2022, Sec. 3.8) lists for its "adjusted version of [Kim et al.]": regression output layers instead of the 3-class
layer, a weighted initial adjacency matrix (the TSER-GCN authors' graph_maker.py weights), and the station coordinates
added to the node features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from phgnn.models import base


class NodeApplyModule(nn.Module) :
    def __init__(self, in_feats, out_feats, activation) :
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node) :
        h = self.linear(node.data['h'])
        if self.activation is not None :
            h = self.activation(h)
        return {'h' : h}


class GCN(nn.Module) :
    def __init__(self, in_feats, out_feats, activation) :
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, adjacency, feature) :
        # authors: g.ndata['h'] = feature; g.update_all(copy_src('h', 'm'), sum('m', 'h')); h = g.ndata['h']
        h = adjacency.transpose(-1, -2) @ feature        # h_i = sum_j A_ji h_j (edge j -> i, weight A_ji)
        return self.linear(h)


class FeatureExtraction(nn.Module) :
    def __init__(self) :
        super(FeatureExtraction, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 3, 1, 1)
        self.conv1_pool = nn.MaxPool1d(3, 2, 1)
        self.conv2 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv2_pool = nn.MaxPool1d(3, 2, 1)
        self.conv3 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv3_pool = nn.MaxPool1d(3, 2, 1)
        self.conv4 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv4_pool = nn.MaxPool1d(3, 2, 1)
        self.conv5 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv5_pool = nn.MaxPool1d(3, 2, 1)
        self.conv6 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv6_pool = nn.MaxPool1d(3, 2, 1)
        self.conv7 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv7_pool = nn.MaxPool1d(3, 2, 1)
        self.conv8 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv8_pool = nn.MaxPool1d(3, 2, 1)

    def forward(self, feature) :
        feature = self.conv1_pool(F.relu((self.conv1(feature))))
        feature = self.conv2_pool(F.relu((self.conv2(feature))))
        feature = self.conv3_pool(F.relu((self.conv3(feature))))
        feature = self.conv4_pool(F.relu((self.conv4(feature))))
        feature = F.relu((self.conv5(feature)))
        feature = F.relu((self.conv6(feature)))
        feature = F.relu((self.conv7(feature)))
        outputs = F.relu((self.conv8(feature)))
        return torch.flatten(outputs, start_dim=1)


class KimGNNModel(base.SeismicRegressor):
    """Kim et al.'s per-station CNN + 5 GCN layers, with TSER-GCN's regression head for the 5 x N outputs."""

    def __init__(self, size_config, arch_config, train_config, operator):
        super().__init__(train_config)
        self.save_hyperparameters(ignore=["operator"])
        nodes, hidden_dim = size_config["num_nodes"], 64
        self.feature_extraction = FeatureExtraction()
        with torch.no_grad():
            in_dim = self.feature_extraction(torch.zeros(1, 3, arch_config["window"])).shape[1]
        in_dim += 2                                             # adaptation: station coordinates as node features
        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        # Kim's classifier (mean_nodes readout + MLP, output layer resized to 5 x N) is built and then replaced; building
        # it keeps the random stream, and so the initial weights, of the reported runs
        self.classify = nn.Sequential(nn.Linear(hidden_dim, 128), torch.nn.ReLU(),
                                      nn.Linear(128, 64), torch.nn.ReLU(),
                                      nn.Linear(64, 5 * nodes))
        # adaptation ("the last layers used for classification were altered for regression"): TSER-GCN's regression head
        self.classify = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(hidden_dim * nodes, 128),
                                      nn.Linear(128, 5 * nodes))
        self.register_buffer("operator", operator.float())

    def predict(self, x, coords):
        batch, nodes, window, channels = x.shape
        h = self.feature_extraction(x.permute(0, 1, 3, 2).reshape(batch * nodes, channels, window)).view(batch, nodes, -1)
        h = torch.cat([h, coords], dim=2)
        for conv in self.layers:
            h = conv(self.operator, h)
        pred = self.classify(h).view(batch, 5, nodes).permute(0, 2, 1)
        return pred, torch.zeros((), device=x.device)          # no L2 term: Adam's weight decay, as the authors
