"""Write the GCN operator D~^-1/2 (A + I) D~^-1/2 of TSER-GCN's graph as a dense matrix.

TSER-GCN releases only its normalized Laplacian L = I - D^-1/2 A D^-1/2 (connected graph, no self-loops). Its
adjacency is recovered exactly: sqrt(degree) is the eigenvector of I - L for eigenvalue 1, which gives A up to a
scale, and the scale is fixed by the largest weight, 0.98 (the closest pair of stations, w = 0.98 - minmax(d)).
The operator is then computed by PyG's gcn_norm, as in our GCNConv layers:

    <folder>/baseline/gcn_normalized_adjacency.npy        TSER-GCN's graph with the GCN operator (graph_path)

    python experiments/build_gcn_operators.py --data_root DIR
"""
import argparse
import os

import numpy as np
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from phgnn import config, data


def dense_gcn_operator(edge_index, edge_weight, nodes):
    """The matrix S with GCNConv(x) = S x W: stored self-loop weights are kept, missing ones get weight 1."""
    index, weight = gcn_norm(edge_index, edge_weight.view(-1).double(), num_nodes=nodes, add_self_loops=True)
    operator = torch.zeros(nodes, nodes, dtype=torch.float64)
    operator.index_put_((index[1], index[0]), weight, accumulate=True)
    return operator.numpy()


def adjacency_from_laplacian(laplacian, max_weight=0.98):
    propagation = np.eye(len(laplacian)) - laplacian
    values, vectors = np.linalg.eigh(propagation)
    assert abs(values[-1] - 1) < 1e-9 and values[-2] < 1 - 1e-9, "graph must be connected"
    root_degree = np.abs(vectors[:, -1])
    adjacency = root_degree[:, None] * propagation * root_degree[None, :]
    return adjacency * max_weight / adjacency.max()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()
    for dataset in data.FOLDERS:
        cfg = dict(config.DEFAULTS, dataset=dataset)
        root = data.dataset_root(cfg, args.data_root)
        adjacency = adjacency_from_laplacian(np.load(os.path.join(root, "baseline", "minmax_normalized_laplacian.npy")))
        edges = torch.as_tensor(np.array(np.nonzero(adjacency > 1e-12)))
        weights = torch.as_tensor(adjacency[edges[0], edges[1]])
        out = os.path.join(root, "baseline", "gcn_normalized_adjacency.npy")
        np.save(out, dense_gcn_operator(edges, weights, len(adjacency)).astype(np.float32))
        print(f"{dataset}: {len(weights) // 2} edges -> {out}")


if __name__ == "__main__":
    main()
