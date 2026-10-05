"""Rewrite the PH graph families as unweighted distance-threshold graphs, the graphs of Algorithm 1 and Eq. 10.

The released pyg_graphs_*.pt files have the right edge sets (all pairs with d_ij <= eps at each death time) but
min-max normalized weights that include the diagonal (the merging edge gets weight 0, self-loops 0.4-1.0). The
clean files keep every edge set, give each edge weight 1 and store no self-loops, so GCNConv uses A + I:

    <folder>/order_<o>/clean/pyg_graphs_<dataset>.pt      for o in 0, 1, 0-1
    <folder>/order_0/clean/samegraph_<dataset>.pt          38 copies of clean G^(0) graph 19 (same-graph control)

    python experiments/build_clean_graphs.py --data_root DIR
"""
import argparse
import os

import torch
from torch_geometric.data import Data

from phgnn import config, data

ORDERS = (0, 1, [0, 1])
SAME_GRAPH = 19  # the graph repeated by the existing same-graph control (e3/<folder>/samegraph_<dataset>.pt)


def clean(graph):
    edge_index = graph.edge_index[:, graph.edge_index[0] != graph.edge_index[1]]
    return Data(edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[1], 1), num_nodes=graph.num_nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()
    for dataset in data.FOLDERS:
        for order in ORDERS:
            cfg = dict(config.DEFAULTS, dataset=dataset, graph_order=order)
            folder = os.path.join(data.dataset_root(cfg, args.data_root), data.order_folder(order))
            graphs = torch.load(os.path.join(folder, f"pyg_graphs_{dataset}.pt"), weights_only=False)
            out = os.path.join(folder, "clean", f"pyg_graphs_{dataset}.pt")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torch.save([clean(g) for g in graphs], out)
            print(f"{dataset} {data.order_folder(order)}: {len(graphs)} graphs -> {out}")
            if order == 0:
                same = os.path.join(folder, "clean", f"samegraph_{dataset}.pt")
                torch.save([clean(graphs[SAME_GRAPH])] * len(graphs), same)
                print(f"{dataset}: same-graph control -> {same}")


if __name__ == "__main__":
    main()
