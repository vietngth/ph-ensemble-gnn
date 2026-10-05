"""Graph families for the hypothesis tests on why G0 > G1 and why G0-1 = G0 (clean, unweighted threshold graphs).

Uses <folder>/geodesic_km.npy (geodesic station distances; its threshold graphs reproduce every released edge set) and
writes <folder>/hypothesis/<name>_<dataset>.pt:
    g0_trunc   G0 without the graphs above the largest H1 death time (G0 limited to the reach of G1)
    g1_reach   G1 plus those G0 graphs (G1 given the reach of G0)
    g1_small   G1 plus the G0 graphs below the smallest H1 death time (G1 given the small scales of G0)
    g0_rand<r> G0 plus as many graphs as G1 at random thresholds (pairwise distances in G1's range that are not death
               times), r = 1, 2, 3: are the H1 death times special, or is any extra scale in that range the same?
    empty      38 graphs without edges (every sub-network sees only its own station)

    python experiments/build_hypothesis_graphs.py --data_root DIR
"""
import argparse
import os

import numpy as np
import torch
from torch_geometric.data import Data

from phgnn import config, data

DRAWS = 3


def threshold(graph, distances):
    edges = graph.edge_index[:, graph.edge_index[0] != graph.edge_index[1]]
    return float(distances[edges[0], edges[1]].max()) if edges.shape[1] else 0.0


def threshold_graph(distances, eps):
    i, j = np.triu_indices(len(distances), 1)
    keep = distances[i, j] <= eps + 1e-9
    edge_index = torch.as_tensor(np.stack([i[keep], j[keep]]), dtype=torch.long)
    return Data(edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[1], 1), num_nodes=len(distances))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()
    for dataset in data.FOLDERS:
        root = data.dataset_root(dict(config.DEFAULTS, dataset=dataset), args.data_root)
        distances = np.load(os.path.join(root, "geodesic_km.npy"))
        g0 = torch.load(os.path.join(root, "order_0", "clean", f"pyg_graphs_{dataset}.pt"), weights_only=False)
        g1 = torch.load(os.path.join(root, "order_1", "clean", f"pyg_graphs_{dataset}.pt"), weights_only=False)
        for g in g0 + g1:  # the clean graphs must be exact threshold graphs of these distances
            assert torch.equal(threshold_graph(distances, threshold(g, distances)).edge_index.sort(dim=1)[0],
                               g.edge_index[:, g.edge_index[0] != g.edge_index[1]].sort(dim=1)[0]) or threshold(g, distances) == 0
        e0 = np.array([threshold(g, distances) for g in g0])
        e1 = np.array([threshold(g, distances) for g in g1])
        families = dict(g0_trunc=[g for g, e in zip(g0, e0) if e <= e1.max()],
                        g1_reach=g1 + [g for g, e in zip(g0, e0) if e > e1.max()],
                        g1_small=[g for g, e in zip(g0, e0) if e < e1.min()] + g1,
                        empty=[Data(edge_index=torch.zeros(2, 0, dtype=torch.long), edge_attr=torch.zeros(0, 1),
                                    num_nodes=len(distances)) for _ in g0])
        pairs = distances[np.triu_indices(len(distances), 1)]
        candidates = np.unique(pairs[(pairs >= e1.min()) & (pairs <= e1.max())])
        candidates = candidates[~np.isin(np.round(candidates, 6), np.round(np.r_[e0, e1], 6))]
        for r in range(1, DRAWS + 1):
            rng = np.random.default_rng(1000 * r + len(dataset))
            eps = np.sort(rng.choice(candidates, size=len(g1), replace=False))
            families[f"g0_rand{r}"] = g0 + [threshold_graph(distances, e) for e in eps]
            print(f"{dataset} g0_rand{r}: thresholds {np.round(eps, 1).tolist()} (H1 death times {np.round(e1, 1).tolist()})")
        os.makedirs(os.path.join(root, "hypothesis"), exist_ok=True)
        for name, graphs in families.items():
            torch.save(graphs, os.path.join(root, "hypothesis", f"{name}_{dataset}.pt"))
            print(f"{dataset} {name}: {len(graphs)} graphs")


if __name__ == "__main__":
    main()
