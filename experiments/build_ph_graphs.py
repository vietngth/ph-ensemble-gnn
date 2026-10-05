"""Build the PH graph families of Algorithm 1 from the station coordinates, and check them against a data folder.

For each dataset, from <data_root>/<folder>/station_coords.npy ((lat, lon) in degrees, one row per station):
  1. distances  d_ij = WGS84 geodesic distance in km, geopy.distance.geodesic (Karney's algorithm via geographiclib, the
     successor of Vincenty's formula that geopy dropped in 2.0). This reproduces geodesic_km.npy bit for bit; pyproj's
     Geod(ellps="WGS84") agrees to 3e-12 km, while haversine is off by up to 0.3 km (CI) / 0.7 km (CW).
  2. persistence of the Vietoris-Rips filtration of d (gudhi, max_dimension=2, Z/2 coefficients):
       H0 finite death times, ascending = the MST edge lengths (Proposition 1; asserted against scipy's MST);
       H1 (birth, death) pairs, ordered by birth (the order of the released files, not sorted by death time).
  3. graphs G_eps = {(i, j): i != j, d_ij <= eps} at every death time:
       order_0 = H0 death times (38 graphs), order_1 = H1 death times (CI 6, CW 8), order_0-1 = order_0 then order_1.

Writes, under <out_root>/<folder>/ (never over the data unless --out_root is the data root):
    geodesic_km.npy                              d (float64)
    baseline/identity_operator.npy               np.eye(N, dtype=float32), the operator of the no-graph baseline
    order_<o>/clean/pyg_graphs_<dataset>.pt      the graphs of the paper: edges i < j, weight 1, no self-loops
    order_<o>/pyg_graphs_<dataset>.pt            the "released" format (see below), input of build_clean_graphs.py
for o in 0, 1, 0-1. The released format is the legacy weighting that the paper replaced by unit weights: each graph has
x = arange(N) and stores the upper triangle (i <= j, row-major) of A + I with weight min-max(I - D^-1/2 W D^-1/2),
W = A * S + 0.98 I and S = 0.98 - minmax(d) (TSER-GCN's similarity, see build_gcn_operators.py), so self-loops weigh
0.4-1.0 and one edge per graph weighs 0. It is rebuilt exactly (float32 bit for bit), so build_clean_graphs.py on these
files gives the same clean files. The same-graph control (order_0/clean/samegraph_*.pt) is left to build_clean_graphs.py.

--check compares everything built (in memory) with the files under --data_root and prints PASS/FAIL per file: graph
count, ordering, per-graph edge sets, thresholds (the largest edge length of each stored clean graph = our death time),
weights and node features of the released files, the geodesic max abs difference and the identity operator.
It exits with status 1 if anything fails.

    python experiments/build_ph_graphs.py --data_root DIR --out_root OUT            build into OUT
    python experiments/build_ph_graphs.py --data_root DIR --check                   only check against DIR
    python experiments/build_ph_graphs.py --data_root DIR --out_root OUT --check    both
Needs the optional dependencies: pip install -e ".[ph]" (gudhi, geopy).
"""
import argparse
import os

import gudhi
import numpy as np
import torch
from geopy.distance import geodesic
from scipy.sparse.csgraph import minimum_spanning_tree
from torch_geometric.data import Data

from build_clean_graphs import clean as clean_released  # experiments/ is on sys.path when run as a script
from phgnn import config, data

GEODESIC_TOL_KM = 1e-6
TSER_MAX_SIMILARITY = 0.98  # S = 0.98 - minmax(d), TSER-GCN's station similarity


def geodesic_matrix(coords):
    """WGS84 geodesic distances in km between (lat, lon) rows, computed for i < j and mirrored."""
    n = len(coords)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = distances[j, i] = geodesic(coords[i], coords[j]).km
    return distances


def persistence(distances):
    """Return (H0 death times ascending, H1 (birth, death) pairs ordered by birth) of the Vietoris-Rips filtration.

    Filtration values are the entries of `distances` themselves (doubles), so the death times are exact distances."""
    tree = gudhi.RipsComplex(distance_matrix=distances).create_simplex_tree(max_dimension=2)
    tree.compute_persistence(homology_coeff_field=2, min_persistence=0)
    h0 = tree.persistence_intervals_in_dimension(0)
    h0 = np.sort(h0[np.isfinite(h0[:, 1]), 1])
    h1 = tree.persistence_intervals_in_dimension(1).reshape(-1, 2)
    h1 = h1[np.lexsort((h1[:, 1], h1[:, 0]))]
    mst = minimum_spanning_tree(distances).toarray()
    assert np.array_equal(h0, np.sort(mst[mst > 0])), "H0 death times must be the MST edge lengths"
    assert len(h0) == len(distances) - 1 and np.isfinite(h1).all()
    return h0, h1


def adjacency(distances, eps):
    a = (distances <= eps).astype(np.float64)
    np.fill_diagonal(a, 0)
    return a


def clean_graph(distances, eps):
    """The graph of Algorithm 1 (Line 6): unweighted, each edge once with i < j, no self-loops."""
    i, j = np.nonzero(np.triu(adjacency(distances, eps), 1))
    edge_index = torch.as_tensor(np.stack([i, j]), dtype=torch.long)
    return Data(edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[1], 1), num_nodes=len(distances))


def released_graph(distances, eps):
    """The legacy weighted graph of the released order_*/pyg_graphs_*.pt files (see the module docstring)."""
    n = len(distances)
    a = adjacency(distances, eps)
    similarity = TSER_MAX_SIMILARITY - distances / distances.max()
    w = a * similarity + TSER_MAX_SIMILARITY * np.eye(n)
    root = 1 / np.sqrt(w.sum(1))
    laplacian = np.eye(n) - (w * root[None, :]) * root[:, None]  # this operation order matches the files bit for bit
    weight = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
    i, j = np.nonzero(np.triu(a + np.eye(n)))
    return Data(x=torch.arange(n, dtype=torch.float32).view(-1, 1),
                edge_index=torch.as_tensor(np.stack([i, j]), dtype=torch.long),
                edge_attr=torch.as_tensor(weight[i, j], dtype=torch.float32).view(-1, 1), num_nodes=n)


def families(h0, h1):
    """{order folder: death times}: G0 by death time, G1 by birth, G0-1 = G0 then G1."""
    return {"order_0": h0, "order_1": h1[:, 1], "order_0-1": np.concatenate([h0, h1[:, 1]])}


def build(coords):
    """All outputs of one dataset as {relative path: object}, plus the distances and death times."""
    distances = geodesic_matrix(coords)
    h0, h1 = persistence(distances)
    fam = families(h0, h1)
    files = {"geodesic_km.npy": distances,
             os.path.join("baseline", "identity_operator.npy"): np.eye(len(coords), dtype=np.float32)}
    for name, eps in fam.items():
        files[os.path.join(name, "pyg_graphs_{ds}.pt")] = [released_graph(distances, e) for e in eps]
    for name, eps in fam.items():
        files[os.path.join(name, "clean", "pyg_graphs_{ds}.pt")] = [clean_graph(distances, e) for e in eps]
    return files, distances, fam


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(obj, np.ndarray):
        np.save(path, obj)
    else:
        torch.save(obj, path)


def threshold(graph, distances):
    edges = graph.edge_index[:, graph.edge_index[0] != graph.edge_index[1]]
    return float(distances[edges[0], edges[1]].max()) if edges.shape[1] else 0.0


def edge_set(graph):
    return set(map(tuple, graph.edge_index.t().tolist()))


def compare_graphs(ours, theirs, distances, eps, released):
    """Return a list of problems (empty = PASS) and a short summary."""
    if len(ours) != len(theirs):
        return [f"{len(ours)} graphs built, {len(theirs)} stored"], ""
    problems = []
    for k, (a, b) in enumerate(zip(ours, theirs)):
        if a.num_nodes != b.num_nodes or edge_set(a) != edge_set(b):
            problems.append(f"graph {k}: edge set differs ({len(edge_set(a))} vs {len(edge_set(b))} edges)")
        elif not torch.equal(a.edge_index, b.edge_index):
            problems.append(f"graph {k}: same edges in a different order")
        if threshold(b, distances) != eps[k]:
            problems.append(f"graph {k}: stored threshold {threshold(b, distances)!r} != death time {float(eps[k])!r}")
        if released:
            if not torch.equal(a.x, b.x):
                problems.append(f"graph {k}: node features differ")
            if a.edge_attr.shape != b.edge_attr.shape or not torch.equal(a.edge_attr, b.edge_attr):
                diff = (a.edge_attr - b.edge_attr).abs().max().item() if a.edge_attr.shape == b.edge_attr.shape else None
                problems.append(f"graph {k}: weights differ (max abs diff {diff})")
        elif not torch.equal(b.edge_attr, torch.ones(b.edge_index.shape[1], 1)):
            problems.append(f"graph {k}: clean weights are not all 1")
    summary = f"{len(ours)} graphs, edge sets + order + thresholds" + (" + weights + x" if released else "") + " equal"
    return problems, summary


def check(dataset, files, distances, fam, root):
    """Print PASS/FAIL per file; return the number of failures."""
    failures = 0

    def report(ok, rel, msg):
        nonlocal failures
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {dataset} {rel}: {msg}")

    for rel_template, ours in files.items():
        rel = rel_template.format(ds=dataset)
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            report(False, rel, "missing under --data_root")
            continue
        if rel.endswith(".npy"):
            theirs = np.load(path)
            if rel == "geodesic_km.npy":
                diff = float(np.abs(theirs - ours).max())
                report(diff <= GEODESIC_TOL_KM, rel, f"max abs diff {diff:.3g} km"
                       + (" (bit-identical)" if np.array_equal(theirs, ours) else ""))
            else:
                ok = theirs.dtype == ours.dtype and np.array_equal(theirs, ours)
                diff = float(np.abs(theirs - ours).max()) if theirs.shape == ours.shape else None
                report(ok, rel, f"{len(ours)} values equal" if ok else f"shape {theirs.shape} vs {ours.shape}, "
                       f"dtype {theirs.dtype} vs {ours.dtype}, max abs diff {diff}")
            continue
        theirs = torch.load(path, weights_only=False)
        parts = rel.split(os.sep)
        is_clean = len(parts) == 3  # order_<o>/clean/pyg_graphs_<ds>.pt
        problems, summary = compare_graphs(ours, theirs, distances, fam[parts[0]], released=not is_clean)
        report(not problems, rel, summary if not problems else "; ".join(problems[:3])
               + (f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""))
    for name in fam:  # build_clean_graphs.py applied to our released files gives our clean files
        rel = os.path.join(name, "pyg_graphs_{ds}.pt")
        ours_clean = files[os.path.join(name, "clean", "pyg_graphs_{ds}.pt")]
        ok = all(torch.equal(clean_released(r).edge_index, c.edge_index) and torch.equal(clean_released(r).edge_attr,
                 c.edge_attr) for r, c in zip(files[rel], ours_clean)) and len(files[rel]) == len(ours_clean)
        report(ok, f"build_clean_graphs.clean({rel.format(ds=dataset)})", "equals our clean family" if ok else "differs")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", required=True, help="reads <folder>/station_coords.npy; --check compares here")
    parser.add_argument("--out_root", help="write the outputs under <out_root>/<folder>/")
    parser.add_argument("--check", action="store_true", help="compare the built files with those under --data_root")
    parser.add_argument("--datasets", nargs="+", default=list(data.FOLDERS), choices=list(data.FOLDERS))
    args = parser.parse_args()
    if not args.out_root and not args.check:
        parser.error("give --out_root, --check or both")
    failures = 0
    for dataset in args.datasets:
        root = data.dataset_root(dict(config.DEFAULTS, dataset=dataset), args.data_root)
        files, distances, fam = build(np.load(os.path.join(root, "station_coords.npy")))
        print(f"{dataset}: {len(fam['order_0'])} H0 death times {np.round(fam['order_0'], 1).tolist()}")
        print(f"{dataset}: {len(fam['order_1'])} H1 death times {np.round(fam['order_1'], 1).tolist()}")
        if args.out_root:
            out = os.path.join(args.out_root, data.FOLDERS[dataset])
            for rel, obj in files.items():
                save(obj, os.path.join(out, rel.format(ds=dataset)))
            print(f"{dataset}: {len(files)} files -> {out}")
        if args.check:
            failures += check(dataset, files, distances, fam, root)
    if args.check:
        print("ALL PASS" if not failures else f"{failures} FAILED")
        raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
