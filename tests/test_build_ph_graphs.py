import os
import sys

import numpy as np
import pytest
import torch

from conftest import DATA_ROOT, REPO

pytest.importorskip("gudhi")
pytest.importorskip("geopy")
sys.path.insert(0, os.path.join(REPO, "experiments"))
import build_ph_graphs as ph  # noqa: E402


def test_persistence_of_a_square_with_a_far_point():
    # unit square (one loop, born at 1, filled at the diagonal sqrt 2) plus a point at distance 3 from a corner
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [-3, 0]], dtype=float)
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    h0, h1 = ph.persistence(distances)
    assert np.allclose(h0, [1, 1, 1, 3])
    assert np.allclose(h1, [[1, np.sqrt(2)]])
    graph = ph.clean_graph(distances, np.sqrt(2))
    assert sorted(map(tuple, graph.edge_index.t().tolist())) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert torch.equal(graph.edge_attr, torch.ones(6, 1))


def test_released_graph_cleans_to_the_threshold_graph():
    rng = np.random.default_rng(0)
    points = rng.random((12, 2)) * 100
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    for eps in ph.persistence(distances)[0]:
        released, clean = ph.released_graph(distances, eps), ph.clean_graph(distances, eps)
        assert torch.equal(ph.clean_released(released).edge_index, clean.edge_index)
        assert (released.edge_index[0] == released.edge_index[1]).sum() == 12
        assert released.edge_attr.min() == 0 and released.edge_attr.max() == 1


@pytest.mark.skipif(not os.path.isfile(os.path.join(DATA_ROOT, "central_it", "station_coords.npy")),
                    reason="data not available")
@pytest.mark.parametrize("dataset", ["ci", "cw"])
def test_build_reproduces_the_stored_graph_families(dataset, capsys):
    root = os.path.join(DATA_ROOT, ph.data.FOLDERS[dataset])
    files, distances, fam = ph.build(np.load(os.path.join(root, "station_coords.npy")))
    assert ph.check(dataset, files, distances, fam, root) == 0, capsys.readouterr().out
