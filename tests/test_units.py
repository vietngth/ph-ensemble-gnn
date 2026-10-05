import os
import sys

import numpy as np
import torch
from torch import nn

from phgnn import config, metrics, models
from phgnn.models import ensemble

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))


def test_encoder_convolves_each_component_over_time():
    conv1, conv2 = nn.Conv1d(3, 4, 5), nn.Conv1d(4, 8, 5)
    x = torch.randn(2, 6, 50, 3)
    expected = torch.relu(conv2(torch.relu(conv1(x.permute(0, 1, 3, 2).reshape(12, 3, 50))))).view(2, 6, -1)
    assert torch.allclose(ensemble.encode_waveforms(conv1, conv2, x), expected)


def test_unknown_config_keys_are_rejected(tmp_path):
    path = tmp_path / "c.yml"
    path.write_text("gate_mode: mean\n")
    try:
        config.load_config([str(path)])
    except ValueError as error:
        assert "gate_mode" in str(error)
    else:
        raise AssertionError("obsolete key accepted")


def test_regression_metrics_per_output():
    pred = np.zeros((2, 3, 2))
    target = np.stack([np.ones((2, 3)), 2 * np.ones((2, 3))], axis=-1)
    scores = metrics.regression_metrics(pred, target)
    assert scores["mae"] == [1.0, 2.0] and scores["mse"] == [1.0, 4.0] and scores["rmse"] == [1.0, 2.0]


def test_set_values_parse_numbers_and_keep_path_templates():
    from phgnn.config import parse_value
    assert parse_value("1e-4") == 1e-4 and parse_value("3") == 3 and parse_value("null") is None
    assert parse_value("{folder}/baseline/x.npy") == "{folder}/baseline/x.npy"
    assert parse_value("bf16-mixed") == "bf16-mixed" and parse_value("[0, 1]") == [0, 1]


def test_competitive_loss_is_the_gate_weighted_expert_error():
    """Jacobs et al. (1991) Eq. 1.2: sum_k alpha_k ||y - y_k||^2; the prediction stays the blended one."""
    stacked = torch.tensor([[[[1.0]], [[3.0]]]])            # B=1, K=2, N=1, 1 output
    weights = torch.tensor([[[[0.25]], [[0.75]]]])
    y = torch.tensor([[[2.0]]])
    competitive = (weights * (stacked - y[:, None]) ** 2).sum(dim=1).mean(dim=(0, 1)).sum()
    blended = ((stacked * weights).sum(dim=1) - y) ** 2
    assert torch.isclose(competitive, torch.tensor(1.0)) and torch.isclose(blended.sum(), torch.tensor(0.25))


def test_input_paths_can_depend_on_the_data_seed():
    from phgnn import data
    cfg = dict(config.DEFAULTS, dataset="cw", data_seed=7, inputs_path="{folder}/sfa/inputs_{dataset}_s{seed}.npy")
    assert data.data_file(cfg, "inputs_path", "/d") == "/d/central_west_it/sfa/inputs_cw_s7.npy"


def test_valley_picks_the_longest_descent_like_fastai():
    from phgnn.lr_find import valley
    lrs = [10 ** (-7 + 0.1 * i) for i in range(60)]
    losses = [2.0] * 10 + [2.0 - 0.05 * i for i in range(30)] + [0.6 + 0.2 * i for i in range(20)]
    lr = valley(lrs, losses)
    assert lrs[10] < lr < lrs[40]          # inside the descent, before the minimum


def test_l2_covers_the_shared_encoder_kernels():
    cfg = dict(config.DEFAULTS, hidden_size=4, window=200, kernel_size=25)
    edges = [torch.tensor([[0, 1], [1, 0]])] * 2
    weights = [torch.ones(2)] * 2
    torch.manual_seed(0)
    size, arch, train = models.model_configs(cfg, (1, 3, 200, 3), (edges, weights))
    model = ensemble.MultiGraphsAggregator(size, arch, train)
    x, coords = torch.randn(2, 3, 200, 3), torch.randn(2, 3, 2)
    before = model.predict(x, coords)[1]
    with torch.no_grad():
        model.shared_conv1.weight.mul_(10)
    assert model.predict(x, coords)[1] > before


def test_results_are_not_reused_across_configs():
    import train
    stored = {k: v for k, v in config.DEFAULTS.items() if k != "lr"}
    assert train.config_changes(stored, dict(config.DEFAULTS, lr=1e-3)) == []
    assert train.config_changes(dict(stored, lr=1e-3), dict(config.DEFAULTS)) == ["lr"]
    assert train.config_changes(dict(stored, keep_best_ckpt=False), dict(config.DEFAULTS)) == []


def test_kim_gcn_dense_aggregation_matches_message_passing():
    from phgnn.models import kim_reference
    torch.manual_seed(0)
    n, f = 6, 4
    adjacency = torch.rand(n, n) * (torch.rand(n, n) > 0.5)
    h = torch.randn(n, f)
    layer = kim_reference.GCN(f, 3, None)
    # DGL update_all(copy_src, sum) with edge weights: node i receives sum over edges j -> i of w_ji * h_j
    expected = torch.stack([sum(adjacency[j, i] * h[j] for j in range(n)) for i in range(n)])
    assert torch.allclose(layer(adjacency, h), layer.linear(expected), atol=1e-6)


def test_kim_gnn_predicts_five_ims_per_station():
    from phgnn.models import kim_reference
    cfg = dict(config.DEFAULTS, exp="kim", window=400)
    size, arch, train = models.model_configs(cfg, (1, 7, 400, 3))
    model = kim_reference.KimGNNModel(size, arch, train, operator=torch.eye(7))
    pred, l2 = model.predict(torch.randn(2, 7, 400, 3), torch.randn(2, 7, 2))
    assert pred.shape == (2, 7, 5) and float(l2) == 0.0
