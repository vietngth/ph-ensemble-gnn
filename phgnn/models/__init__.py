"""Model construction and checkpoint loading."""
from phgnn import data
from phgnn.models.baseline import BaselineModel
from phgnn.models.ensemble import MultiGraphsAggregator
from phgnn.models.kim_reference import KimGNNModel


def model_configs(cfg, input_shape, graphs=None):
    """Split a flat experiment config into the size/arch/train dictionaries the models take."""
    _, nodes, _, channels = input_shape
    size = dict(kernel_size=cfg["kernel_size"], stride=cfg["stride"], hidden_size=cfg["hidden_size"],
                num_nodes=nodes, in_size=channels)
    arch = dict(window=cfg["window"], dropout=cfg["dropout"], mixture_loss=cfg["mixture_loss"])
    if graphs is not None:
        arch["edge_indices"], arch["edge_weights"] = graphs
    train = dict(lr=cfg["lr"], reg_const=cfg["reg_const"], optimizer=cfg["optimizer"],
                 lr_schedule=cfg["lr_schedule"], weight_decay=cfg["weight_decay"],
                 pct_start=cfg["one_cycle_pct_start"], final_div=cfg["one_cycle_final_div"])
    return size, arch, train


def build_model(cfg, input_shape, data_root):
    if cfg["exp"] in ("baseline", "kim"):
        size, arch, train = model_configs(cfg, input_shape)
        cls = BaselineModel if cfg["exp"] == "baseline" else KimGNNModel
        return cls(size, arch, train, operator=data.load_baseline_operator(cfg, data_root))
    size, arch, train = model_configs(cfg, input_shape, data.load_graph_family(cfg, data_root))
    return MultiGraphsAggregator(size, arch, train)


def load_model(checkpoint, cfg, data_root, map_location="cpu"):
    """Load a checkpoint (baseline checkpoints get the graph matrix from the data directory)."""
    if cfg["exp"] not in ("baseline", "kim"):
        return MultiGraphsAggregator.load_from_checkpoint(checkpoint, map_location=map_location)
    cls = BaselineModel if cfg["exp"] == "baseline" else KimGNNModel
    return cls.load_from_checkpoint(checkpoint, map_location=map_location, strict=False,
                                    operator=data.load_baseline_operator(cfg, data_root))
