"""Experiment configuration: defaults and YAML loading."""
import os

import yaml

DEFAULTS = dict(
    dataset="ci", exp="aggr", graph_order=0,
    data_seed=1, fold_seed=1, train_ratio=0.8, k_fold=5, window=1000,
    graph_path=None, graph_family_path=None, inputs_path=None,
    batch_size=20, kernel_size=125, stride=2, hidden_size=32, dropout=0.4,
    mixture_loss="blended", normalize_inputs=True, keep_best_ckpt=True,
    lr_find=False, one_cycle_pct_start=0.3, one_cycle_final_div=1e4,
    optimizer="rmsprop", lr=1e-4, lr_schedule="constant", weight_decay=0.0, reg_const=1e-4,
    epochs=100, patience=10, min_delta=0.001,
    deterministic=True, precision="32-true", matmul_precision="high", run_tag=None,
)


def load_config(config_paths=(), overrides=()):
    """Return DEFAULTS updated in order by the YAML files and key=value overrides.

    The defaults are the TSER-GCN protocol used in the paper, so a config only lists what differs.
    """
    cfg = dict(DEFAULTS)
    for path in config_paths:
        with open(path) as f:
            cfg.update(checked(yaml.safe_load(f) or {}, path))
    for item in overrides:
        key, value = item.split("=", 1)
        cfg.update(checked({key: parse_value(value)}, "--set"))
    last = os.path.splitext(os.path.basename(config_paths[-1]))[0] if config_paths else "cli"
    cfg["run_tag"] = cfg["run_tag"] or f"{cfg['dataset']}_{last}"
    return cfg


def parse_value(text):
    """YAML scalar of a --set value (numbers, booleans, null, lists); anything unparsable stays a string."""
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(value, dict):
        return text
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def checked(values, source):
    unknown = sorted(set(values) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown config keys in {source}: {unknown}")
    return values
