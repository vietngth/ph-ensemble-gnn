from argparse import ArgumentParser
import lightning as L
import os
import torch
import yaml
import numpy as np
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from src.models.forecasting.aggregators_forecasting import MultiGraphsForecastingAggregator
from src.datamodules.forecasting import TrafficForecastingDataModule
from src.utils import log_metrics, compute_mape
import wandb

# Uncomment this for better performance in case device has tensor cores
# torch.set_float32_matmul_precision("high")

def parse_args():
    parser = ArgumentParser(description="Run experiments with different models")

    parser.add_argument("--config", default=None, help="Path to YAML config file")

    # data config
    parser.add_argument("--dataset", type=str, default="bay_network", help="Support bay_network (PEMS-BAY), la_network (METR-LA)")
    parser.add_argument("--graph_order", type=str, default=None, help="0, 1, 0-1 or None (baseline)")
    parser.add_argument("--top_k", type=int, default=None, help="Reduce number of learners for traffic forecasting")
    parser.add_argument("--graph_path", type=str, default=None, help="Custom graph input. The default setting will load corresponding graphs in data folder")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--sigma", type=int, default=None, help="Take graphs that have long lifetime around the mean (acted as std)")

    # size config
    parser.add_argument("--reg_const", type=float, default=1e-3, help="L2 reg const")
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--out_size", type=int, default=1)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--gcn_layers", type=int, default=2)

    # architecture config
    parser.add_argument("--geo_feat", type=bool, default=True, help="Add geographical coordinates into node features")
    parser.add_argument("--window", type=int, default=1000, help="Slicing window size for time-series")
    parser.add_argument("--horizon", type=int, default=12, help="Time-steps, e.g., T=12 * 5(mins) = 60(mins)")
    parser.add_argument("--conv_type", type=str, default="gcn", help="Message Passing type, current support GCN, GAT & GATv2")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads of GAT/GATv2 Convs")
    parser.add_argument("--aggretator", type=str, default="weighted", help="weighted (Linear Softmax), max, average, lifetime")

    # train config
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=30, help="Log every n steps for training")
    parser.add_argument("--project_name", type=str, default="traffic-forecasting", help="Wandb's project name")


    kwargs = parser.parse_args()
    return kwargs 

def create_model(datamodule, size_config, arch_config, train_config):
    item = next(iter(datamodule.train_dataloader()))
    size_config["num_nodes"] = item[0].shape[2]
    size_config["in_size"] = item[0].shape[3]
    arch_config["train_mean"] = datamodule.train_mean
    arch_config["train_std"] = datamodule.train_std

    return MultiGraphsForecastingAggregator(size_config, arch_config, train_config), size_config

if __name__ == "__main__":
    kwargs = vars(parse_args())
    config_path = kwargs["config"]
    if config_path is not None: 
        with open(config_path, "r") as f:
            conf = yaml.safe_load(f)
            for k, v in conf.items():
                kwargs[k] = v
    project_name = kwargs["project_name"]
    # for reproducibility
    seed_everything(42)
    dataset = kwargs["dataset"]
    graph_order = kwargs["graph_order"]
    sigma = kwargs["sigma"]

    # baseline
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder_name = "bay_network"
    if dataset == "la_network":
        data_folder_name = "la_network"

    inputs_path = None 
    graph_input_path = None 
    graph_features_path = None 
    targets_path = None 
    lifetimes_path = None

    # for aggregators
    edge_indices = None 
    edge_weights = None
    lifetime_features = None

    # multiple graphs - our main method: multiple graphs discretized from filtrations of persistent homology
    order_combi_folder = f'order_{"-".join([str(n) for n in graph_order])}' if isinstance(graph_order, list) else f'order_{graph_order}'
    data_dir = os.path.join(PROJECT_ROOT, 'data', data_folder_name, order_combi_folder)
    if sigma is not None:
        data_dir = os.path.join(data_dir, f'sigma_{sigma}')
    else:
        data_dir = os.path.join(data_dir, 'no_sigma')

    suffix = dataset.split('_')[0] # la or bay
    inputs_path = os.path.join(os.path.split(os.path.split(data_dir)[0])[0], f'inputs_{suffix}.npy')
    targets_path = inputs_path.replace(f'inputs_{suffix}', 'targets')
    graph_features_path = os.path.join(os.path.split(os.path.split(data_dir)[0])[0], 'station_coords.npy')

    pyg_graphs = torch.load(os.path.join(data_dir, f'pyg_graphs_{dataset}.pt'))
    lifetimes = np.load(os.path.join(data_dir, f'lifetimes_{dataset}.npy'), allow_pickle=True)
    death_times = np.load(os.path.join(data_dir, f'death_times_{dataset}.npy'), allow_pickle=True)
    edge_indices = [graph.edge_index for graph in pyg_graphs]
    edge_weights = [graph.edge_attr for graph in pyg_graphs]
    lifetime_features = [torch.tensor(lf, dtype=torch.float32)for lf in lifetimes] if kwargs["aggregator"] == 'lifetime' else None    
    thresholds = [torch.tensor(dt, dtype=torch.float32) for dt in death_times] if kwargs["top_k_threshold"] == True else None
    data_paths = {
        "inputs_path": inputs_path,
        "graph_input_path": graph_input_path,
        "graph_features_path": graph_features_path,
        "targets_path": targets_path,
    }

    size_config = {
        "gcn_layers": kwargs["gcn_layers"],
        "rnn_layers": kwargs["rnn_layers"],
        "hidden_size": kwargs["hidden_size"],
        "out_size": kwargs["out_size"]
    }

    data_config = {
        "dataset": dataset,
        "graph_order": kwargs["graph_order"],
        "batch_size": kwargs["batch_size"],
        "num_workers": kwargs["num_workers"],
        "pin_memory": kwargs["pin_memory"],
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_graphs = 1
    if kwargs["exp"] != "baseline":
        num_graphs = len(pyg_graphs)
    print("Number of graphs:", num_graphs)
    arch_config = {
        "top_k": kwargs["top_k"],
        "aggregator": kwargs["aggregator"],
        "geo_feat": kwargs["geo_feat"],
        "window": kwargs["window"],
        "horizon": kwargs["horizon"],
        "conv_type": kwargs["conv_type"],
        "num_heads": kwargs["num_heads"],
        "num_graphs": num_graphs,
        "device": device,
        "edge_weights": edge_weights,
        "edge_indices": edge_indices,
        "lifetime_features": lifetime_features,
        "thresholds": thresholds
    }

    train_config = {
        "optimizer": kwargs["optimizer"],
        "reg_const": kwargs["reg_const"],
        "lr": kwargs["lr"],
        "exp": kwargs["exp"]
    }
    
    aggr = kwargs["aggregator"]
    exp_name = f"{aggr}_order_{graph_order}_{dataset}"

    experiment_path = f'./logs/forecasting/{exp_name}'
    models_path = os.path.join(experiment_path, "models")
    log_path = f'./logs/forecasting/{exp_name}'

    os.makedirs(models_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=models_path,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    # Config log
    logger = WandbLogger(
        project=project_name,
        name=exp_name,
        save_dir=experiment_path,
        log_model=True
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=kwargs["patience"])

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        deterministic=True,
        max_epochs=kwargs["epochs"], 
        callbacks=[early_stop_callback, checkpoint_callback], 
        log_every_n_steps=kwargs["log_every_n_steps"],
        check_val_every_n_epoch=30,
        default_root_dir=models_path,
        logger=logger,
    )

    datamodule = TrafficForecastingDataModule(data_paths, data_config, arch_config)
    datamodule.prepare_data()
    datamodule.setup()    

    model, size_config = create_model(datamodule, size_config, arch_config, train_config)
    model = model.to(device)
    trainer.fit(model=model, datamodule=datamodule)

    with open(os.path.join(models_path, f"best.txt"), "w") as f:
        f.write(checkpoint_callback.best_model_path)

    trainer.test(model, dataloaders=datamodule.test_dataloader())

    # double check metric aggregation manually
    output = trainer.predict(model, dataloaders=datamodule.test_dataloader())
    all_predictions = []
    all_targets = []

    for batch_output in output:
        predictions, targets = batch_output
        all_predictions.append(predictions)
        all_targets.append(targets)

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mae = np.mean(np.abs(all_predictions - all_targets), axis=(0, 2, 3))
    mse = np.mean((all_predictions - all_targets) ** 2, axis=(0, 2, 3))
    rmse = np.sqrt(mse)
    mape = compute_mape(all_predictions, all_targets)

    metrics = {
        "num_graphs": num_graphs,
        "exp": "forecasting",
        "log_path": log_path,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape
    }
    log_args = {**data_paths, **size_config, **data_config, **arch_config, **train_config, **metrics}
    log_metrics(**log_args)

    wandb.finish()
