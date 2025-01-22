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
from src.models.regressions.baseline import BaselineModel
from src.models.regressions.aggregators import MultiGraphsAggregator
from src.datamodules.kfold_regression import EarthquakeKFoldDataModule
from src.utils import log_metrics
import wandb

# Uncomment this for better performance in case device has tensor cores
# torch.set_float32_matmul_precision("high")

def parse_args():
    parser = ArgumentParser(description="Run experiments with different models")

    parser.add_argument("--config", default=None, help="Path to YAML config file")

    # data config
    parser.add_argument("--dataset", type=str, default="ci", help="Support ci (central it), cw (central-west it)")
    parser.add_argument("--exp", type=str, default="baseline", help="Support baseline, max_aggr, mean_aggr, softmax_aggr, forecasting")
    parser.add_argument("--data_seed", type=int, default=1)
    parser.add_argument("--graph_order", type=str, default=None, help="0, 1, 0-1 or None (baseline)")
    parser.add_argument("--top_k", type=int, default=None, help="Reduce number of learners for traffic forecasting")
    parser.add_argument("--graph_path", type=str, default=None, help="Custom graph input. The default setting will load corresponding graphs in data folder")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--sigma", type=int, default=None, help="Take graphs that have long lifetime around the mean (acted as std)")
    parser.add_argument("--k_fold", type=int, default=5, help="K-Fold Validation on (Data Length * Train_Ratio) set")

    # size config
    parser.add_argument("--kernel_size", type=int, default=125, help="Kernel size of 1D-CNNs")
    parser.add_argument("--stride", type=int, default=2, help="Stride of 1D-CNNs filters")
    parser.add_argument("--reg_const", type=float, default=1e-4, help="L2 reg const")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--out_size", type=int, default=5)

    # architecture config
    parser.add_argument("--geo_feat", type=bool, default=True, help="Add geographical coordinates into node features")
    parser.add_argument("--window", type=int, default=1000, help="Window length of input. Default=1000/100=10(s)")
    parser.add_argument("--conv_type", type=str, default="gcn", help="Message Passing type, current support GCN, GAT & GATv2")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads of GAT/GATv2 Convs")
    parser.add_argument("--aggretator", type=str, default="weighted", help="weighted (Linear Softmax), max, average, lifetime")

    # train config
    parser.add_argument("--optimizer", type=str, default="RMSProp", help="Support RMSProp & Adam (not for baselines)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=30, help="Log every n steps for training")
    parser.add_argument("--project_name", type=str, default="regression", help="Wandb's project name")

    # ablation study config
    parser.add_argument("--cheb_k", type=int, default=None, help="For AWGCN only")
    parser.add_argument("--embed_dim", type=int, default=None, help="For AWGCN only")

    kwargs = parser.parse_args()
    return kwargs 

def create_model(datamodule, size_config, arch_config, train_config, ablation_config):
    item = next(iter(datamodule.train_dataloader()))
    size_config["num_nodes"] = item[0].shape[1]
    size_config["out_size"] = item[0].shape[2]
    size_config["in_size"] = item[0].shape[3]
    
    if train_config["exp"] != "baseline":
        return MultiGraphsAggregator(size_config, arch_config, train_config), size_config
        
    else:
        return BaselineModel(size_config, arch_config, train_config, ablation_config), size_config

def test_model(best_model_path, datamodule, model_type, logger):
    if model_type == "baseline":
        best_model = BaselineModel.load_from_checkpoint(best_model_path)
    else: 
        best_model = MultiGraphsAggregator.load_from_checkpoint(best_model_path)
     
    trainer = L.Trainer(
        accelerator="gpu", 
        devices=1,
        default_root_dir=models_path,
        logger=logger
    )
    trainer.test(best_model, dataloaders=datamodule.test_dataloader())
    output = trainer.predict(best_model, dataloaders=datamodule.test_dataloader())[0]
    wandb.finish()

    return output

if __name__ == "__main__":
    kwargs = vars(parse_args())
    config_path = kwargs["config"]
    if config_path is not None: 
        with open(config_path, "r") as f:
            conf = yaml.safe_load(f)
            for k, v in conf.items():
                kwargs[k] = v
    project_name = kwargs["project_name"]
    seed_everything(kwargs["data_seed"])
    dataset = kwargs["dataset"]
    graph_order = kwargs["graph_order"]
    sigma = kwargs["sigma"]

    # baseline
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder_name = "central_it"
    if dataset == "cw":
        data_folder_name = "central_west_it"

    inputs_path = None 
    graph_input_path = None 
    graph_features_path = None 
    targets_path = None 
    lifetimes_path = None

    # for aggregators
    edge_indices = None 
    edge_weights = None
    lifetime_features = None

    # 1 graph baseline
    if kwargs["exp"] == "baseline":
        data_dir = os.path.join(PROJECT_ROOT, 'data', data_folder_name, "baseline")
        inputs_path = os.path.join(os.path.split(data_dir)[0], f'inputs_{dataset}.npy')
        graph_input_path = kwargs["graph_path"]
        if graph_input_path is None:
            graph_input_path = os.path.join(data_dir, 'minmax_normalized_laplacian.npy')
        graph_features_path = os.path.join(os.path.dirname(graph_input_path), 'station_coords.npy')
        targets_path = os.path.join(os.path.split(data_dir)[0], 'targets.npy')
    # multiple graphs - our main method: multiple graphs discretized from filtrations of persistent homology
    else:
        order_combi_folder = f'order_{"-".join([str(n) for n in graph_order])}' if isinstance(graph_order, list) else f'order_{graph_order}'
        data_dir = os.path.join(PROJECT_ROOT, 'data', data_folder_name, order_combi_folder)
        if sigma is not None:
            data_dir = os.path.join(data_dir, f'sigma_{sigma}')
        inputs_path = os.path.join(os.path.split(data_dir)[0], f'inputs_{dataset}.npy')
        targets_path = inputs_path.replace(f'inputs_{dataset}', 'targets')
        graph_features_path = os.path.join(os.path.join(os.path.split(data_dir)[0], 'station_coords.npy'))

        pyg_graphs = torch.load(os.path.join(data_dir, f'pyg_graphs_{dataset}.pt'))
        lifetimes = np.load(os.path.join(data_dir, f'lifetimes_{dataset}.npy'), allow_pickle=True)
        edge_indices = [graph.edge_index for graph in pyg_graphs]
        edge_weights = [graph.edge_attr for graph in pyg_graphs]
        lifetime_features = [torch.tensor(lf, dtype=torch.float32)for lf in lifetimes] if kwargs["aggregator"] == 'lifetime' else None    

    data_paths = {
        "inputs_path": inputs_path,
        "graph_input_path": graph_input_path,
        "graph_features_path": graph_features_path,
        "targets_path": targets_path,
    }

    size_config = {
        "kernel_size": kwargs["kernel_size"],
        "stride": kwargs["stride"],
        "hidden_size": kwargs["hidden_size"]
    }

    data_config = {
        "dataset": dataset,
        "graph_order": kwargs["graph_order"],
        "batch_size": kwargs["batch_size"],
        "num_workers": kwargs["num_workers"],
        "data_seed": kwargs["data_seed"],
        "pin_memory": kwargs["pin_memory"],
        "train_ratio": kwargs["train_ratio"],
        "k_fold": kwargs["k_fold"]
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
        "conv_type": kwargs["conv_type"],
        "num_heads": kwargs["num_heads"],
        "num_graphs": num_graphs,
        "device": device,
        "edge_weights": edge_weights,
        "edge_indices": edge_indices,
        "lifetime_features": lifetime_features
    }

    train_config = {
        "optimizer": kwargs["optimizer"],
        "reg_const": kwargs["reg_const"],
        "lr": kwargs["lr"],
        "exp": kwargs["exp"]
    }
    
    ablation_config = None
    if kwargs["cheb_k"] is not None:
        ablation_config = {
            "cheb_k": kwargs["cheb_k"],
            "embed_dim": kwargs["embed_dim"]
        }
    exp = kwargs["exp"]
    aggr = kwargs["aggregator"]
    seed = kwargs["data_seed"]
    exp_name = f"{exp}_{aggr}_order_{graph_order}_{dataset}_seed_{seed}"
    # ablation studies
    if project_name != "regression":
        exp_name = f'{exp_name}/{kwargs["window"]}'

    experiment_path = f'./logs/regression/{exp_name}'
    models_path = os.path.join(experiment_path, "models")
    log_path = f'./logs/regression/{exp_name}'

    # To track the best model path and score across all folds
    best_model_path = None
    best_val_loss = float("inf")
    best_fold = 0
    fold_results = []
    for k in range(5):
        # Create a ModelCheckpoint callback for the current fold
        checkpoint_path = os.path.join(models_path, f"fold_{k}") 
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=checkpoint_path,
            filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        )

        # Config log
        logger = WandbLogger(
            project=project_name,
            name=os.path.join(exp_name, f"fold_{k}"),
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
            check_val_every_n_epoch=20,
            default_root_dir=models_path,
            logger=logger
        )

        datamodule = EarthquakeKFoldDataModule(k, data_paths, data_config, arch_config, ablation_config)
        datamodule.prepare_data()
        datamodule.setup()

        model, size_config = create_model(datamodule, size_config, arch_config, train_config, ablation_config)
        model = model.to(device)
        trainer.fit(model=model, datamodule=datamodule)

         # Save metrics and model path for this fold
        print(trainer.callback_metrics)
        fold_val_loss = trainer.callback_metrics["val_loss"].item()
        fold_results.append({"fold": k, "val_loss": fold_val_loss, "model_path": checkpoint_callback.best_model_path})

        # Update the best model across folds
        if fold_val_loss < best_val_loss:
            best_val_loss = fold_val_loss
            best_model_path = checkpoint_callback.best_model_path
            best_fold = k

        print(f"Fold {k+1} completed. Best model path: {best_model_path}")
        print(f"Validation loss: {fold_val_loss}")
        wandb.finish()
            

    # After all folds are complete
    print("Best model across all folds:")
    print(f"Path: {best_model_path}, Validation Loss: {best_val_loss}, Fold: {best_fold}")
    with open(os.path.join(models_path, f"best.txt"), "w") as f:
        f.write(best_model_path)

    best_exp = os.path.join(exp_name, f"fold_{best_fold}_inference")
    logger = WandbLogger(
        project=project_name,
        name=best_exp,
        save_dir=experiment_path,
    )

    output = test_model(best_model_path, datamodule, kwargs["exp"], logger)
    metrics = {
        "num_graphs": num_graphs,
        "exp": kwargs["exp"],
        "log_path": log_path,
        "mae": output[0],
        "mse": output[1],
        "rmse": output[2]
    }
    log_args = {**data_paths, **size_config, **data_config, **arch_config, **train_config, **metrics}
    log_metrics(**log_args)
