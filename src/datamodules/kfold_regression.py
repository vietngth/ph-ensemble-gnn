import lightning as L
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from src.utils import normalize

class EarthquakeKFoldDataModule(L.LightningDataModule):
    def __init__(
            self,
            k,
            data_paths,
            data_config,
            arch_config,
            ablation_config = None
        ):
        super().__init__()
        self.data_paths = data_paths
        self.data_config = data_config
        self.arch_config = arch_config
        self.ablation_config = ablation_config
        # k-th fold
        self.k = k 
        self.data_train = None
        self.data_val = None 
        self.data_test = None 
    
    def setup(self, stage=None):
        if not self.data_train and not self.data_val and not self.data_test:
            graph_order = self.data_config["graph_order"]
            window_size = self.arch_config["window"]
            inputs = None
            graph_input = None

            # aggregators
            if graph_order is None:
                graph_input = np.load(self.data_paths["graph_input_path"], allow_pickle=True)
                
            graph_features = np.load(self.data_paths["graph_features_path"], allow_pickle=True)   
            inputs = np.load(self.data_paths["inputs_path"], allow_pickle=True)
            targets = np.load(self.data_paths["targets_path"], allow_pickle=True)
            inputs = inputs[:, :, :window_size, :]

            x_train, x_test, y_train, y_test = train_test_split(
                    inputs, targets, test_size= 1 - self.data_config["train_ratio"], 
                    random_state=self.data_config["data_seed"]
            )
            
            x_train = normalize(x_train)
            x_test = normalize(x_test)

            kf = KFold(n_splits=self.data_config["k_fold"],
                    shuffle=True, 
                    random_state=self.data_config["data_seed"]
            )
            
            # choose fold to train on
            all_splits = [k for k in kf.split(x_train)]
            # pick this according to outer loop in the training process
            # e.g., for k in range(5): ...
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            # transform (normalize)
            x_train, x_val = x_train[train_indexes], x_train[val_indexes]
            y_train, y_val = y_train[train_indexes], y_train[val_indexes]

            if graph_order is not None:
                self.data_train = create_dataset_aggr(graph_features, x_train, y_train, len(train_indexes), self.arch_config["geo_feat"])
                self.data_val = create_dataset_aggr(graph_features, x_val, y_val, len(val_indexes), self.arch_config["geo_feat"])
                self.data_test = create_dataset_aggr(graph_features, x_test, y_test, x_test.shape[0], self.arch_config["geo_feat"])
            else:
                self.data_train = create_dataset(graph_features, graph_input, x_train, y_train, len(train_indexes), self.arch_config["geo_feat"])
                self.data_val = create_dataset(graph_features, graph_input, x_val, y_val, len(val_indexes), self.arch_config["geo_feat"])
                self.data_test = create_dataset(graph_features, graph_input, x_test, y_test, x_test.shape[0], self.arch_config["geo_feat"])

            print("Train set: ", len(self.data_train))
            print("Val set: ", len(self.data_val))
            print("Test set: ", len(self.data_test))
    
    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, 
                           batch_size=self.data_config["batch_size"], 
                           num_workers=self.data_config["num_workers"], 
                           pin_memory=self.data_config["pin_memory"], 
                           shuffle=True,
                        )
    
    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, 
                           batch_size=self.data_config["batch_size"], 
                           num_workers=self.data_config["num_workers"], 
                           pin_memory=self.data_config["pin_memory"],
                        )
    
    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, 
                           batch_size=len(self.data_test), 
                           num_workers=self.data_config["num_workers"], 
                           pin_memory=self.data_config["pin_memory"],
                        )
    
def create_dataset(g_feat, g_in, data_in, data_target, data_length, geo_feat=False):
        if geo_feat: 
            return TensorDataset(
                torch.from_numpy(data_in).float(), 
                torch.from_numpy(g_in).float().repeat(data_length, 1, 1),
                torch.from_numpy(g_feat).float().repeat(data_length, 1, 1),
                torch.from_numpy(data_target).float()
            )
        return TensorDataset(
                torch.from_numpy(data_in).float(), 
                torch.from_numpy(g_in).float().repeat(data_length, 1, 1),
                torch.from_numpy(data_target).float()
            )


def create_dataset_aggr(g_feat, data_in, data_target, data_length, geo_feat=False):
        if geo_feat: 
            return TensorDataset(
                torch.from_numpy(data_in).float(), 
                torch.from_numpy(g_feat).float().repeat(data_length, 1, 1),
                torch.from_numpy(data_target).float().repeat(data_length, 1, 1),
            )
        return TensorDataset(
                torch.from_numpy(data_in).float(), 
                torch.from_numpy(data_target).float()
            )
    