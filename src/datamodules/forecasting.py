import lightning as L
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class TrafficForecastingDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_paths,
            data_config,
            arch_config,
        ):
        super().__init__()
        self.data_paths = data_paths
        self.data_config = data_config
        self.arch_config = arch_config

        self.train_mean = None 
        self.train_std = None
        self.data_train = None
        self.data_val = None 
        self.data_test = None 
    
    @property
    def get_train_mean(self):
        return self.train_mean
    
    @property
    def get_train_std(self):
        return self.train_std

    def setup(self, stage=None):
        if not self.data_train and not self.data_val and not self.data_test:
            window_size = self.arch_config["window"]
            horizon = self.arch_config["horizon"]

            graph_features = np.load(self.data_paths["graph_features_path"], allow_pickle=True)
            graph_features = torch.from_numpy(graph_features).float()
            
            inputs = np.load(self.data_paths["inputs_path"], allow_pickle=True)
            windows = []
            targets_list = []
            
            # For each possible starting position
            for i in range(len(inputs) - window_size - horizon + 1):
                # Get window_size steps as input sequence
                window = inputs[i:i+window_size]
                # Get next horizon steps as target sequence
                target = inputs[i+window_size:i+window_size+horizon]
                
                # Only keep sequences where we have full horizon steps ahead
                if len(target) == horizon:
                    windows.append(window)
                    targets_list.append(target)
            
            # Stack into arrays:
            # inputs: [num_sequences, window_size, num_nodes, channels]
            # targets: [num_sequences, horizon, num_nodes, channels]
            inputs = np.stack(windows) 
            targets = np.stack(targets_list)
            print(f"Created {len(inputs)} sequences with window_size={window_size} and horizon={horizon}")

            # Train - Test: 80% - 20% in chronological order
            x_train_val, x_test, y_train_val, y_test = train_test_split(inputs, targets, test_size=0.2, shuffle=False)    
            # Train - Val: 70% - 10% in chronological order
            x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.125, shuffle=False)

            # Normalize using training statistics
            self.train_mean = np.mean(x_train)
            self.train_std = np.std(x_train)
            
            # Normalize using training statistics
            x_train = (x_train - self.train_mean) / self.train_std
            x_val = (x_val - self.train_mean) / self.train_std
            x_test = (x_test - self.train_mean) / self.train_std
            
            # Also normalize targets for forecasting
            y_train = (y_train - self.train_mean) / self.train_std
            y_val = (y_val - self.train_mean) / self.train_std
            y_test = (y_test - self.train_mean) / self.train_std

            self.data_train = create_dataset(graph_features, x_train, y_train, self.arch_config["geo_feat"])
            self.data_val = create_dataset(graph_features, x_val, y_val, self.arch_config["geo_feat"])
            self.data_test = create_dataset(graph_features, x_test, y_test, self.arch_config["geo_feat"])

            print("Train set: ", len(self.data_train))
            print("Val set: ", len(self.data_val))
            print("Test set: ", len(self.data_test))
            print("Geo features:", self.arch_config["geo_feat"])
    
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
                           batch_size=self.data_config["batch_size"], 
                           num_workers=self.data_config["num_workers"], 
                           pin_memory=self.data_config["pin_memory"],
                        )
    
def create_dataset(g_feat, data_in, data_target, geo_feat=False):
    if geo_feat:
        return TensorDataset(torch.from_numpy(data_in).float(), 
                                    g_feat.repeat(len(data_in), 1, 1),
                                    torch.from_numpy(data_target).float())
    else:
        return TensorDataset(torch.from_numpy(data_in).float(), 
                                    torch.from_numpy(data_target).float())
