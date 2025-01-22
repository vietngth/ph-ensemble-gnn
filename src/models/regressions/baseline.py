import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import lightning as L
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv

from src.models.regressions.avwgcn import AVWGCN

logger = logging.getLogger(__name__)

def adj_matrix_to_edge_index_and_weight(adj_matrix):
    if not isinstance(adj_matrix, torch.Tensor):
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)

    if len(adj_matrix.shape) == 2:
        edge_index = torch.nonzero(adj_matrix).t()[:2]
        edge_weight = adj_matrix[edge_index[0], edge_index[1]]
    else:
        # Vectorized batch processing
        batch_indices = torch.nonzero(adj_matrix)
        edge_index = batch_indices[:, 1:3].t()
        edge_weight = adj_matrix[batch_indices[:, 0], batch_indices[:, 1], batch_indices[:, 2]]

    if edge_weight.numel() > 0:
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-8)
        edge_weight = edge_weight.flatten()

    return edge_index, edge_weight

class BaselineModel(L.LightningModule):
    def __init__(self, size_config, arch_config, train_config, ablation_config=None):
        super(BaselineModel, self).__init__()
        # save hyperparameters for logging
        self.save_hyperparameters()

        self.kernel_size = size_config["kernel_size"]
        self.stride = size_config["stride"]
        self.num_nodes = size_config["num_nodes"]
        self.out_size = size_config["out_size"]
        self.in_size = size_config["in_size"]
        self.hidden_size = size_config["hidden_size"]

        # embed geographical locations to node features
        self.geo_feat = arch_config["geo_feat"]
        self.window = arch_config["window"]
        self.conv_type = arch_config["conv_type"]
        self.num_heads = arch_config["num_heads"] if self.conv_type == "gat" else None

        # for ablation study of graph generation
        self.cheb_k = ablation_config["cheb_k"] if self.conv_type == "avwgcn" else None
        self.embed_dim = ablation_config["embed_dim"] if self.conv_type == "avwgcn" else None

        self.optimizer = train_config["optimizer"]
        self.reg_const = train_config["reg_const"]
        self.lr = train_config["lr"]

        # 1D-CNNs
        self.conv1 = nn.Conv1d(self.in_size, self.hidden_size , kernel_size=self.kernel_size, stride=self.stride)
        self.conv2 = nn.Conv1d(self.hidden_size , self.hidden_size  * 2, kernel_size=self.kernel_size, stride=self.stride)
        
        self.cnn_out_size = self._get_cnn_output_size(self.window, self.in_size)
        
        # GNNs
        if self.conv_type == 'gcn':
            self.graph_conv1 = GCNConv(self.cnn_out_size + (2 if self.geo_feat else 0), self.hidden_size * 2, bias=False)
            self.graph_conv2 = GCNConv(self.hidden_size * 2, self.hidden_size * 2, bias=False)
        elif self.conv_type == 'gat':
            self.graph_conv1 = GATConv(self.cnn_out_size + (2 if self.geo_feat else 0), self.hidden_size * 2, heads=self.num_heads, bias=False)
            self.graph_conv2 = GATConv(self.hidden_size * 2 * self.num_heads, self.hidden_size * 2, heads=self.num_heads, bias=False, concat=False)
        elif self.conv_type == 'gatv2':
            self.graph_conv1 = GATv2Conv(self.cnn_out_size + (2 if self.geo_feat else 0), self.hidden_size * 2, heads=self.num_heads, bias=False)
            self.graph_conv2 = GATv2Conv(self.hidden_size * 2 * self.num_heads, self.hidden_size * 2, heads=self.num_heads, bias=False, concat=False)
        elif self.conv_type == 'avw_gcn':
            if self.embed_dim is None:
                raise ValueError("embed_dim must be specified when using 'avw_gcn' as conv_type.")
            self.graph_conv1 = AVWGCN(dim_in=self.cnn_out_size + (2 if self.geo_feat else 0),
                                      dim_out=self.hidden_size * 2,
                                      cheb_k=self.cheb_k,
                                      embed_dim=self.embed_dim)
            self.graph_conv2 = AVWGCN(dim_in=self.hidden_size * 2,
                                      dim_out=self.hidden_size * 2,
                                      cheb_k=self.cheb_k,
                                      embed_dim=self.embed_dim)
        else:
            raise ValueError(f"Unsupported conv_type: {self.conv_type}")

        self.fc = nn.Linear(self.hidden_size * 2 * self.num_nodes, self.hidden_size * 4)
        self.output_layers = nn.ModuleList([nn.Linear(self.hidden_size * 4, self.num_nodes) for _ in range(5)])
        self.dropout = nn.Dropout(0.4)

    def _get_cnn_output_size(self, seq_len, input_channels):
        with torch.no_grad():
            x = torch.randn(1, input_channels, seq_len)
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
        return x.shape[1] * x.shape[2]
    
    def l2_regularization(self):
        l2_loss = 0
        for layer in [self.graph_conv1, self.graph_conv2]:
            for param in layer.parameters():
                l2_loss += param.norm(2)
        return self.reg_const * l2_loss

    def forward(self, x, graph_input, graph_features=None, node_embeddings=None):
        """
        Args:
            x: Input tensor of shape (batch_size, num_nodes, seq_len, channels)
            graph_input: Adjacency matrices of shape (batch_size, num_nodes, num_nodes)
            graph_features: Optional graph features of shape (batch_size, num_nodes, feature_dim)
            node_embeddings: Optional node embeddings of shape (num_nodes, embed_dim)
        
        Returns:
            outputs: Tensor of shape (batch_size, self.num_nodes, 5)
        """
        batch_size, num_nodes, seq_len, channels = x.shape
        
        logger.debug(f"Input shapes: x={x.shape}, graph_input={graph_input.shape}")
        if self.geo_feat and graph_features is not None:
            logger.debug(f"graph_features shape: {graph_features.shape}")
        if self.conv_type == 'avw_gcn' and node_embeddings is not None:
            logger.debug(f"node_embeddings shape: {node_embeddings.shape}")
        
        # Reshape and apply Conv1d layers
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, N, C, L)
        x = x.view(batch_size * num_nodes, channels, seq_len)  # (B*N, C, L)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.view(batch_size, num_nodes, -1)  # (B, N, hidden_features)
        
        if self.geo_feat and graph_features is not None:
            x = torch.cat([x, graph_features], dim=2)  # (B, N, hidden_features + feature_dim)
        
        outputs = []
        for i in range(batch_size):
            x_i = x[i]  # (N, hidden_features + feature_dim)
            graph_input_i = graph_input[i]  # (N, N)
            
            if self.conv_type in ['gcn', 'gat', 'gatv2']:
                edge_index, edge_weight = adj_matrix_to_edge_index_and_weight(graph_input_i)
                # Coalesce the edges to remove duplicates and sum duplicate edge weights
                edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())
            
            if self.conv_type == 'gcn':
                x_i = self.graph_conv1(x_i, edge_index, edge_weight)
                x_i = F.relu(x_i)
                x_i = self.graph_conv2(x_i, edge_index, edge_weight)
                x_i = F.tanh(x_i)
            elif self.conv_type == 'gat' or self.conv_type == "gatv2":
                x_i = self.graph_conv1(x_i, edge_index)
                x_i = F.relu(x_i)
                x_i = self.graph_conv2(x_i, edge_index)
                x_i = F.tanh(x_i)
            elif self.conv_type == 'avw_gcn':
                # Use node_embeddings directly, as it should already be of shape (num_nodes, embed_dim)
                if node_embeddings is None:
                    node_emb_i = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True).to(x.device)
                else:
                    node_emb_i = node_embeddings

                # Add batch dimension to x_i only
                x_i = x_i.unsqueeze(0)  # shape (1, N, dim_in)

                x_i = self.graph_conv1(x_i, node_emb_i)
                x_i = F.relu(x_i)
                x_i = self.graph_conv2(x_i, node_emb_i)
                x_i = F.tanh(x_i)

                x_i = x_i.squeeze(0)  # back to (N, hidden)
            else:
                raise ValueError(f"Unsupported conv_type: {self.conv_type}")
            
            outputs.append(x_i)
        
        x = torch.stack(outputs, dim=0)
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.fc(x)

        outputs = [layer(x) for layer in self.output_layers]
        outputs = torch.stack(outputs, dim=2)
        
        return outputs
    
    def configure_optimizers(self):
        # replicate setup of the baseline
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, alpha=0.9)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        out = None
        batch_y = None
        if self.geo_feat:
            if self.conv_type == 'avw_gcn':
                batch_x, batch_graph, batch_features, batch_y = batch
                batch_x, batch_graph, batch_features, batch_y = batch_x.to(self.device), batch_graph.to(self.device), batch_features.to(self.device), batch_y.to(self.device)
                out = self(batch_x, batch_graph, graph_features=batch_features, node_embeddings=self.embed_dim)
            else:
                batch_x, batch_graph, batch_features, batch_y = [b.to(self.device) for b in batch]
                out = self(batch_x, batch_graph, graph_features=batch_features)
        else:
            batch_x, batch_graph, batch_y = [b.to(self.device) for b in batch]
            out = self(batch_x, batch_graph)

        loss = F.mse_loss(out, batch_y) + self.l2_regularization()
        self.log("Training loss", loss, on_epoch=True, prog_bar=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        val_loss, mae, mse, rmse = self.custom_eval(batch, batch_idx)
        values = {"val_loss": val_loss, "Val MSE": np.mean(mse), "Val RMSE": np.mean(rmse), "Val MAE": np.mean(mae)} 
        self.log_dict(values, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        test_loss, mae, mse, rmse = self.custom_eval(batch, batch_idx)
        values = {"test_loss": test_loss, "Test MSE": np.mean(mse), "Test RMSE": np.mean(rmse), "Test MAE": np.mean(mae)} 
        self.log_dict(values)
        return test_loss
    
    def predict_step(self, batch, batch_idx):
        _, mae, mse, rmse = self.custom_eval(batch, batch_idx)
        return mae, mse, rmse
        
    def custom_eval(self, batch, batch_idx):
        # re-use for val, test & predict
        out = None
        batch_y = None
        all_predictions = []
        all_targets = []
            
        if self.geo_feat:
            if self.conv_type == 'avw_gcn':
                batch_x, batch_graph, batch_features, batch_y = batch
                batch_x, batch_graph, batch_features, batch_y = batch_x.to(self.device), batch_graph.to(self.device), batch_features.to(self.device), batch_y.to(self.device)
                out = self(batch_x, batch_graph, graph_features=batch_features, node_embeddings=self.embed_dim)
            else:
                batch_x, batch_graph, batch_features, batch_y = [b.to(self.device) for b in batch]
                out = self(batch_x, batch_graph, graph_features=batch_features)
        else:
            batch_x, batch_graph, batch_y = [b.to(self.device) for b in batch]
            out = self(batch_x, batch_graph)
        
        loss = F.mse_loss(out, batch_y) 
        all_predictions.append(out.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

        # Concatenate tensors and calculate metrics
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        mae = np.mean(np.abs(predictions - targets), axis=(0, 1))
        mse = np.mean((predictions - targets) ** 2, axis=(0, 1))
        rmse = np.sqrt(mse)
 
        return loss, mae, mse, rmse