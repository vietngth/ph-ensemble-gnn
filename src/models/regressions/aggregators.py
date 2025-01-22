import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv

class GraphExpert(nn.Module):
    def __init__(self, size_config, arch_config, train_config):
        super(GraphExpert, self).__init__()
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

        self.optimizer = train_config["optimizer"]
        self.reg_const = train_config["reg_const"]
        self.lr = train_config["lr"]

        
        self.conv1 = nn.Conv1d(self.in_size, self.hidden_size, kernel_size=self.kernel_size, stride=self.stride)
        self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size * 2, kernel_size=self.kernel_size, stride=self.stride)
        
        self.cnn_output_size = self._get_cnn_output_size(self.window , self.in_size)
        
        if self.conv_type == 'gcn':
            self.graph_conv1 = GCNConv(self.cnn_output_size + (2 if self.geo_feat else 0), self.hidden_size*2, bias=False)
            self.graph_conv2 = GCNConv(self.hidden_size*2, self.hidden_size*2, bias=False)
        elif self.conv_type == 'gat':
            self.graph_conv1 = GATConv(self.cnn_output_size + (2 if self.geo_feat else 0), self.hidden_size*2, heads=self.num_heads, bias=False)
            self.graph_conv2 = GATConv(self.hidden_size*2, self.hidden_size*2, heads=self.num_heads, bias=False, concat=False)
        elif self.conv_type == 'gatv2':
            self.graph_conv1 = GATv2Conv(self.cnn_output_size + (2 if self.geo_feat else 0), self.hidden_size*2, heads=self.num_heads, bias=False)
            self.graph_conv2 = GATv2Conv(self.hidden_size*2, self.hidden_size*2, heads=self.num_heads, bias=False, concat=False)
        else:
            raise ValueError("Invalid conv_type. Choose 'gcn', 'gat', or 'gatv2'.")

        self.fc1 = nn.Linear(self.hidden_size*self.num_nodes*2, self.hidden_size*2*2)

        self.fc21 = nn.Linear(self.hidden_size*2*2, self.num_nodes)
        self.fc22 = nn.Linear(self.hidden_size*2*2, self.num_nodes)
        self.fc23 = nn.Linear(self.hidden_size*2*2, self.num_nodes)
        self.fc24 = nn.Linear(self.hidden_size*2*2, self.num_nodes)
        self.fc25 = nn.Linear(self.hidden_size*2*2, self.num_nodes)

        self.dropout = nn.Dropout(0.4)

    def _get_cnn_output_size(self, seq_len, input_channels):
        with torch.no_grad():
            x = torch.randn(1, input_channels, seq_len)
            x = self.conv1(x)
            x = self.conv2(x)
        return x.shape[1] * x.shape[2]

    def forward(self, x, edge_index, edge_weight, graph_features=None, return_graph_output=False):
        batch_size, num_nodes, seq_len, input_channels = x.shape
        
        # print(f"Input shapes: x={x.shape}, edge_index={edge_index.shape}, edge_weight={edge_weight.shape}")
        
        # Reshape x for Conv1D: [batch_size * num_nodes, self.in_size, seq_len]
        x = x.view(batch_size * num_nodes, input_channels, seq_len)
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Reshape back: [batch_size, num_nodes, -1]
        x = x.view(batch_size, num_nodes, -1)

        if self.geo_feat and graph_features is not None:
            x = torch.cat([x, graph_features], dim=2)
        
        # print(f"After CNN: x shape = {x.shape}")
        
        # Ensure edge_index is within bounds
        mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]
        
        # print(f"After edge_index adjustment: edge_index shape = {edge_index.shape}, edge_weight shape = {edge_weight.shape}")
        
        # Apply Graph Convolutional layers
        if self.conv_type == 'gcn':
            x = torch.relu(self.graph_conv1(x, edge_index, edge_weight))
            x = torch.tanh(self.graph_conv2(x, edge_index, edge_weight))
        else:  # 'gat' or 'gatv2'
            x = torch.relu(self.graph_conv1(x, edge_index))
            x = torch.tanh(self.graph_conv2(x, edge_index))
        
        # Global average pooling
        graph_output = x.flatten().reshape(batch_size, -1)
        
        x_new = self.fc1(self.dropout(graph_output))

        outputs = []
        outputs.append(self.fc21(x_new))
        outputs.append(self.fc22(x_new))
        outputs.append(self.fc23(x_new))
        outputs.append(self.fc24(x_new))
        outputs.append(self.fc25(x_new))

        outputs = torch.stack(outputs)
        
        if return_graph_output:
            return outputs, x
        return outputs

    def l2_regularization(self):
        l2_loss = 0
        for layer in [self.graph_conv1, self.graph_conv2]:
            for param in layer.parameters():
                l2_loss += param.norm(2)
        return self.reg_const * l2_loss

class MultiGraphsAggregator(L.LightningModule):
    def __init__(self, size_config, arch_config, train_config):
        super(MultiGraphsAggregator, self).__init__()
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
        self.top_k = arch_config["top_k"]
        self.aggregator = arch_config["aggregator"]
        self.num_experts = arch_config["num_graphs"]
        self.edge_weights = arch_config["edge_weights"]
        self.edge_indices = arch_config["edge_indices"]
        self.lifetime_features = arch_config["lifetime_features"]

        self.optimizer = train_config["optimizer"]
        self.reg_const = train_config["reg_const"]
        self.lr = train_config["lr"]
  
            
        if self.top_k is None or self.top_k >= self.num_experts:
            # Use original version with all experts
            self.experts = nn.ModuleList([GraphExpert(size_config, arch_config, train_config) for _ in range(self.num_experts)])
            self.gating = nn.Linear(self.hidden_size * 2, self.num_experts)
        else:
            # Create only self.top_k experts with deterministic lifetime selection
            self.experts = nn.ModuleList([GraphExpert(size_config, arch_config, train_config) for _ in range(self.top_k)])
            # Multi-head attention for better graph selection
            self.num_attention_heads = 4
            self.attention_heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(self.window * 2),
                    nn.Linear(self.window * 2, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, 1)
                ) for _ in range(self.num_attention_heads)
            ])
            
    def forward(self, x, edge_indices, edge_weights, features=None, lifetimes=None):
        x = x.to(self.device)
        edge_indices = [ei.to(self.device) for ei in edge_indices]
        edge_weights = [ew.to(self.device) for ew in edge_weights]
        if features is not None:
            features = features.to(self.device)
            
        if self.top_k is None or self.top_k >= len(edge_indices):
            # Original version implementation
            l2_losses = []
            expert_outputs = []
            graph_outputs = []

            for expert, edge_index, edge_weight in zip(self.experts, edge_indices, edge_weights):
                if self.geo_feat and features is not None:
                    expert_output, graph_output = expert(x, edge_index, edge_weight, features, return_graph_output=True)
                else:
                    expert_output, graph_output = expert(x, edge_index, edge_weight, return_graph_output=True)
                        
                expert_outputs.append(expert_output)
                graph_outputs.append(graph_output)
                l2_losses.append(expert.l2_regularization())

            expert_outputs = torch.stack(expert_outputs)
            expert_outputs = torch.einsum('eobn->beno', expert_outputs)

            if self.aggregator == 'weighted':
                graph_output_avg = torch.stack(graph_outputs, dim=1).mean(dim=1)
                gating_weights = torch.softmax(self.gating(graph_output_avg), dim=-1).unsqueeze(1)
                gating_weights = torch.einsum('bone->beno', gating_weights)
                output = torch.sum(expert_outputs * gating_weights, dim=1)
            elif self.aggregator == 'lifetime':
                if lifetimes is None:
                    raise ValueError("Lifetime aggregation requires lifetime values")
                if not hasattr(self, '_lifetime_weights'):
                    lifetimes_tensor = torch.tensor(lifetimes, device=self.device) if not isinstance(lifetimes, torch.Tensor) else lifetimes
                    self._lifetime_weights = torch.softmax(torch.exp(0.1 * lifetimes_tensor), dim=0)
                lifetime_weights = self._lifetime_weights.view(1, -1, 1, 1).expand_as(expert_outputs)
                output = torch.sum(expert_outputs * lifetime_weights, dim=1)
            elif self.aggregator == 'average':
                output = torch.mean(expert_outputs, dim=1)
            elif self.aggregator == 'max':
                output = torch.max(expert_outputs, dim=1)[0]

        else:
            # Version with self.top_k experts and deterministic lifetime selection
            if lifetimes is not None and self.aggregator == 'lifetime':
                if not hasattr(self, '_lifetime_weights'):
                    lifetime_tensor = torch.tensor(lifetimes, device=self.device).float()
                    self._lifetime_weights = torch.softmax(torch.exp(0.1 * lifetime_tensor), dim=0)
                    _, self._k_indices = torch.topk(self._lifetime_weights, k=self.top_k)
                    self._k_weights = self._lifetime_weights[self._k_indices]
                    self._k_weights = torch.softmax(self._k_weights, dim=0)
            else:
                head_scores = []
                for head in self.attention_heads:
                    graph_scores = []
                    for edge_index, edge_weight in zip(edge_indices, edge_weights):
                        if self.expert_type == 'regression':
                            mean = torch.mean(x, dim=(0, 2))
                            std = torch.std(x, dim=(0, 2))
                            score_input = torch.cat([mean, std], dim=0).unsqueeze(0)
                        else:
                            score_input = torch.cat([
                                torch.mean(x, dim=(0, 2, 3)),
                                torch.std(x, dim=(0, 2, 3))
                            ], dim=0).unsqueeze(0)
                        score = head(score_input)
                        graph_scores.append(score)
                    scores = torch.cat(graph_scores)
                    head_scores.append(scores)
                
                attention_scores = torch.mean(torch.stack(head_scores), dim=0)
            

                scores = attention_scores    
                scores = torch.softmax(scores, dim=0)
                _, top_k_indices = torch.topk(scores.squeeze(), k=self.top_k)
            
            l2_losses = []
            expert_outputs = []
            graph_outputs = []
            
            for expert_idx, graph_idx in enumerate(top_k_indices if self.aggregator != 'lifetime' else self._k_indices):
                expert = self.experts[expert_idx]
                edge_index = edge_indices[graph_idx]
                edge_weight = edge_weights[graph_idx]
                
                if self.geo_feat and features is not None:
                    expert_output, graph_output = expert(x, edge_index, edge_weight, features, return_graph_output=True)
                else:
                    expert_output, graph_output = expert(x, edge_index, edge_weight, return_graph_output=True)

                        
                expert_outputs.append(expert_output)
                graph_outputs.append(graph_output)
                l2_losses.append(expert.l2_regularization())
            
            expert_outputs = torch.stack(expert_outputs)
            
            expert_outputs = torch.einsum('eobn->beno', expert_outputs)
            if self.aggregator == 'weighted':
                weights = scores[top_k_indices]
                weights = weights.view(1, -1, 1, 1).expand_as(expert_outputs)
                output = torch.sum(expert_outputs * weights, dim=1)
            elif self.aggregator == 'lifetime':
                lifetime_weights = self._k_weights.view(1, -1, 1, 1).expand_as(expert_outputs)
                output = torch.sum(expert_outputs * lifetime_weights, dim=1)
            elif self.aggregator == 'average':
                output = torch.mean(expert_outputs, dim=1)
            elif self.aggregator == 'max':
                output = torch.max(expert_outputs, dim=1)[0]

        total_l2_loss = sum(l2_losses)

        return output, total_l2_loss

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, alpha=0.9)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        if self.geo_feat:
            batch_x, batch_features, batch_y = [b.to(self.device) for b in batch]
            outputs, total_l2_reg = self(batch_x, self.edge_indices, self.edge_weights, batch_features, self.lifetime_features)
        else:
            batch_x, batch_y = [b.to(self.device) for b in batch]
            outputs, total_l2_reg = self(batch_x, self.edge_indices, self.edge_weights, lifetimes=self.lifetime_features)

        loss = F.mse_loss(outputs, batch_y) + total_l2_reg
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
        batch_y = None
        outputs = None
        all_predictions = []
        all_targets = []
            
        if self.geo_feat:
            batch_x, batch_features, batch_y = [b.to(self.device) for b in batch]
            outputs, _ = self(batch_x, self.edge_indices, self.edge_weights, batch_features, self.lifetime_features)
        else:
            batch_x, batch_y = [b.to(self.device) for b in batch]
            outputs, _ = self(batch_x, self.edge_indices, self.edge_weights, lifetimes=self.lifetime_features)
        
        loss = F.mse_loss(outputs, batch_y) 
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

        # Concatenate tensors and calculate metrics
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        mae = np.mean(np.abs(predictions - targets), axis=(0, 1))
        mse = np.mean((predictions - targets) ** 2, axis=(0, 1))
        rmse = np.sqrt(mse)
 
        return loss, mae, mse, rmse