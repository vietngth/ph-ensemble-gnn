import torch
import torch.nn as nn
import lightning as L
import einops
import numpy as np
from .modules import RNN, GCNDecoder
from src.utils import compute_mape

class GraphForecastingExpert(nn.Module):
    def __init__(self, size_config, arch_config, train_config):
        super(GraphForecastingExpert, self).__init__()
        self.num_nodes = size_config["num_nodes"]
        self.out_size = size_config["out_size"]
        self.in_size = size_config["in_size"]
        self.hidden_size = size_config["hidden_size"]
        self.rnn_layers = size_config["rnn_layers"]
        self.gcn_layers = size_config["gcn_layers"]

         # embed geographical locations to node features
        self.geo_feat = arch_config["geo_feat"]
        self.conv_type = arch_config["conv_type"]
        self.num_heads = arch_config["num_heads"] if self.conv_type == "gat" else None
        self.horizon = arch_config["horizon"]
        self.window = arch_config["window"]

        self.optimizer = train_config["optimizer"]
        self.reg_const = train_config["reg_const"]
        self.lr = train_config["lr"]
        
        total_input_channels = self.in_size
        if self.geo_feat:
            total_input_channels += 2  # Double channels to accommodate graph features

        self.input_encoder = torch.nn.Linear(total_input_channels, self.hidden_size)
        
        self.encoder = RNN(input_size=self.hidden_size,
                           hidden_size=self.hidden_size,
                           n_layers=self.rnn_layers,
                           dropout=0.2,
                           return_only_last_state=True)
        
        self.decoder = GCNDecoder(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.out_size,
            horizon=self.horizon,
            n_layers=self.gcn_layers,
            conv_type=self.conv_type,
            num_heads=self.num_heads,
            dropout=0.2
        )
        
    def forward(self, x, edge_index, edge_weight, graph_features=None, return_graph_output=False):
        if self.geo_feat and graph_features is not None:
            x = torch.cat([x, einops.repeat(graph_features, 'm n p -> m k n p', k=x.shape[1])], dim=-1)
        x = self.input_encoder(x)
        x = self.encoder(x)

        return self.decoder(x, edge_index, edge_weight)[0] if not return_graph_output else self.decoder(x, edge_index, edge_weight)

    def l2_regularization(self):
        l2_loss = 0
        for layer in [self.decoder]:
            for param in layer.parameters():
                l2_loss += param.norm(2)
        return self.reg_const * l2_loss

class MultiGraphsForecastingAggregator(L.LightningModule):
    def __init__(self, size_config, arch_config, train_config):
        super(MultiGraphsForecastingAggregator, self).__init__()
        self.save_hyperparameters()

        self.num_nodes = size_config['num_nodes']
        self.out_size = size_config['out_size']
        self.in_size = size_config['in_size']
        self.hidden_size = size_config['hidden_size']

         # embed geographical locations to node features
        self.geo_feat = arch_config['geo_feat']
        self.window = arch_config['window']
        self.conv_type = arch_config['conv_type']
        self.num_heads = arch_config['num_heads'] if self.conv_type == "gat" else None
        self.top_k = arch_config['top_k']
        self.aggregator = arch_config['aggregator']
        self.num_experts = arch_config['num_graphs']
        self.edge_weights = arch_config['edge_weights']
        self.edge_indices = arch_config['edge_indices']
        self.lifetime_features = arch_config['lifetime_features']
        self.thresholds = arch_config['thresholds']
        self.train_mean = arch_config['train_mean']
        self.train_std = arch_config['train_std']

        self.optimizer = train_config['optimizer']
        self.reg_const = train_config['reg_const']
        self.lr = train_config['lr']

        # for logging
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
  
            
        if self.top_k is None or self.top_k >= self.num_experts:
            # Use original version with all experts
            self.experts = nn.ModuleList([GraphForecastingExpert(size_config, arch_config, train_config) for _ in range(self.num_experts)])
            self.gating = nn.Linear(self.hidden_size, self.num_experts)
        else:
            # Create only self.top_k experts with deterministic lifetime selection
            self.experts = nn.ModuleList([GraphForecastingExpert(size_config, arch_config, train_config) for _ in range(self.top_k)])
            # Multi-head attention for better graph selection
            if self.thresholds is not None:
                self.gating = nn.Linear(self.hidden_size, self.num_experts)
            else:
                self.num_attention_heads = 4
                self.attention_heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(self.window * 2 + 2),
                    nn.Linear(self.window * 2 + 2, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, 1)
                    ) for _ in range(self.num_attention_heads)
                ])
            
    def forward(self, x, edge_indices, edge_weights, features=None, lifetimes=None, thresholds=None):
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
            expert_outputs = torch.einsum('ebntc->bnetc', expert_outputs)

            if self.aggregator == 'weighted':
                graph_outputs = torch.stack(graph_outputs, dim=1).mean(dim=1)
                gating_weights = torch.softmax(self.gating(graph_outputs), dim=-1).unsqueeze(1)
                gating_weights = torch.einsum('bcne->bnec', gating_weights).unsqueeze(-1)
                output = torch.sum(expert_outputs * gating_weights, dim=2)
            elif self.aggregator == 'lifetime':
                if lifetimes is None:
                    raise ValueError("Lifetime aggregation requires lifetime values")
                lifetimes_tensor = torch.tensor(lifetimes, device=self.device) if not isinstance(lifetimes, torch.Tensor) else lifetimes
                lifetime_weights = torch.softmax(torch.exp(0.1 * lifetimes_tensor), dim=0)
                lifetime_weights = lifetime_weights.view(1, 1, -1, 1, 1).expand_as(expert_outputs)
                output = torch.sum(expert_outputs * lifetime_weights, dim=2)
            elif self.aggregator == 'average':
                output = torch.mean(expert_outputs, dim=2)
            elif self.aggregator == 'max':
                output = torch.max(expert_outputs, dim=2)[0]

        else:
            # Version with self.top_k experts and deterministic lifetime selection
            if lifetimes is not None and self.aggregator == 'lifetime':
                if not hasattr(self, '_lifetime_weights'):
                    lifetime_tensor = torch.tensor(lifetimes, device=self.device).float()
                    self._lifetime_weights = torch.softmax(torch.exp(0.1 * lifetime_tensor), dim=0)
                    _, self._k_indices = torch.topk(self._lifetime_weights, k=self.top_k)
                    self._k_weights = self._lifetime_weights[self._k_indices]
                    self._k_weights = torch.softmax(self._k_weights, dim=0)
            elif thresholds is not None:
                if not hasattr(self, '_threshold_weights'):
                    threshold_tensor = torch.tensor(thresholds, device=self.device).float()
                    self._threshold_weights = 1-torch.softmax(threshold_tensor, dim=0)
                    _, self._k_indices = torch.topk(self._threshold_weights, k=self.top_k)
                top_k_indices = self._k_indices
            else:
                # Compute attention scores based on graph structure and node features
                head_scores = []
                for head in self.attention_heads:
                    graph_scores = []
                    for edge_index, edge_weight in zip(edge_indices, edge_weights):
                        # Get graph-level features
                        num_nodes = x.shape[2]
                        edge_density = edge_index.shape[1] / (num_nodes * num_nodes)
                        avg_weight = edge_weight.mean()
                        
                        # Combine with temporal features
                        temporal_feats = torch.cat([
                            torch.mean(x, dim=(0,2,3)), 
                            torch.std(x, dim=(0,2,3))
                        ]).to(x.device)
                        
                        graph_feat = torch.cat([
                            temporal_feats,
                            torch.tensor([edge_density, avg_weight], device=x.device)
                        ]).unsqueeze(0)
                        
                        score = head(graph_feat)
                        graph_scores.append(score)
                        
                    scores = torch.cat(graph_scores)
                    head_scores.append(scores)

                scores = torch.mean(torch.stack(head_scores), dim=0)
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
            expert_outputs = torch.einsum('ebntc->bnetc', expert_outputs)

            if thresholds is not None:
                if self.aggregator == 'weighted':
                    graph_outputs = torch.stack(graph_outputs, dim=1).mean(dim=1)
                    gating_weights = torch.softmax(self.gating(graph_outputs), dim=-1).unsqueeze(1)
                    gating_weights = torch.einsum('bcne->bnec', gating_weights).unsqueeze(-1)
                    output = torch.sum(expert_outputs * gating_weights[:,:,top_k_indices,:,:], dim=2)
                elif self.aggregator == 'lifetime':
                    if lifetimes is None:
                        raise ValueError("Lifetime aggregation requires lifetime values")
                    lifetimes_tensor = torch.tensor(lifetimes, device=self.device) if not isinstance(lifetimes, torch.Tensor) else lifetimes
                    lifetime_weights = torch.softmax(torch.exp(0.1 * lifetimes_tensor), dim=0)
                    lifetime_weights = lifetime_weights.view(1, 1, -1, 1, 1).expand_as(expert_outputs)
                    output = torch.sum(expert_outputs * lifetime_weights[:,:,top_k_indices,:,:], dim=2)
                elif self.aggregator == 'average':
                    output = torch.mean(expert_outputs, dim=2)
                elif self.aggregator == 'max':
                    output = torch.max(expert_outputs, dim=2)[0]
            else:
                if self.aggregator == 'weighted':
                    weights = scores[top_k_indices]
                    weights = weights.view(1, 1, -1, 1, 1).expand_as(expert_outputs)
                    output = torch.sum(expert_outputs * weights, dim=2)
                elif self.aggregator == 'lifetime' or self.aggregator == 'threshold':
                    lifetime_weights = self._k_weights.view(1, 1, -1, 1, 1).expand_as(expert_outputs)
                    output = torch.sum(expert_outputs * lifetime_weights, dim=2)
                elif self.aggregator == 'average':
                    output = torch.mean(expert_outputs, dim=2)
                elif self.aggregator == 'max':
                    output = torch.max(expert_outputs, dim=2)[0]

        total_l2_loss = sum(l2_losses)

        return output, total_l2_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        if self.geo_feat:
            batch_x, batch_features, batch_y = [b.to(self.device) for b in batch]
            outputs, total_l2_reg = self(batch_x, self.edge_indices, self.edge_weights, batch_features, self.lifetime_features, self.thresholds)
        else:
            batch_x, batch_y = [b.to(self.device) for b in batch]
            outputs, total_l2_reg = self(batch_x, self.edge_indices, self.edge_weights, lifetimes=self.lifetime_features, thresholds=self.thresholds)

        loss_fn = nn.L1Loss()
        loss = loss_fn(outputs.permute(0,2,1,3), batch_y) # total_l2_reg
        self.log("Training loss", loss, on_epoch=True, prog_bar=True)

        return loss 
    
    def validation_step(self, batch, batch_idx):
        val_loss, outputs, batch_y = self.custom_eval(batch, batch_idx)
        self.val_predictions.append(outputs)
        self.val_targets.append(batch_y)
        self.log("val_loss", val_loss)
        return val_loss
    
    def on_validation_epoch_end(self):
        # concatenate all predictions and targets
        all_predictions = np.concatenate(self.val_predictions, axis=0)
        all_targets = np.concatenate(self.val_targets, axis=0)

        # clear stored values for next epoch
        self.val_predictions.clear()
        self.val_targets.clear()

        # calculate metrics
        mae = np.mean(np.abs(all_predictions - all_targets), axis=(0, 2, 3))
        mse = np.mean((all_predictions - all_targets) ** 2, axis=(0, 2, 3))
        rmse = np.sqrt(mse)
        mape = compute_mape(all_predictions, all_targets)

        # Log metrics
        self.log_dict({"val_mae_epoch": np.mean(mae), 
                       "val_mse_epoch": np.mean(mse), 
                       "val_rmse_epoch": np.mean(rmse),
                       "val_mape_epoch": np.mean(mape)}, 
                       prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        test_loss, outputs, batch_y = self.custom_eval(batch, batch_idx)
        self.test_predictions.append(outputs)
        self.test_targets.append(batch_y)
        self.log("test_loss", test_loss)
        return test_loss

    def on_test_epoch_end(self):
        # Concatenate all predictions and targets
        all_predictions = np.concatenate(self.test_predictions, axis=0)
        all_targets = np.concatenate(self.test_targets, axis=0)

        # Calculate metrics
        mae = np.mean(np.abs(all_predictions - all_targets), axis=(0, 2, 3))
        mse = np.mean((all_predictions - all_targets) ** 2, axis=(0, 2, 3))
        rmse = np.sqrt(mse)
        mape = compute_mape(all_predictions, all_targets)

        # Log metrics
        self.log_dict({"test_mae": np.mean(mae), 
                       "test_mse": np.mean(mse), 
                       "test_rmse": np.mean(rmse),
                       "test_mape": np.mean(mape)}, 
                      prog_bar=True)

    def predict_step(self, batch, batch_idx):
        _, outputs, batch_y = self.custom_eval(batch, batch_idx)
        return outputs, batch_y
    
    def custom_eval(self, batch, batch_idx):
        # re-use for val, test & predict
        batch_y = None
        outputs = None
            
        if self.geo_feat:
            batch_x, batch_features, batch_y = [b.to(self.device) for b in batch]
            outputs, _ = self(batch_x, self.edge_indices, self.edge_weights, batch_features, self.lifetime_features, self.thresholds)
        else:
            batch_x, batch_y = [b.to(self.device) for b in batch]
            outputs, _ = self(batch_x, self.edge_indices, self.edge_weights, lifetimes=self.lifetime_features, thresholds=self.thresholds)
        
        loss_fn = nn.L1Loss()
        loss = loss_fn(outputs.permute(0,2,1,3), batch_y)
        # Denormalize predictions and targets for metric calculation
        outputs = outputs.permute(0,2,1,3).cpu().numpy() * (self.train_std + 1e-10) + self.train_mean
        batch_y = batch_y.cpu().numpy() * (self.train_std + 1e-10) + self.train_mean

        return loss, outputs, batch_y