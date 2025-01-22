import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.transforms import BaseTransform
from einops import rearrange
from einops import rearrange
from typing import Union, Tuple, List, Optional
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv

def expand_then_cat(tensors: Union[Tuple[Tensor, ...], List[Tensor]],
                    dim: int = -1) -> Tensor:
    """Match the dimensions of tensors in the input list and then concatenate.

    Args:
        tensors (list): Tensors to concatenate.
        dim (int): Dimension along which to concatenate.
            (default: -1)
    """
    shapes = [t.shape for t in tensors]
    expand_dims = torch.max(torch.tensor(shapes), 0).values
    expand_dims[dim] = -1
    tensors = [t.expand(*expand_dims) for t in tensors]
    return torch.cat(tensors, dim=dim)

def maybe_cat_exog(x, u, dim=-1):
    r"""
    Concatenate `x` and `u` if `u` is not `None`.

    We assume `x` to be a 4-dimensional tensor, if `u` has only 3 dimensions we
    assume it to be a global exog variable.

    Args:
        x: Input 4-d tensor.
        u: Optional exogenous variable.
        dim (int): Concatenation dimension.

    Returns:
        Concatenated `x` and `u`.
    """
    if u is not None:
        if u.dim() == 3:
            u = rearrange(u, 'b s f -> b s 1 f')
        x = expand_then_cat([x, u], dim)
    return x

class GCNDecoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int, 
                 output_size: int,
                 horizon: int = 1,
                 n_layers: int = 1,
                 activation: str = 'relu',
                 dropout: float = 0.,
                 conv_type: str = 'gcn',
                 num_heads: int = 4):
        super(GCNDecoder, self).__init__()
        
        graph_convs = []
        for i in range(n_layers):
            in_channels = input_size if i == 0 else hidden_size
            if conv_type == 'gcn':
                conv = GCNConv(in_channels, hidden_size, bias=False)
            elif conv_type == 'gat':
                conv = GATConv(in_channels if i == 0 else hidden_size * num_heads, 
                             hidden_size, heads=num_heads, bias=False,
                             concat=True if i < n_layers-1 else False)
            elif conv_type == 'gatv2':
                conv = GATv2Conv(in_channels if i == 0 else hidden_size * num_heads,
                               hidden_size, heads=num_heads, bias=False,
                               concat=True if i < n_layers-1 else False)
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")
            graph_convs.append(conv)
            
        self.convs = nn.ModuleList(graph_convs)
        if activation == 'relu':
            self.activation = torch.nn.functional.relu
        elif activation == 'tanh':
            self.activation = torch.nn.functional.tanh
        elif activation == 'sigmoid':
            self.activation = torch.nn.functional.sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size * horizon),
            nn.Unflatten(-1, (horizon, output_size))
        )

    def forward(self, x, edge_index, edge_weight=None):
        graph_outputs = []
        for conv in self.convs:
            x = self.activation(conv(x, edge_index, edge_weight))
            if self.training:
                x = self.dropout(x)
            graph_outputs.append(x)
        return self.readout(x), graph_outputs[-1]

class RNN(nn.Module):
    # https://github.com/TorchSpatiotemporal/tsl/blob/main/tsl/nn/blocks/encoders/recurrent/rnn.py
    """Simple RNN encoder with optional linear readout.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        exog_size (int, optional): Size of the optional exogenous variables.
        output_size (int, optional): Size of the optional readout.
        n_layers (int, optional): Number of hidden layers.
            (default: ``1``)
        cell (str, optional): Type of cell that should be use (options:
            ``'gru'``, ``'lstm'``). (default: ``'gru'``)
        dropout (float, optional): Dropout probability.
            (default: ``0.``)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = None,
                 output_size: int = None,
                 n_layers: int = 1,
                 return_only_last_state: bool = False,
                 cell: str = 'gru',
                 bias: bool = True,
                 dropout: float = 0.
                 ):
        super(RNN, self).__init__()

        self.return_only_last_state = return_only_last_state

        if cell == 'gru':
            cell = nn.GRU
        elif cell == 'lstm':
            cell = nn.LSTM
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')

        if exog_size is not None:
            input_size += exog_size

        self.rnn = cell(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=n_layers,
                        bias=bias,
                        dropout=dropout)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def forward(self, x: Tensor, u: Optional[Tensor] = None):
        """Process the input sequence :obj:`x` with optional exogenous variables
        :obj:`u`.

        Args:
            x (Tensor): Input data.
            u (Tensor): Exogenous data.

        Shapes:
            x: :math:`(B, T, N, F_x)` where :math:`B` is the batch dimension,
                :math:`T` is the number of time steps, :math:`N` is the number
                of nodes, and :math:`F_x` is the number of input features.
            u: :math:`(B, T, N, F_u)` or :math:`(B, T, F_u)` where :math:`B` is
                the batch dimension, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes (optional), and :math:`F_u` is
                the number of exogenous features.
        """
        # x: [batches, steps, nodes, features]
        x = maybe_cat_exog(x, u)
        b, *_ = x.size()
        x = rearrange(x, 'b s n f -> s (b n) f')
        x, *_ = self.rnn(x)
        # [steps batches * nodes, features] -> [steps batches, nodes, features]
        x = rearrange(x, 's (b n) f -> b s n f', b=b)
        if self.return_only_last_state:
            x = x[:, -1]
        if self.readout is not None:
            return self.readout(x)
        return x

