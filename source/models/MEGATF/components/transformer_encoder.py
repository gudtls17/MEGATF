import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, Linear, LayerNorm
from torch import Tensor
from typing import Optional, Any, Union, Callable
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn

class GraphTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, n_classes, nheads: int, dim_feedforward: int = 2048, dropout: float = 0.1, mha_conv: str = "EdgeGAT",
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
                 layer_norm_eps: float = 1e-5,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GraphTransformerEncoder, self).__init__()
        
        print("nHeads: ", nheads)
        # select Graph Multi-head Attention convolution
        if mha_conv == "EdgeGAT":
            self.gmha = "EdgeGAT"
            print(f"EdgeGAT GMHA: {self.gmha}")
            self.gatconv = dglnn.EdgeGATConv(in_feats=hidden_dim, edge_feats=1, out_feats=hidden_dim, num_heads=nheads, feat_drop=dropout, allow_zero_in_degree=True)
        elif mha_conv == "GAT":
            self.gmha = "GAT"
            print(f"GAT GMHA: {self.gmha}")
            self.gatconv = dglnn.GATConv(hidden_dim, hidden_dim, num_heads=nheads, feat_drop=dropout)
        elif mha_conv == "GATv2":
            self.gmha = "GATv2"
            print(f"GATv2 GMHA: {self.gmha}")
            self.gatconv = dglnn.GATv2Conv(hidden_dim, hidden_dim, num_heads=nheads, feat_drop=dropout)
        self.linear_mha = nn.Linear(hidden_dim*nheads, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.classify = nn.Linear(hidden_dim, n_classes)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim, **factory_kwargs)
        
        self.activation = activation

    def forward(self, g, h, e, batch_size):
        h_init = h  # for residual, (num of whole batch node, hidden_dim) 
        # Graph Multi-head Attention
        if self.gmha == "EdgeGAT":
            h = self.activation(self.gatconv(g, h, e))
        elif self.gmha == "GAT" or self.gmha == "GATv2":
            h = self.activation(self.gatconv(g, h)) # Graph Multi-head Attention, (num of whole batch node, nheads, hidden_dim)
        h = h.reshape(h.shape[0], -1) # (num of whole batch node, nheads, hidden_dim) -> (num of whole batch node, nheads*hidden_dim)
        h = self.dropout1(self.activation(self.linear_mha(h))) # (num of whole batch node, nheads*hidden_dim) -> (num of whole batch node, hidden_dim)
        
        # Add & Norm
        h = self.norm1(h_init + h)  
        # print('h1', h.shape)
        h = h.reshape(batch_size, -1, h.shape[1])  # (batch_size, num of node, hidden_dim)
        
        # Feedforward
        h = self.norm2(h + self._ff_block(h))
        
        # print("h", h.shape)
        return h
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    def get_attention_weights(self) -> Optional[Tensor]:
        return None