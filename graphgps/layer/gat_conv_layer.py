import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer

class GATConvLayer(nn.Module):
    """GCNII Layer from https://arxiv.org/abs/2007.02133.
    """
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        self.model = pyg_nn.GATConv(self.dim_in, self.dim_in)
        # alpha value is set using results from the GCNII paper

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)

        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch