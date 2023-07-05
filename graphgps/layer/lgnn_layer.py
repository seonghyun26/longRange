import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_scatter import scatter, scatter_mean

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.config import cfg

    
class LGNNGINELayer(nn.Module):
    """LGNNGINELayer
    """
    def __init__(self, dim_in, dim_out, dropout, residual, linegraph):
        super().__init__()
        # print("FLAG - LGNN")
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.linegraph = linegraph

        gin_nn = nn.Sequential(
            pyg_nn.Linear(dim_in, dim_out), nn.ReLU(),
            pyg_nn.Linear(dim_out, dim_out))
        
        self.model = pyg_nn.GINEConv(gin_nn)
        # layers = []
        # for _ in range(3):
        #     layers.append(pyg_nn.GINEConv(gin_nn))
        # self.model = torch.nn.Sequential(*layers)
        

    def forward(self, batch):
        x_in = batch.x
        
        lg_node_idx, lg_edge_idx, lg_edge_attr_idx = self.graph2linegraph(batch.edge_index)
        linegraphX = batch.x[lg_node_idx].mean(dim=1)
        linegraphEdge = batch.x[lg_edge_attr_idx]
        
        for _ in range(self.linegraph):
            linegraphX = self.model(
                linegraphX,
                lg_edge_idx,
                linegraphEdge
            )
        batch.x = self.linegraph2graph(linegraphX, lg_node_idx, batch.x.shape)
        
        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch    
    
    def graph2linegraph(self, edge_index):
        # Convert graph into a line graph, where batch.x is the features of graph nodes
        # Return node feature x, edge index, edge feature edge_attr
        
        # NOTE: New node feature index
        lg_node_idx = edge_index.T

        # NOTE: New edge index
        col0, col1 = lg_node_idx[:, 0], lg_node_idx[:, 1]   
        lg_edge_idx = torch.nonzero((col1[:, None] == col0[:, None].t()) & (col0[:, None] != col1[:, None]), as_tuple=False)
        del col0
        del col1
        
        # NOTE: New edge attribute index
        edge_attr_idx = lg_node_idx[lg_edge_idx[:, 0]][:, 1]
    
        return lg_node_idx, lg_edge_idx.T, edge_attr_idx
    
    def linegraph2graph(self, linegraphX, lg_node_idx, batchShape):
        # Convert line graph to graph, where x is the features of line graph nodes
        graph_node_idx = lg_node_idx[:,1].unsqueeze(1)
        graphX = scatter_mean(linegraphX, graph_node_idx, dim=0)
        graphX = torch.cat([graphX, torch.zeros(batchShape[0] - graphX.shape[0], batchShape[1], device=graphX.device)])

        return graphX
    

class LGNNGatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim), nn.ReLU(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        # if self.residual:
        #     x_in = x
        #     e_in = e
        
        lg_node_idx, edge_index, lg_edge_attr_idx = self.graph2linegraph(batch.edge_index)
        
        x = batch.x[lg_node_idx].mean(dim=1)
        e = batch.x[lg_edge_attr_idx]
        
        for _ in range(3):
            Ax = self.A(x)
            Bx = self.B(x)
            Ce = self.C(e)
            Dx = self.D(x)
            Ex = self.E(x)

            pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

            x, e = self.propagate(edge_index,
                                Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                                e=e, Ax=Ax,
                                PE=pe_LapPE)

            x = self.bn_node_x(x)
            e = self.bn_edge_e(e)

            x = F.relu(x)
            e = F.relu(e)

            x = F.dropout(x, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)

        graphX = self.linegraph2graph(x, lg_node_idx, batch.x.shape)
        # graphEdgeAttr = x
        
        # if self.residual:
        #     graphX = x_in + graphX
        #     graphEdgeAttr = e_in + graphEdgeAttr
            
        batch.x = graphX
        # batch.edge_attr = graphEdgeAttr

        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out
    
    def graph2linegraph(self, edge_index):
        # Convert graph into a line graph, where batch.x is the features of graph nodes
        # Return node feature x, edge index, edge feature edge_attr
        
        # NOTE: New node feature index
        lg_node_idx = edge_index.T

        # NOTE: New edge index
        col0, col1 = lg_node_idx[:, 0], lg_node_idx[:, 1]   
        lg_edge_idx = torch.nonzero((col1[:, None] == col0[:, None].t()) & (col0[:, None] != col1[:, None]), as_tuple=False)
        del col0
        del col1
        
        # NOTE: New edge attribute index
        edge_attr_idx = lg_node_idx[lg_edge_idx[:, 0]][:, 1]
    
        return lg_node_idx, lg_edge_idx.T, edge_attr_idx
    
    def linegraph2graph(self, linegraphX, lg_node_idx, batchShape):
        # Convert line graph to graph, where x is the features of line graph nodes
        graph_node_idx = lg_node_idx[:,1].unsqueeze(1)
        graphX = scatter_mean(linegraphX, graph_node_idx, dim=0)
        graphX = torch.cat([graphX, torch.zeros(batchShape[0] - graphX.shape[0], batchShape[1], device=graphX.device)])

        return graphX 
      
 
class graph2linegraph(nn.Module):
    # Convert graph into a line graph
    # NOTE: x, x0, edge_index, edge_attr
    def __init__(self):
        super().__init__()
    def forward(self, batch):
        batch.shape = batch.x.shape
        batch.org_x = batch.x
        if hasattr(batch, 'x0'):
            batch.org_x0 = batch.x0
        batch.org_edge_index = batch.edge_index
        batch.org_edge_attr = batch.edge_attr
        
        # NOTE: Line graph node feature 
        lg_node_idx = batch.edge_index.T
        batch.lg_node_idx = lg_node_idx
        batch.x = batch.x[lg_node_idx].mean(dim=1)
        if hasattr(batch, 'edge_attr'):
            batch.x = torch.stack([batch.x, batch.edge_attr], dim=2).mean(dim=2)
        if hasattr(batch, 'x0'):
            batch.x0 = batch.x0[lg_node_idx].mean(dim=1)

        # NOTE: New edge index
        # TODO: Better computation for finding lg_edge_idx?
        col0, col1 = lg_node_idx[:, 0], lg_node_idx[:, 1]   
        lg_edge_idx = torch.nonzero((col1[:, None] == col0[:, None].t()) & (col0[:, None] != col1[:, None]), as_tuple=False)
        batch.edge_index = lg_edge_idx.T
        col0 = None
        col1 = None
        
        # NOTE: New edge attribute index
        lg_edge_attr_idx = lg_node_idx[lg_edge_idx[:, 0]][:, 1]
        batch.edge_attr = batch.x[lg_edge_attr_idx]

        return batch

class linegraph2graph(nn.Module):
    # Convert line graph to graph
    # NOTE: x, x0, edge_index, edge_attr
    def __init__(self):
        super().__init__()
    
    def forward(self, batch):
        if hasattr(batch, 'x0'):
            batch.x0 = batch.org_x0
        batch.edge_index = batch.org_edge_index
        
        # batch.edge_attr = batch.org_edge_attr
        batch.edge_attr = torch.stack([batch.org_edge_attr, batch.x], dim=2).mean(dim=2)
        
        graph_node_idx = batch.lg_node_idx[:,1].unsqueeze(1)
        graphX = scatter_mean(batch.x, graph_node_idx, dim=0)
        graphX = torch.cat([graphX, torch.zeros(batch.shape[0] - graphX.shape[0], batch.shape[1], device=graphX.device)])
        batch.x = graphX
        
        return batch