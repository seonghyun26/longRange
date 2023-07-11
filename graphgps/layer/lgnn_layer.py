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
    
class graph2linegraph(nn.Module):
    # Convert graph into a line graph
    # NOTE: x, x0, edge_index, edge_attr
    def __init__(self):
        super().__init__()
    def forward(self, batch):
        # Save original information in batch
        batch.shape = batch.x.shape
        batch.org_x = batch.x
        batch.org_edge_index = batch.edge_index
        batch.org_edge_attr = batch.edge_attr
        if hasattr(batch, 'x0'):
            batch.org_x0 = batch.x0
        
        lg_node_idx = batch.edge_index.T
        batch.lg_node_idx = lg_node_idx
        # NOTE: Line graph node feature 
        lg_x = batch.x[lg_node_idx]
        batch.x = torch.reshape(lg_x, (lg_x.shape[0], -1))
        lg_x = None
        if hasattr(batch, 'edge_attr'):
            batch.x = torch.stack([batch.x, batch.edge_attr.repeat(1,2)], dim=2).mean(dim=2)
        # NOTE: Line graph node feature 2
        # batch.x = batch.edge_attr
            
        if hasattr(batch, 'x0'):
            lg_x0 = batch.x0[lg_node_idx]
            batch.x0 = torch.reshape(lg_x0, (lg_x0.shape[0], -1))
            lg_x0 = None


        # NOTE: line graph edge index
        # TODO: Better computation for finding lg_edge_idx?
        # startNode, endNode = lg_node_idx[:, 0], lg_node_idx[:, 1]   
        lg_edge_idx = torch.nonzero((lg_node_idx[:, 1, None] == lg_node_idx[:, 0]) & (lg_node_idx[:, 0, None] != lg_node_idx[:, 1]))
        batch.edge_index = lg_edge_idx.T
        
        
        # NOTE: line graph edge
        new_edge_idx = lg_node_idx[lg_edge_idx]
        lg_edge_attr = batch.org_x[new_edge_idx[:, 0, 1]].repeat(1,2)
        if hasattr(batch, 'edge_attr'):
            startEdge = new_edge_idx[:, 0]
            endEdge = new_edge_idx[:, 1]
            startIndices = torch.where(torch.all(batch.org_edge_index.T[:, None] == startEdge, dim=2))
            startEdgeAttr = batch.org_edge_attr[scatter(startIndices[0], startIndices[1])]
            endIndices = torch.where(torch.all(batch.org_edge_index.T[:, None] == endEdge, dim=2))
            endEdgeAttr = batch.org_edge_attr[scatter(endIndices[0], endIndices[1], dim=0)]
            lg_edge_attr = torch.stack([lg_edge_attr, torch.cat([startEdgeAttr, endEdgeAttr], dim=1)], dim=2).mean(dim=2)
            startIndices = None
            endIndices = None
            startEdgeAttr = None
            endEdgeAttr = None
        batch.edge_attr = lg_edge_attr
        
        # Garbage Collecting
        lg_edge_idx = None
        lg_edge_attr = None

        return batch

class linegraph2graph(nn.Module):
    # Convert line graph to graph
    # NOTE: x, x0, edge_index, edge_attr
    def __init__(self):
        super().__init__()
        
    def pad(self, tensor, originalShape):
        if originalShape[0] - tensor.shape[0] > 0:
            return torch.cat([tensor, torch.zeros(originalShape[0] - tensor.shape[0], originalShape[1], device=tensor.device)])
        else:
            return tensor
    
    def forward(self, batch):
        # Recover node feature
        # graph_node_idx = batch.lg_node_idx[:,1].unsqueeze(1)
        # graphX = scatter_mean(batch.x, graph_node_idx, dim=0)
        shape = batch.shape
        frontNode = scatter_mean(batch.x, batch.lg_node_idx[:,0], dim=0)[:, shape[1]:]
        frontNode = self.pad(frontNode, shape)
        backNode = scatter_mean(batch.x, batch.lg_node_idx[:,1], dim=0)[:, :shape[1]]
        backNode = self.pad(backNode, shape)
        batch.x = torch.add(
            frontNode,
            backNode
        )
        
        # Recover edge feature
        # batch.edge_attr = torch.stack([batch.org_edge_attr, batch.x], dim=2).mean(dim=2)
        shape = batch.org_edge_attr.shape
        frontEdge = scatter_mean(batch.edge_attr, batch. edge_index.T[:,0], dim=0)[:, shape[1]:]
        frontEdge = self.pad(frontEdge, shape)
        backEdge = scatter_mean(batch.edge_attr, batch.edge_index.T[:,1], dim=0)[:, :shape[1]]
        backEdge = self.pad(backEdge, shape)
        batch.edge_attr = torch.add(
            frontEdge,
            backEdge,
        )
        
        if hasattr(batch, 'x0'):
            batch.x0 = batch.org_x0
        batch.edge_index = batch.org_edge_index
        
        # Garbage Collecting 
        frontNode = None
        backNode = None
        graphX = None
        frontEdge = None
        backEdge = None
        graphEdgeAttr = None
        shape = None
        
        return batch