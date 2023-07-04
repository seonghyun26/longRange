import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_scatter

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer

def GMul(W, x):
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-3]
    J = W_size[-1]
    W_lst = W.split(1, 3)
    if N > 5000:
        output_lst = []
        for W in W_lst:
            output_lst.append(torch.bmm(W.squeeze(3),x))
        output = torch.cat(output_lst, 1)
    else:
        W = torch.cat(W_lst, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
        output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output

class gnn_atomic_lg(nn.Module):
    def __init__(self, feature_maps, J):
        # feature_maps: number of lists
        # J: hop neighbors
        super(gnn_atomic_lg, self).__init__()
        self.feature_maps = feature_maps
        self.J = J
        self.fcx2x_1 = nn.Linear(J * feature_maps[0], feature_maps[2])
        self.fcy2x_1 = nn.Linear(2 * feature_maps[1], feature_maps[2])
        self.fcx2x_2 = nn.Linear(J * feature_maps[0], feature_maps[2])
        self.fcy2x_2 = nn.Linear(2 * feature_maps[1], feature_maps[2])
        self.fcx2y_1 = nn.Linear(J * feature_maps[1], feature_maps[2])
        self.fcy2y_1 = nn.Linear(4 * feature_maps[2], feature_maps[2])
        self.fcx2y_2 = nn.Linear(J * feature_maps[1], feature_maps[2])
        self.fcy2y_2 = nn.Linear(4 * feature_maps[2], feature_maps[2])
        self.bn_x = nn.BatchNorm1d(2 * feature_maps[2])
        self.bn_y = nn.BatchNorm1d(2 * feature_maps[2])

    def forward(self, WW, x, WW_lg, y, P):
        xa1 = GMul(WW, x) # out has size (bs, N, num_inputs)
        xa1_size = xa1.size()
        xa1 = xa1.contiguous()
        xa1 = xa1.view(-1, self.J * self.feature_maps[0])
        xb1 = GMul(P, y)
        xb1 = xb1.contiguous()
        xb1 = xb1.view(-1, 2 * self.feature_maps[1])
        z1 = F.relu(self.fcx2x_1(xa1) + self.fcy2x_1(xb1)) # has size (bs*N, num_outputs)
        yl1 = self.fcx2x_2(xa1) + self.fcy2x_2(xb1)
        zb1 = torch.cat((yl1, z1), 1)
        # zc1 = self.bn_x(zb1.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
        zc1 = self.bn_x(zb1)
        zc1 = zc1.view(*xa1_size[:-1], 2 * self.feature_maps[2])
        x_output = zc1

        xda1 = GMul(WW_lg, y)
        xda1_size = xda1.size()
        xda1 = xda1.contiguous()
        xda1 = xda1.view(-1, self.J * self.feature_maps[1])
        xdb1 = GMul(torch.transpose(P, 2, 1), zc1)
        xdb1 = xdb1.contiguous()
        xdb1 = xdb1.view(-1, 4 * self.feature_maps[2])
        zd1 = F.relu(self.fcx2y_1(xda1) + self.fcy2y_1(xdb1))
        ydl1 = self.fcx2y_2(xda1) + self.fcy2y_2(xdb1)
        zdb1 = torch.cat((ydl1, zd1), 1)
        # zdc1 = self.bn_y(zdb1.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
        zdc1 = self.bn_y(zdb1)

        zdc1 = zdc1.view(*xda1_size[:-1], 2 * self.feature_maps[2])
        y_output = zdc1

        return WW, x_output, WW_lg, y_output, P

class gnn_atomic_lg_final(nn.Module):
    def __init__(self, feature_maps, J, n_classes):
        super(gnn_atomic_lg_final, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_inputs_2 = 2 * feature_maps[1]
        self.num_outputs = n_classes
        self.fcx2x_1 = nn.Linear(self.num_inputs, self.num_outputs)
        self.fcy2x_1 = nn.Linear(self.num_inputs_2, self.num_outputs)

    def forward(self, W, x, W_lg, y, P):
        x2x = GMul(W, x) # out has size (bs, N, num_inputs)
        x2x_size = x2x.size()
        x2x = x2x.contiguous()
        x2x = x2x.view(-1, self.num_inputs)
        y2x = GMul(P, y)
        y2x_size = x2x.size()
        y2x = y2x.contiguous()
        y2x = y2x.view(-1, self.num_inputs_2)
        xy2x = self.fcx2x_1(x2x) + self.fcy2x_1(y2x) # has size (bs*N, num_outputs)

        x_output = xy2x.view(*x2x_size[:-1], self.num_outputs)

        return W, x_output

class LGNNOriginalLayer(nn.Module):
    """LGNN Layer from ICLR 2019
    """
    def __init__(self, dim_in, dim_out, dropout, residual, num_features=10, num_layers=4, J=4, n_classes=2):
        super().__init__()
        print("FLAG")
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_in = [1, 1, num_features // 2]
        self.featuremap_mi = [num_features, num_features, num_features // 2]
        self.featuremap_end = [num_features, num_features, 1]
        self.layer0 = gnn_atomic_lg(self.featuremap_in, J)
        for i in range(num_layers):
            module = gnn_atomic_lg(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = gnn_atomic_lg_final(self.featuremap_end, J, n_classes)
        
        # NOTE: connect lgnn layer to self.model
        # self.model = pyg_nn.GCN2Conv(self.featuremap_mi[0], alpha=0.2)

    # NOTE: Original forward function for LGNN
    # def forward(self, W, x, W_lg, y, P):
    #     cur = self.layer0(W, x, W_lg, y, P)
    #     for i in range(self.num_layers):
    #         cur = self._modules['layer{}'.format(i+1)](*cur)
    #     out = self.layerlast(*cur)
    #     return out[1]

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index, bathc.edge_attr)

        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch
    
    
class LGNNLayer(nn.Module):
    """LGNN Layer
    """
    def __init__(self, dim_in, dim_out, dropout, residual, num_features=10, num_layers=4, J=4, n_classes=2):
        super().__init__()
        # print("FLAG - LGNN")
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        gin_nn = nn.Sequential(
            pyg_nn.Linear(dim_in, dim_out), nn.ReLU(),
            pyg_nn.Linear(dim_out, dim_out))
        
        self.model = pyg_nn.GINEConv(gin_nn)
        # layers = []
        # for _ in range(3):
        #     layers.append(pyg_nn.GINEConv(gin_nn))
        # self.model = torch.nn.Sequential(*layers)
        

    def forward(self, batch):
        # x_in = batch.x
        
        lg_node_idx, lg_edge_idx, lg_edge_attr_idx = self.graph2linegraph(batch.edge_index)
        
        linegraphX = batch.x[lg_node_idx].mean(dim=1)
        linegraphEdgeAttr = batch.x[lg_edge_attr_idx]
        for _ in range(3):
            linegraphX = self.model(
                linegraphX,
                lg_edge_idx,
                linegraphEdgeAttr
            )
        batch.x = self.linegraph2graph(linegraphX, lg_node_idx, batch.x.shape)
        
        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        # NOTE: Since we use line graph, skip redisual connection for now
        # So for now, no residual
        # if self.residual:
        #     batch.x = x_in + batch.x  # residual connection

        return batch    
    
    def graph2linegraph(self, edge_index):
        # Convert graph into a line graph, where batch.x is the features of graph nodes
        # Return node feature x, edge index, edge feature edge_attr
        
        # NOTE: New node feature index
        lg_node_idx = edge_index.T
        # newX = batch.x[self.graphNode2linegraphNodeMapper]
        # NOTE: 1 Concatenate two node features to make new one
        # 2 Average of two node features
        # newX = newX.reshape(newX.shape[0], -1)
        # newX = newX.mean(dim=1)

        # NOTE: New edge index
        # matchIdx = self.graphNode2linegraphNodeMapper[:,1] == self.graphNode2linegraphNodeMapper[:,0].unsqueeze(1)
        # matchNonLoopIdx = self.graphNode2linegraphNodeMapper[:,0] != self.graphNode2linegraphNodeMapper[:,1].unsqueeze(1)
        # lg_edge = torch.nonzero(matchIdx & matchNonLoopIdx)
        # lg_edge_idx = lg_edge[:, [1, 0]]
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
        # graphX = torch.zeros(batchShape)
        
        # graphX.scatter_reduce(0, graph_node_idx.expand(-1, linegraphX.shape[1]), linegraphX, reduce="mean")
        graphX = torch_scatter.scatter_mean(linegraphX, graph_node_idx, dim=0)
        graphX = torch.cat([graphX, torch.zeros(batchShape[0] - graphX.shape[0], batchShape[1], device=graphX.device)])

        return graphX
    
# register_layer('lgnn', LGNNLayer)
    