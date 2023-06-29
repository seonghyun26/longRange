import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer

class gnn_atomic_lg(nn.Module):
    def __init__(self, feature_maps, J):
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

class LGNNLayer(nn.Module):
    """LGNN Layer
    """
    def __init__(self, num_features, num_layers=30, J=4, n_classes=2):
        super().__init__()
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
    
    
register_layer('lgnn', LGNNLayer)
    