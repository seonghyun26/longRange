from torch_geometric.graphgym.register import register_config


def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    
    cfg.gnn.linegraph = False


register_config('custom_gnn', custom_gnn_cfg)
