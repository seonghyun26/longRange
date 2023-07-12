from torch_geometric.graphgym.register import register_config


def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    # Convert graph to line grpah before GNN layers.
    cfg.gnn.linegraph = False
    # line graph variant version
    cfg.gnn.lgvariant = 1


register_config('custom_gnn', custom_gnn_cfg)
