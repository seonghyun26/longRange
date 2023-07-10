import datetime
import os
import torch
import logging

import graphgps  # noqa, register custom modules

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_agg_dir, set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optimizer import create_optimizer, \
    create_scheduler, OptimizerConfig, SchedulerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger



import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)
        

def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices

def effectiveResistance(edge_index):
    num_nodes = edge_index.max().item() + 1
    adj_matrix = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (num_nodes, num_nodes)).to_dense()
    
    degree_matrix = torch.diag(adj_matrix.sum(dim=1)+adj_matrix.sum(dim=0))
    laplacian_matrix = (degree_matrix - adj_matrix)
    # laplacian_pseudoinv = torch.pinverse(laplacian_matrix)
    laplacian_pseudoinv = torch.from_numpy(np.linalg.pinv(laplacian_matrix))
    # sign = np.sign(laplacian_pseudoinv[0][0])

    effective_resistance = num_nodes * laplacian_pseudoinv.trace()
    # effective_resistance= np.sign(laplacian_pseudoinv[0][0]) * num_nodes * laplacian_pseudoinv.trace()
    assert(effective_resistance > 0)

    return effective_resistance


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        # cfg.device = 2
        # Set machine learning pipeline
        loaders = create_loader()
        
        
        batchesTrain = [batch for batch in loaders[0]]
        batchesValid = [batch for batch in loaders[1]]
        batchesTest = [batch for batch in loaders[2]]
        
        split = ["train", "valid", "test"]
        color = ["red", "green", "blue"]
        for idxBatches, batches in enumerate([batchesTrain, batchesValid, batchesTest]):
            datasetName = cfg.dataset.name + "_"+ split[idxBatches]
            # + "_"+ "dataset"
            print("Effective resistance for "+ datasetName + ": Processing...")
        
                
            graph_effective_resistance_list = []
            linegraph_effective_resistance_list = []
            for idxBatch, batch in enumerate(tqdm(batches)):
                edge_index = batch.edge_index
                ptr = batch.ptr
                prevPt = 0;
                
                for pt in ptr[1:]:
                    idx = max(torch.nonzero(edge_index[0,:]<pt)).item()+1
                    graph_edge_idx = edge_index[:,:idx]-prevPt

                    lg_node_idx = graph_edge_idx.T
                    col0, col1 = lg_node_idx[:, 0], lg_node_idx[:, 1]   
                    linegraph_edge_idx = torch.nonzero((lg_node_idx[:, 1, None] == lg_node_idx[:, 0]) & (lg_node_idx[:, 0, None] != lg_node_idx[:, 1])).T
                    col0 = None
                    col1 = None
                    
                    er_graph = effectiveResistance(graph_edge_idx).item()
                    er_linegraph = effectiveResistance(linegraph_edge_idx).item()
                    
                    graph_effective_resistance_list.append(math.log2(er_graph))  
                    linegraph_effective_resistance_list.append(math.log2(er_linegraph))
                    
                    edge_index = edge_index[:,idx:]
                    prevPt = pt

            graph_avg_er = np.mean(graph_effective_resistance_list)
            linegraph_avg_er = np.mean(linegraph_effective_resistance_list)
            print(graph_avg_er)
            print(linegraph_avg_er)
            
            plt.figure()
            plt.hist(graph_effective_resistance_list, bins='auto', color='lightgreen', alpha=0.7, label="graph")
            plt.hist(linegraph_effective_resistance_list, bins='auto', color='mediumpurple', alpha=0.7, label="line graph")
            plt.xlabel('log2(Effective Resistance)')
            plt.ylabel('count')
            plt.legend()
            
            plt.title(datasetName + " (graph, linegraph)")
            plt.savefig("ER_" + datasetName + "(graph, line graphs).png")
            # if cfg.gnn.linegraph:
            #     plt.title("Distribution of Effective Resistance "+datasetName+" (line graphs)")
            #     plt.savefig("ER_"+datasetName+"(line graphs).png")
            # else:
            #     plt.title("Distribution of Effective Resistance "+datasetName+" (graphs)")
            #     plt.savefig("ER_"+datasetName+"(line graphs).png")

            print("Effective resistance for "+ datasetName + ": Done!!")
            
            # if cfg.gnn.linegraph:
            #     print("Effective resistance for line graphs in " + datasetName + ": Done!!")
            # else:
            #     print("Effective resistance for graphs in " + datasetName + ": Done!!")
        