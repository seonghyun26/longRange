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

from tqdm import tqdm


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return SchedulerConfig(scheduler=cfg.optim.scheduler,
                           steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
                           max_epoch=cfg.optim.max_epoch)


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

def preprocess(loaders, lgvariant):
    for loaderIdx, loader in enumerate(loaders):
        loaderSplit = ["train", "valid", "test"]
        print("Preprocessing {}".format(loaderSplit[loaderIdx]))
        data = loader.dataset.data
        slices = loader.dataset.slices
        
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        numberOfGraphs = slices['y'].shape[0]-1
        
        x_index_slice = slices['x']
        edge_index_slice = slices['edge_index']
        
        lg_node_index = edge_index
        lg_node_index_slice = edge_index_slice
        lg_edge_idx_list = []
        lg_edge_idx_list_slice = [0]
        
        for graphIdx in tqdm(range(numberOfGraphs)):
            graphNodeIdx = lg_node_index[:, lg_node_index_slice[graphIdx]:lg_node_index_slice[graphIdx+1]]
            # graphEdgeIdx = edge_index[:, edge_index_slice[graphIdx]:edge_index_slice[graphIdx+1]]

            # How to make egees
            if lgvariant ==1 :
                # Non-backtracking edges
                lg_edge_index = torch.nonzero(
                    (graphNodeIdx[:, 1, None] == graphNodeIdx[:, 0]) &
                    (graphNodeIdx[:, 0, None] != graphNodeIdx[:, 1])
                )
            elif lgvariant == 2:
                # Connect ij and jk, regardless of i and k
                lg_edge_index = torch.nonzero(
                    (graphNodeIdx[:, 1, None] == graphNodeIdx[:, 0])
                )
            
            lg_edge_idx_list.append(lg_edge_idx.T)
            lg_edge_idx_list_slice.append(lg_edge_idx_list_slice[-1]+lg_edge_idx.shape[0])
            
        loader.dataset.data.lg_edge_idx = torch.cat(lg_edge_idx_list, dim=1)
        loader.dataset.slices['lg_edge_idx'] = torch.tensor(lg_edge_idx_list_slice, dtype=torch.int)
        
        # lg_node_x = torch.cat(lg_node_x, dim=0)
        # lg_node_index = torch.cat(lg_node_index, dim=1)
        # lg_node_index_slice = torch.tensor(lg_node_index_slice, dtype=torch.int)
        # lg_edge_index = torch.cat(lg_edge_index, dim=1)
        # lg_edge_attr = torch.cat(lg_edge_attr, dim=0)
        # lg_edge_index_slice = torch.tensor(lg_edge_slice, dtype=torch.int)
        # lg_edge_attr_slice = torch.tensor(lg_edge_slice, dtype=torch.int)
        
        #NOTE: to fix
        # loader.dataset.data.org_x = x
        # loader.dataset.slices['org_x'] = x_index_slice
        # loader.dataset.data.org_edge_index = edge_index
        # loader.dataset.slices['org_edge_index'] = edge_index_slice

        # loader.dataset.data.lg_node_index = lg_node_index
        # loader.dataset.data.edge_index = lg_edge_index
        # loader.dataset.slices['edge_index'] = lg_edge_index_slice
        # loader.dataset.data.x = edge_attr
        # loader.dataset.slices['x'] = edge_index_slice
        # loader.dataset.data.edge_attr = lg_edge_attr
        # loader.dataset.slices['edge_attr'] = lg_edge_attr_slice
        
        
        # pt_file[0].x = lg_x
        # pt_file[1]['x'] = lg_x_slice
        # pt_file[0].edge_index = lg_edge_index
        # pt_file[1]['edge_index'] = lg_edge_index_slice
        # pt_file[0].edge_attr = lg_edge_attr
        # pt_file[1]['edge_attr'] = lg_edge_attr_slice
        # pt_file[0].lg_node_idx = lg_node_index
        # pt_file[1]['lg_node_idx'] = lg_node_index_slice
        print("FLAG")
        
    return loaders

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
        
        if cfg.train.finetune:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        loaders = create_loader()
            
        loggers = create_logger()
        model = create_model()
        
        if cfg.gnn.lgvariant == -1:
            loaders = preprocess(loaders, cfg.gnn.lgvariant)
        if cfg.train.finetune:
            model = init_model_from_pretrained(model, cfg.train.finetune,
                                               cfg.train.freeze_pretrained)
        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: {}'.format(cfg.params))
        # Start training
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            train(loggers, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, '{}_done'.format(args.cfg_file))
    logging.info(f"[*] All done: {datetime.datetime.now()}")
