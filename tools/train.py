# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import yaml
import shutil

import torch
import torch.distributed as dist

from utils.parallel import get_dist_info, init_dist, setup_multi_processes


from apis import init_random_seed, set_random_seed, train_model
from utils import auto_select_device, get_root_logger
from configs import config_from_file

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from', default='')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = config_from_file(args.config)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = './exprs/tmp'
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    cfg.gpu_ids = args.gpu_ids

    # init distributed env first, since logger depends on the dist info.
    if len(cfg.gpu_ids) < 2:
        distributed = False
    else:
        distributed = True
        init_dist('pytorch', **cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
        

    # create work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)
    # dump config
    # yaml.safe_dump(cfg, osp.join(cfg.work_dir, osp.basename(args.config)))
    # shutil.copyfile(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg}')

    # set random seeds
    cfg.device = f'cuda:{dist.get_rank()}'
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    # add an attribute for visualization convenience
    train_model(
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        device=cfg.device,
        meta=meta)


if __name__ == '__main__':
    main()
