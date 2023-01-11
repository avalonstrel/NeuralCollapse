# Inheritaed from open-mmlab/mmclassification
# Modified by Hangyu LIN

import copy
import platform
import random
from functools import partial

import numpy as np
import torch

from utils.parallel import get_dist_info
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler 


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def build_from_cfg(cfg):
    """Bulid a dataset from config for dataset
    Args:
        cfg (dict): A dictionary contains the information to contruct dataset object.
        Typically, there should be a `type` and other `kwargs`.
    """
    raise NotImplementedError("Need to be implemented accroding to different types of datasets.")

def build_sampler(cfg, default_args=None):
    if cfg is None:
        return None
    elif cfg['type'] == 'DistributedSampler':
        return DistributedSampler(default_args)
        

def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg,  default_args)
    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     drop_last=False,
                     seed=None,
                     pin_memory=True,
                     persistent_workers=True,
                     sampler_cfg=None,
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        drop_last (bool): Whether to drop extra samples to make it evenly divisible. 
            Default: False.
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0.
            Default: True
        sampler_cfg (dict): sampler configuration to override the default
            sampler
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()

    # Custom sampler logic
    if sampler_cfg:
        # shuffle=False when val and test
        sampler_cfg.update(shuffle=shuffle)
        # TODO: Not implemented now.
        sampler = build_sampler(
            sampler_cfg,
            default_args=dict(
                dataset=dataset, num_replicas=world_size, rank=rank,
                seed=seed))
    # Default sampler logic
    elif dist:
        sampler = build_sampler(
            dict(
                type='DistributedSampler',)
            dict(dataset=dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=seed))
    else:
        sampler = None

    # If sampler exists, turn off dataloader shuffle
    if sampler is not None:
        shuffle = False

    if dist:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader

