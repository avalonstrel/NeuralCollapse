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

from datasets.transforms import build_transforms
from datasets.samplers import build_sampler
from .cifar import CIFAR10, CIFAR100


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def build_datasets(cfg, default_args=None):
    """Bulid a dataset from a config for dataset
    Args:
        cfg (dict): A dictionary contains the information to contruct dataset object.
        Typically, there should be a `type` and other `kwargs`.
    """
    assert len(cfg.type) == len(cfg.root), 'The number of dataset types should be equal to roots'
    datasets = {}

    if default_args is None:
        default_args = {}

    for data_type, root, trans_list, resized_size in zip(cfg.type, cfg.root, cfg.transforms, cfg.resized_size):
        trans_kwargs = {
            'RandomResizedCrop':{'size':resized_size},
            'Resize':{'size':resized_size}
        }
        if data_type == 'cifar10':
             dataset = CIFAR10(
                root=root,
                transforms=build_transforms(trans_list, trans_kwargs),
                **default_args)
        if data_type == 'cifar100':
             dataset = CIFAR100(
                root=root,
                transforms=build_transforms(trans_list, trans_kwargs),
                **default_args)
        elif data_type == 'quickdraw':
            dataset = QuickDraw(
                root=root,
                transforms=build_transforms(trans_list, trans_kwargs),
                **default_args)
        elif data_type == 'tuberlin':
            dataset = TUBerlin(
                root=root,
                transforms=build_transforms(trans_list, trans_kwargs),
                **default_args)
        else:
            ValueError(f'Unsupported type {data_type} of dataset.')
        datasets[data_type] = dataset
    return datasets

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
                type='DistributedSampler',),
            dict(dataset=dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=seed))
    else:
        sampler = None
    # print('sampler', sampler_cfg, sampler)
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

    # print(batch_size, samples_per_gpu, num_workers)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        **kwargs)
    
    return data_loader

