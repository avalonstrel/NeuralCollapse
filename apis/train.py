# Inheritaed from open-mmlab/mmclassification
# Modified by Hangyu LIN

import random
import warnings

import numpy as np
import torch
import torch.distributed as dist

from models import build_models, build_losses
from datasets import build_dataloader, build_datasets
from runner import build_optimizers, build_runner

from utils import  get_root_logger
from utils.parallel import wrap_distributed_model, wrap_non_distributed_model, auto_select_device


def init_random_seed(seed=None, device=None):
    """Initialize random seed.
    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed
    if device is None:
        device = auto_select_device()
    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(cfg,
                distributed=False,
                validate=True,
                device=None,
                meta=None):
    """Train a model.
    This method will build models, dataloaders, runner for training, wrap the model and build a runner
    according to the provided config.
    Args:
        cfg (:obj:`utils.Config`): The configs of the experiment.
        distributed (bool): Whether to train the model in a distributed
            environment. Defaults to False.
        validate (bool): Whether to do validation with
            :obj:`utils.hooks.EvalHook`. Defaults to False.
        device (str, optional): TODO
        meta (dict, optional): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
    """
    seed = meta['seed']
    set_random_seed(seed, deterministic=True)

    logger = get_root_logger()

    # build model and initialize, dict
    models = build_models(cfg.model)

    

    # build losses
    losses = build_losses(cfg.loss)

    logger.info('Models and losses building finished.')
    logger.info(f'Model Summary: {models}')
    # build datasets
    datasets = build_datasets(cfg.data.train)  # A dict of datasets
    if len(cfg.workflow) == 2:
        # Not Implemented yet, may usable for train+val
        pass
    logger.info('Datasets building finished.')

    # The default loader config
    loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        drop_last=False,
        seed=cfg.get('seed'),
        sampler_cfg=cfg.get('sampler', None),
    )

    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_loader', 'val_loader',
            'test_loader'
        ]
    })

    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_loader', {})}

    # A dict of data loaders
    train_data_loaders = {ds:build_dataloader(datasets[ds], **train_loader_cfg) for ds in datasets}

    logger.info('Dataloader building finished.')
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        for model_name in models:
            if model_name == 'w' or 'm_' in model_name: # consider how to derive the w
                continue
            models[model_name] = wrap_distributed_model(
                models[model_name],
                cfg.device,
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        for model_name in models:
            models[model_name] = wrap_non_distributed_model(
                models[model_name], cfg.device, device_ids=cfg.gpu_ids)


    # build optimizer, a dict
    optims = build_optimizers(models, cfg.optim)

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'NC',
            'max_epochs': cfg.max_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    runner_args = {key:cfg.runner[key] for key in cfg.runner}

    runner_args.update({
            'models':models,
            'losses':losses,
            'optims':optims,
            'logger':logger,
            'work_dir':cfg.work_dir,
            'max_epochs':cfg.max_epochs,
            'meta':meta
        })
    
    runner = build_runner(runner_args)
    
    # register eval hooks
    if validate:
        # build datasets
        val_datasets = build_datasets(cfg.data.val)  # A dict of datasets

        # The specific dataloader settings
        val_loader_cfg = {
            **loader_cfg,
            'shuffle': False,  # Not shuffle by default
            'sampler_cfg': None,  # Not use sampler by default
            'drop_last': False,  # Not drop last by default
            **cfg.data.get('val_dataloader', {}),
        }
        val_data_loaders = {ds:build_dataloader(datasets[ds], **val_loader_cfg) for ds in datasets}
        
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(cfg.data.train.type, 
               train_data_loaders, val_data_loaders, cfg.workflow)