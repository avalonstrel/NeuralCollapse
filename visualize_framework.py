import os
import copy
import random
import torch
import numpy as np
import scipy.linalg
import torch.nn.functional as F
from configs import config_from_file
from datasets import build_datasets, build_dataloader
from models import build_models
from functools import partial
from torch.utils.tensorboard import SummaryWriter

# # Plot
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# # from utils.evaluation import fuzziness


def sample_data(sample_size, resized_size, cfg):
    data_cfg = copy.deepcopy(cfg.data.train)
    data_cfg.update({
            'transforms':[
                  ['Resize',  'ToTensor', 'Normalize'],
                ],
            'resized_size':[resized_size]
        })
    data_names = cfg.data.train.type
    datasets = build_datasets(data_cfg)  # A dict of datasets
    # The default loader config
    loader_cfg = dict(
        num_gpus=1,
        dist=False,
        shuffle=True,
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

    train_data_loader = train_data_loaders[data_names[0]]
    for data in train_data_loader:
        break
    return data

device = torch.device('cuda:3')
model_name = 'cifar10_32_100_vit_b_16'
config_path = f'./exprs/{model_name}/config.yaml'
cfg = config_from_file(config_path)
resized_size = cfg.data.train.resized_size[0]

models = build_models(cfg.model)  # models['cls']

cfg.samples_per_gpu = 2
images, labels = sample_data(64, resized_size, cfg)
model = models['cls'].to(device)
model.forward = partial(model.forward, onnx=True)
# model = partial(model, {'onnx':True})
images = images.to(device)
# pred = model(images)
os.makedirs('logs/tensorboard_logs', exist_ok=True)
writer = SummaryWriter(f'logs/tensorboard_logs/{model_name}')
writer.add_graph(model, images)
writer.close()

