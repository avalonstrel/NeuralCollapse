import torch
import numpy as np
from models.embeddings import ConvEmbedding
from models.backbones import ResnetBackbone
from models.heads import ClsHead
from models.classifiers import (VGGClassifier, ResNetClassifier, InceptionClassifier,
                                SqueezeNetClassifier, DenseNetClassifier, SENetClassifier, SwinClassifier,
                                ODEClassifier, ViTClassifier, HybridNetClassifier, PureHybridNetClassifier, HybridODEClassifier, ResNetSSLModel)



def build_models(cfg):
    """Build a model from cfg (dict)
    """
    models = {}
    for model_type in cfg.type:
        if 'vgg' in model_type:
            no_mean = False
            no_var = False
            if hasattr(cfg, 'no_mean'):
                no_mean = cfg.no_mean
            if hasattr(cfg, 'no_var'):
                no_var = cfg.no_var
            print(cfg)
            models['cls'] = VGGClassifier(cfg.num_classes, 
                                          norm_type=cfg.norm_type,
                                          dropout=cfg.dropout,
                                          image_size=cfg.image_size,
                                          model_type=model_type, 
                                          no_mean=no_mean,
                                          no_var=no_var,
                                          pretrained=False)
        elif 'ssl_resnet' in model_type:
            models['ssl'] = ResNetSSLModel(cfg.out_dim, 
                                          model_type=model_type, 
                                          image_size=cfg.image_size,
                                          norm_type=cfg.norm_type,
                                          pretrained=False)
        elif 'resnet' in model_type:
            models['cls'] = ResNetClassifier(cfg.num_classes, 
                                          model_type=model_type, 
                                          image_size=cfg.image_size,
                                          norm_type=cfg.norm_type,
                                          pretrained=False)
        elif 'inception' in model_type:
            models['cls'] = InceptionClassifier(cfg.num_classes, 
                                                dropout=cfg.dropout, 
                                                image_size=cfg.image_size,
                                                add_settings=cfg.add_settings,
                                                pretrained=False)
        elif 'densenet' in model_type:
            models['cls'] = DenseNetClassifier(cfg.num_classes, 
                                                model_type,)
        elif 'squeezenet' in model_type:
            models['cls'] = SqueezeNetClassifier(version="1_1", 
                                                num_classes=cfg.num_classes, 
                                                dropout=cfg.dropout)
        elif 'senet' in model_type:
            models['cls'] = SENetClassifier(cfg.num_classes, 
                                          model_type=model_type, 
                                          pretrained=False)
        elif 'odenet' in model_type:
            method = None
            if hasattr(cfg, 'method'):
                method = cfg.method
            models['cls'] = ODEClassifier(cfg.num_classes, 
                                          model_type=model_type,
                                          atol=cfg.atol,
                                          rtol=cfg.rtol,
                                          t_list=np.linspace(0, 1, cfg.num_t),
                                          method=method)
        elif 'swin' in model_type:
            models['cls'] = SwinClassifier(cfg.num_classes, 
                                            model_type)
        elif 'vit' in model_type:
            models['cls'] = ViTClassifier(cfg.num_classes, 
                                            model_type,
                                            image_size=cfg.image_size)
        elif 'hybridode' in model_type:
            models['cls'] = HybridODEClassifier(cfg.num_classes, 
                                                cfg.structures, 
                                                image_size=cfg.image_size,
                                                norm_type=cfg.norm_type,
                                                atol=cfg.atol,
                                                rtol=cfg.rtol,)
        elif 'purehybrid' in model_type:
            models['cls'] = PureHybridNetClassifier(cfg.num_classes, 
                                                cfg.structures, 
                                                image_size=cfg.image_size,
                                                norm_type=cfg.norm_type)
        elif 'hybrid' in model_type:
            models['cls'] = HybridNetClassifier(cfg.num_classes, 
                                                cfg.structures, 
                                                image_size=cfg.image_size,
                                                norm_type=cfg.norm_type)
        
        else:
            raise ValueError(f'Unrecognized model type {model_type}')

    return models
    

from models.losses import CrossEntropyLoss, L1Loss, MSELoss

def build_losses(cfg):
    """Build losses used for the model
    """
    losses = {}
    for idx, loss_type in enumerate(cfg.type):
        if loss_type == 'CE':
            loss = CrossEntropyLoss()
        elif loss_type == 'L1':
            loss = L1Loss()
        elif loss_type == 'MSE':
            loss = MSELoss()
        else:
            raise ValueError(f'Unsupported loss type {loss_type}')

        losses['cls'] = loss
    return losses

