import torch

from models.embeddings import ConvEmbedding
from models.backbones import ResnetBackbone
from models.heads import ClsHead
from models.classifiers import VGGClassifier



def build_models(cfg):
    """Build a model from cfg (dict)
    """
    models = {}
    for model_type in cfg.type:
        if 'vgg' in model_type:
            models['cls'] = VGGClassifier(cfg.num_classes, 
                                          dropout=cfg.dropout,
                                          model_type=model_type, 
                                          pretrained=False, 
                                          )
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

