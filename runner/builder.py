import torch

from .optimzer import SGD, Adam, AdamW
from .runner import Runner

def build_optimizer(model, cfg):
    """Build an optimizer accroding to the cfg.
    Args:
        model (nn.Module): The model to be optimized.
        cfg (dict): A dictionary contains args for building an optimizer.
    Returns:
        optim (torch.optim.Optimizer)
    """
    pass


def build_runner(cfg):
    """Build an optimizer accroding to the cfg.
    Args:
        cfg (dict): A dictionary contains args for building an optimizer.
    Returns:
        runner (runner.Runner)
    """
    pass
