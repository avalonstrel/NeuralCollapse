import torch

from .optimizer import SGD, Adam, AdamW
from .nc_runner import NCRunner


OPTIMIZER_DICT = {
    'SGD':SGD,
    'Adam':Adam,
    'AdamW':AdamW
}

RUNNER_DICT = {
    'NC':NCRunner
}
def build_optimizers(models, cfg):
    """Build an optimizer accroding to the cfg.
    Args:
        model (nn.Module): The model to be optimized.
        cfg (dict): A dictionary contains args for building an optimizer.
    Returns:
        optim (torch.optim.Optimizer)
    """
    assert cfg['type'] in OPTIMIZER_DICT, 'Unsupported optimizer {}.'.format(cfg['type'])
    optims = {}
    optim_args = {key:cfg[key] for key in cfg if key not in ['type']}

    for key in models:
        optims[key] = OPTIMIZER_DICT[cfg.type](models[key].parameters(), **optim_args)
    return optims


def build_runner(cfg):
    """Build an optimizer accroding to the cfg.
    Args:
        cfg (dict): A dictionary contains args for building an optimizer.
    Returns:
        runner (runner.Runner)
    """
    assert cfg['type'] in  RUNNER_DICT, 'Unsupported optimizer {}.'.format(cfg['type'])
    runners = {}
    runner_args = {key:cfg[key] for key in cfg if key not in ['type']}
        
    return RUNNER_DICT[cfg['type']](**runner_args)