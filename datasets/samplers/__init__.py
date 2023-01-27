
from torch.utils.data.distributed import DistributedSampler


def build_sampler(cfg, default_args=None):
    if cfg is None:
        return None
    elif cfg['type'] == 'DistributedSampler':
        return DistributedSampler(**default_args)

__all__ = ('DistributedSampler')
