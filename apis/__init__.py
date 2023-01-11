# The same structure with the open-mmlab/mmclassification (https://github.com/open-mmlab/mmclassification)

# from .inference import inference_model, init_model
from .test import single_gpu_test
from .train import init_random_seed, set_random_seed, train_model

__all__ = [
    'set_random_seed', 'train_model', 
    'single_gpu_test',
    'init_random_seed'
]