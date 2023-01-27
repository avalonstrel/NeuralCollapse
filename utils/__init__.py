# Inheritaed from open-mmlab/mmclassification
# Modified by Hangyu LIN

from .misc import (check_prerequisites, concat_list, deprecated_api_warning,
                   has_method, import_modules_from_strings, is_list_of,
                   is_method_overridden, is_seq_of, is_str, is_tuple_of,
                   iter_cast, list_cast, requires_executable, requires_package,
                   slice_list, to_1tuple, to_2tuple, to_3tuple, to_4tuple,
                   to_ntuple, tuple_cast)
from .path import (check_file_exist, fopen, is_filepath, mkdir_or_exist,
                   scandir, symlink)
from .progressbar import (ProgressBar, track_iter_progress,
                          track_parallel_progress, track_progress)

from .timer import Timer, TimerError, check_time

from .logging import get_logger, print_log
from .logger import get_root_logger

from .registry import Registry, build_from_cfg
from .seed import worker_init_fn
# import parallel
# try:
#     import torch
# except ImportError:
__all__ = [
    # 'parallel',
    'get_root_logger', 
    'Config', 'ConfigDict', 'DictAction', 'is_str', 'iter_cast',
    'list_cast', 'tuple_cast', 'is_seq_of', 'is_list_of', 'is_tuple_of',
    'slice_list', 'concat_list', 'check_prerequisites', 'requires_package',
    'requires_executable', 'is_filepath', 'fopen', 'check_file_exist',
    'mkdir_or_exist', 'symlink', 'scandir', 'ProgressBar',
    'track_progress', 'track_iter_progress', 'track_parallel_progress',
    'Timer', 'TimerError', 'check_time', 'deprecated_api_warning',
    'to_1tuple', 'to_2tuple', 'to_3tuple', 'to_4tuple', 'to_ntuple',
    'import_modules_from_strings', 'get_logger', 'print_log',
    'Registry', 'build_from_cfg', 'worker_init_fn'
]
