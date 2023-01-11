# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
import pickle as pkl
from numbers import Number

import numpy as np
import torch


from apis import single_gpu_test
from datasets import build_dataloader, build_dataset
from models import build_model
from utils import (auto_select_device, get_root_logger,
                         setup_multi_processes, wrap_distributed_model,
                         wrap_non_distributed_model)
from utils.evaluation import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
        f'result file. You can choose some of ({", ".join(out_options)}), '
        'or use "all" to include all above, or use "none" to disable all of '
        'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument('--device', help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    assert args.metrics or args.out, \
        'Please specify at least one of output path and evaluation metrics.'

    return args


def main():
    args = parse_args()

    cfg = config_from_file(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    cfg.gpu_id = args.gpu_id
    cfg.device = cfg.gpu_id

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=False,
        drop_last=False,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_model(cfg.model)
    
    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES

    
    model = wrap_non_distributed_model(
        model, device=cfg.device, device_ids=cfg.gpu_ids)
    if cfg.device == 'ipu':
        from mmcv.device.ipu import cfg2options, ipu_model_wrapper
        opts = cfg2options(cfg.runner.get('options_cfg', {}))
        if fp16_cfg is not None:
            model.half()
        model = ipu_model_wrapper(model, opts, fp16_cfg=fp16_cfg)
        data_loader.init(opts['inference'])
    model.CLASSES = CLASSES
    show_kwargs = args.show_options or {}
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                              **show_kwargs)
    
    results = {}
    logger = get_root_logger()
    if args.metrics:
        eval_results = evaluate(
            results=outputs,
            metric=args.metrics,
            metric_options=args.metric_options,
            logger=logger)
        results.update(eval_results)
        for k, v in eval_results.items():
            if isinstance(v, np.ndarray):
                v = [round(out, 2) for out in v.tolist()]
            elif isinstance(v, Number):
                v = round(v, 2)
            else:
                raise ValueError(f'Unsupport metric type: {type(v)}')
            print(f'\n{k} : {v}')
    if args.out:
        if 'none' not in args.out_items:
            scores = np.vstack(outputs)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [CLASSES[lb] for lb in pred_label]
            res_items = {
                'class_scores': scores,
                'pred_score': pred_score,
                'pred_label': pred_label,
                'pred_class': pred_class
            }
            if 'all' in args.out_items:
                results.update(res_items)
            else:
                for key in args.out_items:
                    results[key] = res_items[key]
        print(f'\ndumping results to {args.out}')
        pkl.dump(results, args.out)


if __name__ == '__main__':
    main()
