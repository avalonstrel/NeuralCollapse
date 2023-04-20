import os
import copy
import random
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from configs import config_from_file
from datasets import build_datasets, build_dataloader
from models import build_models
from scipy.stats import pearsonr

from utils.evaluation.nc_measure import (general_signal_to_noise, signal_to_noise, minimal_margin,  projected_distance, 
                                         separation_loss, separation_stn, separation_stn_loss, norm_distance)

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')
# from utils.evaluation import fuzziness

def parse_args():
    parser = argparse.ArgumentParser(description='Analysis a model')
    parser.add_argument('--model_names', help='model names for analysis')
    parser.add_argument('--gpu_id', help='gpu_id')
    parser.add_argument('--seeds', help='seeds')
    parser.add_argument('--C_intermediate', type=int, default=64, help='the channel used in sepereation loss ')
    parser.add_argument('--epoch_num', type=int, default=200, help='the epoch num used in sepereation loss ')
    parser.add_argument('--feat_types', help='feat types like block down noin.')
    parser.add_argument('--optim_method', help='optim method used in separation.')
    parser.add_argument('--norm_optim_method', default='pytorch', help='optim method used in separation.')
    parser.add_argument('--net_type', default='fc', help='net type used in separation.')
    parser.add_argument('--norm_type', default=2, help='norm type used in projection distance.')
    parser.add_argument('--dist_norm_type', default='fro', help='norm type used in norm_distance.')
    parser.add_argument('--dist_type', default='mean', help='.')
    parser.add_argument('--lr', type=float, default=0.1, help='lr used in optimization in separation.')
    parser.add_argument('--metric_types', help='a metric like eculidean signal to noise value, or other metrics.')
    parser.add_argument('--expr_prefix', help='the prefix of the experiements.')
    parser.add_argument('--data-dir', help='the dir to the datasets, just for different servers.')
    parser.add_argument('--gstn_dim', default=2, type=int, help='the dim for gstn')
    args = parser.parse_args()

    return args

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
        shuffle=False,
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
    batch_num = sample_size // train_data_loader.batch_size + 1
    sampled_data = []
    for i, data_batch_ in enumerate(train_data_loader):
        if i >= batch_num:
            break
        sampled_data.append(data_batch_)
    return sampled_data

def pre_feat_trans(feats):
    """
    Need to implement for different structure.
    """
    return feats

def get_features_from_data(is_odenet, is_rand_proj, feat_type, t_list, model, sampled_data, device):
    model = model.to(device)
    model.eval()
    whole_labels = []
    
    loss_vals = []
    messages = []
    rand_mats, cat_feats, feat_sizes = {}, {}, {}
    down_channel = 2048

    with torch.no_grad():
        for i, data_batch in enumerate(sampled_data):
            in_datas, labels = [dterm.to(device) for dterm in data_batch]
            kwargs = {'feat_type':feat_type}
            if is_odenet:
                kwargs['t_list'] = t_list

            feats = model.get_features(in_datas, **kwargs)
            # check loss value
            preds = model(in_datas)
            loss_val = F.cross_entropy(preds, labels)
            loss_vals.append(loss_val.detach().cpu().item())
            whole_labels.append(labels)
            layer_num = len(feats)
            # feats
            for l_i in range(layer_num):
                if i == 0:
                    print(f'Feature Shape Layer {l_i}', feats[l_i].size())
                    messages.append(f'Feature Shape Layer {l_i}: {feats[l_i].size()}')
                
                tmp_feats = feats[l_i].to(device)
                C = (torch.flatten(tmp_feats, 1)).size(1)
                
                if l_i in rand_mats:
                    rand_mat = rand_mats[l_i]
                else:
                    rand_mat = torch.randn(C, down_channel)
                    rand_mats[l_i] = rand_mat
                rand_mat = rand_mat.to(device)

                # recording the feature shape
                feat_sizes[l_i] = tmp_feats.size()[1:]
                C_num = torch.flatten(tmp_feats, 1).size(1)
                # Random Reduction
                if is_rand_proj and C_num > down_channel:
                    tmp_feats = torch.flatten(tmp_feats, 1)
                    tmp_feats = torch.matmul(tmp_feats, rand_mat)
                if l_i in cat_feats:
                    cat_feats[l_i].append(tmp_feats.cpu())
                else:
                    cat_feats[l_i] = [tmp_feats.cpu()]

        combined_feats = []
        for l_i in range(layer_num):
            combined_feats.append(torch.cat(cat_feats[l_i], 0))
        print(f'Whole Num of Samples:{combined_feats[0].size(0)}')

        mean_loss_val = sum(loss_vals) / len(loss_vals)
        print('Mean Loss Vals:', mean_loss_val)
        messages.append(f'Mean Loss Vals:{mean_loss_val}')

        combined_labels = torch.cat(whole_labels, dim=0)
    return '\n'.join(messages), combined_feats, feat_sizes, combined_labels, mean_loss_val

# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_plot_settings(sample_size):
    if sample_size <= 1000:
        plot_settings = [   
                        [(2500,), (3000,)],
                        [(3500,), (4000,)],
                    ]
        if 'simclr' in model_name:
            plot_settings = [   
                        [(500,), (1000,)],
                        [(1500,), (1500,)],
                    ]
    elif sample_size <= 10000:
        plot_settings = [
                        [(25000,), (30000,)],
                        [(35000,), (40000,)],
                    ]

    elif sample_size <= 60000:
        plot_settings = [
                        [(150000,), (180000,)],
                        [(210000,), (230000,)],
                    ]
        if 'simclr' in model_name:
            plot_settings = [   
                        [(100000,), (105000,)],
                        [(110000,), (115000,)],
                    ]
    elif sample_size <= 100000:
        plot_settings = [
                        [(100000,), (150000,)],
                        [(200000,), (250000,)],
                    ]
    return plot_settings

def filtrate(metric_vals, layer_idxs):
    layers, filtrated_metric_vals = [], []
    for i, metric_val in enumerate(metric_vals):
        if i in layer_idxs:
            layers.append(i)
            filtrated_metric_vals.append(metric_val)
    return layers, filtrated_metric_vals

def plot_and_save(x, y, x_label, y_label, expr_prefix, model_name, seed, feat_tag, plot_metric_type, iter_num, mean_loss_val):
    pearson_val = pearsonr(x,y)
    plt.figure()
    plt.suptitle(f'{model_name}-{seed}-{feat_tag}{plot_metric_type}')
    plt.title(f'Iter {iter_num} Loss {mean_loss_val:.4f} Pearson {pearson_val[0]:.4f}')
    plt.plot(x, y, 'ro', markersize=7)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'./figures/{expr_prefix}/{model_name}/{feat_tag}/{seed}_{feat_tag}{plot_metric_type}_iter{iter_num}.png')

def get_metric_vals(layer_i, metric_types, feats, labels, num_classes, args):
    metric_vals, real_metric_types = [], []
    augment_datas = {}
    for metric_type in metric_types:
        if layer_i == 0:
            if metric_type == 'sep_stn_loss':
                metric_vals.extend([0, 0])
                real_metric_types.extend(['sep_stn', 'sep_loss'])
            elif metric_type in ['proj_dist', 'norm_dist']:
                metric_vals.extend([0, 0, 0, 0, 0])
                real_metric_types.extend(['dist_w', 'dist_b', 'dist_mb', metric_type, metric_type+'_mb'])
            else:
                metric_vals.append(0)
                real_metric_types.append(metric_type)
            continue
        assert metric_type in ['stn', 'gstn', 'margin', 'sep_loss', 'sep_stn', 'sep_stn_loss', 'proj_dist', 'norm_dist'], 'Metric not supported'
        if metric_type == 'stn':
            out = signal_to_noise(feats, labels, num_classes)
            d_sep_val = out[0]
            metric_val = d_sep_val.abs().log().cpu().item()
        elif metric_type == 'gstn':
            kernel_size, stride, padding = args.after_settings
            out = general_signal_to_noise(feats, labels, num_classes, int(args.gstn_dim), kernel_size, stride, padding)
            d_sep_val = out[0]
            metric_val = d_sep_val.abs().log().cpu().item()
        elif metric_type == 'margin':
            out = minimal_margin(feats, labels, num_classes)
            metric_val = out.cpu().item()
        elif metric_type == 'sep_loss':
            metric_val = separation_loss(feats, labels, num_classes, int(args.C_intermediate), args.after_settings, 
                                        net_type=args.net_type, optim_method=args.optim_method, 
                                        lr=float(args.lr), batch_num=1, epoch_num=int(args.epoch_num), device=args.device)
        elif metric_type == 'sep_stn':
            metric_val = separation_stn(feats, labels, num_classes, int(args.C_intermediate), args.after_settings, 
                                        net_type=args.net_type, optim_method=args.optim_method, 
                                        lr=float(args.lr), batch_num=1, epoch_num=int(args.epoch_num), device=args.device)
        elif metric_type == 'sep_stn_loss':
            stn_metric_val, loss_metric_val = separation_stn_loss(feats, labels, num_classes, int(args.C_intermediate), args.after_settings, 
                                        net_type=args.net_type, optim_method=args.optim_method, 
                                        lr=float(args.lr), batch_num=1, epoch_num=int(args.epoch_num), device=args.device)
        elif metric_type == 'proj_dist':
            proj_norm_type = int(args.norm_type) if args.norm_type != 'fro' else args.norm_type
            dist_norm_type = int(args.dist_norm_type) if args.dist_norm_type != 'fro' else args.dist_norm_type
            kernel_size, stride, padding = args.after_settings
            dist_within, dist_between, mean_dist_between, augment_data = projected_distance(feats, labels, num_classes, kernel_size, stride, padding, 
                                            method=args.norm_optim_method, proj_norm_type=proj_norm_type, dist_norm_type=dist_norm_type, dist_type=args.dist_type,
                                            optim_method=args.optim_method, epoch_num=int(args.epoch_num), lr=float(args.lr), device=args.device)
            augment_datas[metric_type] = augment_data
        elif metric_type == 'norm_dist':
            norm_type = int(args.norm_type) if not args.norm_type in ['fro', 'orifro'] else args.norm_type
            kernel_size, stride, padding = args.after_settings
            dist_within, dist_between, mean_dist_between = norm_distance(feats, labels, num_classes, kernel_size, stride, padding,
                            norm_type=norm_type, dist_type=args.dist_type, device=args.device)
            
        if metric_type == 'sep_stn_loss':
            metric_vals.extend([stn_metric_val, loss_metric_val])
            real_metric_types.extend(['sep_stn', 'sep_loss'])
        elif metric_type in ['proj_dist', 'norm_dist']:
            metric_vals.extend([dist_within, dist_between, mean_dist_between, dist_within / dist_between, dist_within / mean_dist_between])
            real_metric_types.extend(['dist_w', 'dist_b', 'dist_mb', metric_type, metric_type+'_mb'])
        elif metric_type in ['sstn', 'gsstn']:
            signal_b, signal_w = out[1:]
            metric_vals.extend([metric_val, torch.trace(signal_b), torch.trace(signal_w)])
            real_metric_types.extend([metric_type, 'trace_b', 'trace_w'])
        else:
            metric_vals.append(metric_val)
            real_metric_types.append(metric_type)

        
    return metric_vals, real_metric_types, augment_datas

def get_features(cfg, checkpoint):
    # loading training parameters from config file
        
    resized_size = cfg.data.train.resized_size[0]
    max_class_num = 100
    cfg.data.train.root = [os.path.join(args.data_dir, os.path.basename(root)) for root in cfg.data.train.root]
    # cfg.data.train.root[0] = cfg.data.train.root[0].replace('home/lhy', 'sdc1/hylin')
    print(cfg.data.train)
    if hasattr(cfg.data.train, 'max_class_num'):
        max_class_num = cfg.data.train.max_class_num
    if 'cifar100' in model_name:
        max_class_num = min(max_class_num, 600)
    if not hasattr(cfg.model, 'image_size'):
        cfg.model['image_size'] = 32
    sample_size = cfg.model.num_classes * max_class_num
    num_classes = cfg.model.num_classes
    is_odenet = 'ode' in model_name
    is_rand_proj = 'rand' in feat_type
    print(f'is random projection: {is_rand_proj}, {feat_type}')
    print('Sample Size:', sample_size)
    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = 4

    # Sample Data
    sampled_data = sample_data(sample_size, resized_size, cfg)
    # Construct model
    models = build_models(cfg.model)
    key = 'cls'
    if cfg.runner.type == 'NCSSL':
        key = 'ssl'
    
    models[key].load_state_dict(checkpoint['models'][key])
    
    # Get features
    message, feats, feat_sizes, labels, mean_loss_val = get_features_from_data(is_odenet, is_rand_proj, feat_type, t_list, models[key], sampled_data, device)
    return message, feats, feat_sizes, labels, num_classes, mean_loss_val
    


def plot_from_records(records, iter_num, plot_layer_idxs):
    record = records[iter_num]
    metric_vals_dict, mean_loss_val = record['metric_vals'], record['loss_val']
    for metric_type_ in metric_vals_dict:
        metric_vals = metric_vals_dict[metric_type_]
        layers, filtrated_metric_vals = filtrate(metric_vals, plot_layer_idxs)
        if len(metric_vals_dict) == 1:
            plot_metric_type = ''
        else:
            plot_metric_type = '_' + metric_type_
        plot_and_save(layers, filtrated_metric_vals, 'depth', metric_type_, \
                    expr_prefix, model_name, seed, feat_tag, plot_metric_type, iter_num, mean_loss_val)
    plot_and_save(layers, filtrated_metric_vals, 'depth', 'metric vals', \
                    expr_prefix, model_name, seed, feat_tag, '', iter_num, mean_loss_val)

def update_metric_dicts(metric_vals, real_metric_types, augment_datas, metric_vals_dict, augment_datas_dict):
    for metric_val, real_metric_type in zip(metric_vals, real_metric_types):
        if real_metric_type in metric_vals_dict:
            metric_vals_dict[real_metric_type].append(metric_val)
        else:
            metric_vals_dict[real_metric_type] = [metric_val]
        # add augment datas
        if real_metric_type in augment_datas:
            if real_metric_type in augment_datas_dict:
                augment_datas_dict[real_metric_type].append(augment_datas[real_metric_type])
            else:
                augment_datas_dict[real_metric_type] = [augment_datas[real_metric_type]]

def save_records(metric_vals_dict, augment_datas_dict, records):
    for metric_type in metric_vals_dict:
        metric_vals = torch.tensor(metric_vals_dict[metric_type])
        print(f'metric:{metric_vals} \n')

        records[iter_num]['loss_val'] = mean_loss_val
        records[iter_num]['metric_vals'][metric_type] = metric_vals
        if metric_type in augment_datas_dict:
            records[iter_num]['augment_datas'] = augment_datas_dict[metric_type]
        torch.save(records, record_file)
# margin, sep_loss, sep_stn, sep_stn_loss, stn, proj_dist, norm_dist
YLabelDICT = {
    ''
}
if __name__ == '__main__':

    set_seed(91)
    args = parse_args()
    seeds = args.seeds.split()
    # Hyper parameters
    device = torch.device(f'cuda:{args.gpu_id}')
    args.device = device
    feat_type = ['relu']
    batch_size = 16
    plot_layer_idxs = list(range(2,100))  # some filtration rules
    
    expr_prefix = args.expr_prefix
    model_names = args.model_names
    metric_types = args.metric_types.split('|')

    for model_name in model_names.split(' '):
        args.after_settings = None
        if 'k3' in model_name:
            args.after_settings = (3, 1, 1)
        elif 'k5' in model_name:
            args.after_settings = (5, 1, 2)
        elif 'k7' in model_name:
            args.after_settings = (7, 1, 3)
        else:
            args.after_settings = (0, 0, 0)
        print(model_name, args.after_settings)
        
        plot_settings = get_plot_settings(sample_size)

        feat_type = args.feat_types.split('-')
        config_path = f'./exprs/{expr_prefix}/{model_name}/config.yaml'
        cfg = config_from_file(config_path)
        
        if 'ode' in model_name:
            num_ts = [1,2,3,4,5,6,7,8,9,10]
        else:
            num_ts = [1]

        for num_t in num_ts:
            t_list = list(np.linspace(0, 1, num_t))
            if 'ode' in model_name:
                feat_type.append(f'-{len(t_list)}')

            for seed in seeds:
                feat_tag = '-'.join(feat_type)
                os.makedirs(f'./logs/{model_name}/{feat_tag}', exist_ok=True)
                os.makedirs(f'./figures/{expr_prefix}/{model_name}/{feat_tag}', exist_ok=True)
                log_file = open(f'./logs/{model_name}/{feat_tag}/{seed}_{feat_tag}.txt', 'a')
                record_file = f'./logs/{model_name}/{feat_tag}/{seed}_{feat_tag}.pth'
                
                lenx, leny = len(plot_settings[0]), len(plot_settings)
                ckpt_not_exist = False
                records = {}

                # if there exist some record just load from the record but not commpute
                if os.path.exists(record_file):
                    records = torch.load(record_file)
                    # if records is newest load otherwise update it.
                    if isinstance(records[plot_settings[0][0][0]]['metric_vals'], dict):
                        print('record exist', len(records))
                    else:
                        records = {}

                for j in range(leny):
                    for i in range(lenx):
                        iter_num = plot_settings[j][i][0]
                        # if record exists
                        if j * leny + i < len(records):
                            print(f'iter_num:{iter_num} finished, just plot.')
                            plot_from_records(records, iter_num, plot_layer_idxs)
                            continue
                        
                        print(f'iter_num:{iter_num} compute and plot.')
                        # loading ckpt
                        ckpt_path = f'exprs/{expr_prefix}/{model_name}/{seed}/chckpoints/iter_{iter_num}.pth'
                        if not os.path.exists(ckpt_path):
                            ckpt_not_exist = True
                            print(ckpt_path, 'not exist.')
                            break

                        checkpoint = torch.load(ckpt_path, map_location=device)
                        # get features
                        message, feats, feat_sizes, labels, num_classes, mean_loss_val = get_features()
                        log_file.write(message + f'{mean_loss_val} \n')

                        # Compute metrics
                        metric_vals_dict = {}
                        augment_datas_dict = {}
                        records[iter_num] = {'metric_vals':{}}
                        # For each layer features
                        for layer_i, c_feat in enumerate(feats):
                            feat_size = feat_sizes[layer_i]
                            print(f'Starting {layer_i} metric computing.')
                            # get metrics values
                            metric_vals, real_metric_types, augment_datas = get_metric_vals(layer_i, metric_types, c_feat, labels, num_classes, args)
                            log_file.write(f'metric type:{real_metric_types}, metric vals:{metric_vals} \n')
                            print(f'metric type:{real_metric_types}, metric vals:{metric_vals} \n')
                            update_metric_dicts(metric_vals, real_metric_types, augment_datas, metric_vals_dict, augment_datas_dict)
                            # Save for each layer 
                            save_records(metric_vals_dict, augment_datas_dict, records)    
                            log_file.flush()

                        plot_from_records(records, iter_num, plot_layer_idxs)
                    
                # if not ckpt_not_exist:
                #     fig, axs  = plt.subplots(leny, lenx, sharex=True, sharey=True)
                #     for metric_type_ in records[plot_settings[j][i][0]]['metric_vals']:
                #         for j in range(leny):
                #             for i in range(lenx):
                #                 iter_num = plot_settings[j][i][0]
                #                 record = records[iter_num]
                #                 metric_vals, mean_loss_val = record['metric_vals'][metric_type_], record['loss_val']
                #                 axs[j][i].set_title(f'Iter {iter_num} Loss {mean_loss_val:.4f}')
                #                 layers, filtrated_metric_vals = filtrate(metric_vals, plot_layer_idxs)
                #                 axs[j][i].plot(layers, filtrated_metric_vals, 'ro', markersize=7)
                #                 # axs[j][i].set_ylim([min_lim, max_lim])
                #         for ax in axs.flat:
                #             ax.set(xlabel='depth', ylabel=metric_type_)
                #             ax.label_outer()
                        
                #         plt.suptitle(f'{model_name}-{seed}-{feat_tag}')
                #         if len(records[plot_settings[j][i][0]]['metric_vals']) == 1:
                #             plot_metric_type = ''
                #         else:
                #             plot_metric_type = '_' + metric_type_
                #         plt.savefig(f'./figures/{expr_prefix}/{model_name}/{feat_tag}/{seed}_{feat_tag}{plot_metric_type}.png')
                #     plt.savefig(f'./figures/{expr_prefix}/{model_name}/{feat_tag}/{seed}_{feat_tag}.png')
                