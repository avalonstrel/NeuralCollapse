import os
import copy
import random
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from configs import config_from_file
from scipy.stats import pearsonr

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')
# from utils.evaluation import fuzziness
from utils.evaluation.nc_measure import sigma_to_mu
def parse_args():
    parser = argparse.ArgumentParser(description='Analysis a model')
    parser.add_argument('--model_names', help='model names for analysis')
    parser.add_argument('--seeds', help='seeds')
    parser.add_argument('--feat_types', help='feat types like block down noin.')
    parser.add_argument('--metric_type', help='a metric like eculidean signal to noise value, or other metrics.')
    parser.add_argument('--plot_metric_type', help='a metric like eculidean signal to noise value, or other metrics.')
    parser.add_argument('--add_metric_types', help='some additional metrics based on original metrics')
    parser.add_argument('--metric_trans_type', help='some calculation transformation on metrics to find some law')
    parser.add_argument('--expr_prefix', help='the prefix of the experiements.')
    args = parser.parse_args()

    return args

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
    if np.any(np.isnan(y)):
        pearson_val = 0
    else:
        pearson_coeff = pearsonr(x,y)
        pearson_val = pearson_coeff[0]
    plt.figure()
    plt.suptitle(f'{model_name}-{seed}-{feat_tag}{plot_metric_type}')
    plt.title(f'Iter {iter_num} Loss {mean_loss_val:.4f} Pearson {pearson_val:.4f}')
    plt.plot(x, y, 'ro', markersize=7)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'./figures/{expr_prefix}/{model_name}/{feat_tag}/{seed}_{feat_tag}{plot_metric_type}_iter{iter_num}.png')

def post_metric_trans(metric_vals, metric_trans_type):
    post_metric_vals = np.array(metric_vals)
    assert metric_trans_type in ['ori', 'exp', 'log'] or 'exproot' in metric_trans_type or 'root' in metric_trans_type
    if metric_trans_type == 'exp':
        post_metric_vals = np.exp(post_metric_vals)
    elif metric_trans_type == 'log':
        post_metric_vals = np.log(post_metric_vals)
    elif 'exproot' in metric_trans_type:
        root_val = 1 / float(metric_trans_type[7:])
        post_metric_vals = np.exp(post_metric_vals)**(root_val)
    elif 'root' in metric_trans_type:
        root_val = 1 / float(metric_trans_type[4:])
        post_metric_vals = post_metric_vals**(root_val)
    return post_metric_vals

def new_metric_trans(metric_vals_dict, augment_data_dict, add_metric_types):
    for add_metric_type in add_metric_types:
        print(add_metric_type)
        if 'vtm_d2' in add_metric_type:
            # augment_data_dict # list of layers[]
            print(len(augment_data_dict[0]['proj_dist']))
            add_metric_values = []
            for layer_i in range(len(augment_data_dict)):
                layer_augment_data = augment_data_dict[layer_i]['proj_dist']
                projected_feats, projected_labels, projected_feats_means = layer_augment_data

                def dist_func(x, y):
                    return (torch.linalg.matrix_norm(x-y, 'fro')).detach().cpu().item()
                
                dist_w, dist_b = sigma_to_mu(projected_feats, projected_labels, projected_feats_means, dist_func)
                new_value = ((dist_w) / dist_b)
                add_metric_values.append(new_value)
                print(layer_i, new_value)
            metric_vals_dict[add_metric_type] = np.array(add_metric_values)

        elif 'vtm' in add_metric_type:  # variance to mean
            dist_w = np.array(metric_vals_dict['dist_w'])
            dist_b = np.array(metric_vals_dict['dist_mb'])
            metric_vals_dict[add_metric_type] = dist_w**2  / dist_b 
        elif 'v2tm' in add_metric_type:  # variance to mean
            dist_w = np.array(metric_vals_dict['dist_w2'])
            dist_b = np.array(metric_vals_dict['dist_mb'])
            metric_vals_dict[add_metric_type] = dist_w  / dist_b 
        
    return metric_vals_dict

if __name__ == '__main__':

    set_seed(91)
    args = parse_args()
    seeds = args.seeds.split()
    # Hyper parameters
    
    feat_type = ['relu']
    batch_size = 16
    plot_layer_idxs = list(range(2,100))  # some filtration rules
    
    expr_prefix = args.expr_prefix
    model_names = args.model_names
    metric_type = args.metric_type
    # print(args)
    plot_metric_type = args.plot_metric_type
    add_metric_types = args.add_metric_types.split()
    for model_name in model_names.split(' '):
        # loading training parameters from config file
        config_path = f'./exprs/{expr_prefix}/{model_name}/config.yaml'
        sample_size = 1000
        plot_settings = get_plot_settings(sample_size)

        feat_type = args.feat_types.split('-')

        for seed in seeds:
            feat_tag = '-'.join(feat_type)
            os.makedirs(f'./figures/{expr_prefix}/{model_name}/{feat_tag}', exist_ok=True)
            record_file = f'./logs/{model_name}/{feat_tag}/{seed}_{feat_tag}.pth'
            
            lenx, leny = len(plot_settings[0]), len(plot_settings)
            ckpt_not_exist = False
            records = {}

            # if there exist some record just load from the record but not commpute
            print(f'loading from {record_file}')
            if os.path.exists(record_file):
                records = torch.load(record_file)
                # if records is newest load otherwise update it.
            else:
                print('Record not exist, skip.')
                continue
            for j in range(leny):
                for i in range(lenx):
                    iter_num = plot_settings[j][i][0]
                    if j * leny + i < len(records):
                        print(f'iter_num:{iter_num} finished, just plot.')
                        record = records[iter_num]
                        mean_loss_val = record['loss_val']
                        metric_vals_dict, augment_data_dict = {}, {}
                        if 'metric_vals' in record:
                            metric_vals_dict = record['metric_vals']
                        elif 'metric_val' in record:
                            metric_vals = record['metric_val']
                            metric_vals_dict[args.metric_type] = metric_vals
                        elif 'fuzz_log' in record:
                            metric_vals = record['fuzz_log']
                            metric_vals_dict[args.metric_type] = metric_vals
                        if 'augment_datas' in record:
                            augment_data_dict = record['augment_datas']
                        metric_vals_dict = new_metric_trans(metric_vals_dict, augment_data_dict, add_metric_types)
                        for metric_type_ in metric_vals_dict:
                            print(plot_metric_type, metric_type_)
                            if plot_metric_type == metric_type_:
                                metric_vals = metric_vals_dict[metric_type_]
                                layers, filtrated_metric_vals = filtrate(metric_vals, plot_layer_idxs)
                                filtrated_metric_vals = post_metric_trans(filtrated_metric_vals, args.metric_trans_type)
                                if len(metric_vals_dict) == 1:
                                    plot_metric_tag = ''
                                else:
                                    plot_metric_tag = '_' + metric_type_
                                plot_and_save(layers, filtrated_metric_vals, 'depth', metric_type_ + '-' + args.metric_trans_type, \
                                            expr_prefix, model_name, seed, feat_tag, plot_metric_tag + f'_{args.metric_trans_type}', iter_num, mean_loss_val)
                        
                    