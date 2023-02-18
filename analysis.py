import os
import copy
import random
import torch
import numpy as np
import scipy.linalg
import torch.nn.functional as F
from configs import config_from_file
from datasets import build_datasets, build_dataloader
from models import build_models

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')
# from utils.evaluation import fuzziness


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
        shuffle=True,
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

def get_features(is_odenet, feat_type, layer_idxs, t_list, pooled, add_fc, model, sampled_data, device):
    model = model.to(device)
    model.eval()
    whole_labels = []
    
    loss_vals = []
    messages = []
    rand_mats = {}
    cat_feats = {}
    down_channel = 256
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
                if l_i in layer_idxs:
                    tmp_feats = feats[l_i].to(device)
                    C = (torch.flatten(tmp_feats, 1)).size(1)
                    
                    if l_i in rand_mats:
                        rand_mat = rand_mats[l_i]
                    else:
                        rand_mat = torch.randn(C, down_channel)
                        rand_mats[l_i] = rand_mat
                    rand_mat = rand_mat.to(device)

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
    return '\n'.join(messages), combined_feats, combined_labels, mean_loss_val


def fuzziness(feats, labels, num_classes=10):
    """
    Compute fuzziness according to features and corresponding labels.
    Args:
        feats[torch.tensor, N x [*]]: features used to compute fuzziness
        labels[torch.tensor, N]: labels of each sample
    Return:
        fuzziness[float]
    """
    device = labels.device
    
    feats = copy.deepcopy(feats)
    labels = copy.deepcopy(labels)

    feats = feats.reshape(feats.size(0), -1).to(device)  # [N x C]
    C = feats.size(1)
    # down_channel = 1024
    # rand_mat = torch.randn(C, down_channel).to(feats.device)
    # feats = torch.matmul(feats, rand_mat)

    feats = feats.to(device)

    whole_feat_mean = feats.mean(dim=0)  # [C]

    feats = feats - whole_feat_mean.view(1, -1)
    
    # compute mean
    class_idxs = list(range(num_classes))
    feat_means = [[] for _ in class_idxs]  #{0:[C], 1:[C], 2:[C], ...}
    
    for idx, label in enumerate(labels):
        label_ = int(label.item())
        feat_means[label_].append(feats[idx])
    feats = feats.cpu()
    
    signal_b = torch.zeros(C, C).to(device)
    signal_w = torch.zeros(C, C).to(device)
    for k in class_idxs:
        class_feats = torch.stack(feat_means[k])
        feat_mean = class_feats.mean(dim=0)
        
        diff_b = feat_mean #- whole_feat_mean
        # tmp_test = torch.matmul(diff_b.reshape(-1, 1), diff_b.reshape(1, -1))
        
        signal_b = signal_b + \
                      torch.matmul(diff_b.reshape(-1, 1), diff_b.reshape(1, -1)) * len(feat_means[k])
        
        diff_w =  class_feats - feat_mean.view(1, -1)
        
        signal_w = signal_w + torch.matmul(torch.transpose(diff_w, 0, 1), diff_w)

    signal_b = signal_b / float(feats.size(0))
    signal_w = (signal_w / float(feats.size(0))).cpu()
    inv_signal_b = torch.linalg.pinv(signal_b, rcond=1e-5)
    signal_b = signal_b.cpu()

    D_seperation = torch.trace(torch.matmul(signal_w.to(device), inv_signal_b))
    
    return D_seperation, signal_b.cpu(), signal_w.cpu()

# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(93)

# Hyper parameters``
device = torch.device('cuda:6')
feat_type = ['relu']
batch_size = 8
layer_idxs = list(range(0,100))
pooled = False
add_fc = True

# "cifar10_32_1000_swin_t cifar10_32_1000_vgg13_bn cifar10_32_1000_vit_b_16 cifar10_64_100_resnet50_bn cifar10_128_100_resnet50_bn cifar10_128_100_hybrid_v1"
# cifar10_32_100_vgg13_bn cifar10_64_100_vgg13_bn cifar10_128_100_vgg13_bn cifar10_224_100_vgg13_bn
# model_names = "cifar10_32_100_resnet50_bn cifar10_32_100_swin_t cifar10_32_100_vit_b_16 " + \
#                 "cifar10_64_100_resnet50_bn cifar10_64_100_swin_t cifar10_64_100_vit_b_16 cifar10_64_100_hybrid_v1 cifar10_64_100_hybrid_v2 " + \
#                 "cifar10_128_100_resnet50_bn cifar10_128_100_swin_t cifar10_128_100_vit_b_16 cifar10_128_100_hybrid_v1 cifar10_128_100_hybrid_v2 " + \
#                 "cifar10_256_100_swin_t cifar10_256_100_vit_b_16 cifar10_256_100_hybrid_v4 cifar10_256_100_hybrid_v2"
# cifar10_256_1000_hybrid_v2 cifar10_256_1000_hybrid_v4
# model_names = "cifar10_256_1000_swin_t " + \
#               "cifar10_256_1000_vit_b_16 cifar10_224_1000_resnet50_bn cifar10_224_1000_vgg13_bn"
# model_names = "cifar10_224_1000_resnet50_bn cifar10_224_1000_vgg13_bn"
# model_names = "cifar100_224_100_resnet50_bn cifar100_224_100_vgg13_bn"
# model_names = "cifar100_256_100_swin_t cifar100_256_100_vit_b_16"
# model_names = "cifar100_224_100_resnet50_bn cifar100_224_100_vgg13_bn cifar100_256_100_hybrid_v2 cifar100_256_100_swin_t cifar100_256_100_vit_b_16"
# model_names = "cifar100_32_100_hybrid_v1 cifar100_32_100_hybrid_v2 cifar100_32_100_resnet50_bn cifar100_32_100_swin_t cifar100_32_100_vit_b_16 cifar100_32_100_vgg13_bn"
model_names = "cifar10_32_100_odenet_midpoint cifar10_32_100_odenet_rk"
for model_name in model_names.split(' '):
    # model_name = 'cifar10_64_100_hybridode_v1'
    config_path = f'./exprs/{model_name}/config.yaml'
    cfg = config_from_file(config_path)
    resized_size = cfg.data.train.resized_size[0]
    max_class_num = 100
    cfg.data.train.root[0] = cfg.data.train.root[0].replace('home/lhy', 'sdc1/hylin')
    print(cfg.data.train)
    if hasattr(cfg.data.train, 'max_class_num'):
        max_class_num = cfg.data.train.max_class_num
    if not hasattr(cfg.model, 'image_size'):
        cfg.model['image_size'] = 32
    sample_size = cfg.model.num_classes * max_class_num
    num_classes = cfg.model.num_classes
    print('Sample Size:', sample_size)

    if sample_size <= 1000:
        plot_settings = [   
                        [(2500,), (3000,)],
                        [(3500,), (4000,)],
                    ]
    elif sample_size <= 10000:
        plot_settings = [
                        [(25000,), (30000,)],
                        [(35000,), (40000,)],
                    ]
    feat_types = [['block', 'down', 'rand']]

    is_odenet = 'ode' in model_name
    if 'ode' in model_name:
        num_ts = [1,2,3,4,5,6,7,8,9,10]
    else:
        num_ts = [1]

    ori_feat_types = feat_types

    pooled_tag = '_pool' if pooled else ''
    pooled_tag = pooled_tag + '_fc' if add_fc else pooled_tag
    ep_name = '20230130'
    seeds = ['97', '177', '197']
    # seeds = ['7', '99', '97'] #, '103', '177', '197',]
    # seeds = ['7', '99', '97', '103', '177', '197',
    #             '223', '337', '463', '759', 
    #             '777', '919']

    # seeds = ['97', '103', '177', '197',
    #             '223', '337', '463', '759', 
    #             '777', '919']
    for num_t in num_ts:
        t_list = list(np.linspace(0, 1, num_t))
        if 'ode' in model_name:
            for feat_type in feat_types:
                feat_type.append(f'-{len(t_list)}')
            # feat_types = [ori_feat_type.append(f'-{len(t_list)}') for ori_feat_type in ori_feat_types]
        for seed in seeds:
            for feat_type in feat_types:
                feat_tag = '-'.join(feat_type)
                os.makedirs(f'./logs/{model_name}/{feat_tag}{pooled_tag}', exist_ok=True)
                log_file = open(f'./logs/{model_name}/{feat_tag}{pooled_tag}/{ep_name}_{seed}_{feat_tag}{pooled_tag}.txt', 'w')

                
                # plot_settings = [   
                #     [(1000,), (1500,)],
                #     [(2000,), (2500,)],
                # ]
                lenx, leny = len(plot_settings[0]), len(plot_settings)

                fig, axs  = plt.subplots(leny, lenx, sharex=True, sharey=True)
                ckpt_not_exist = False
                for j in range(leny):
                    for i in range(lenx):
                        iter_num = plot_settings[j][i][0]
                        ckpt_path = f'exprs/{model_name}/{ep_name}_{seed}/chckpoints/iter_{iter_num}.pth'
                        if not os.path.exists(ckpt_path):
                            ckpt_not_exist = True
                            break
                        checkpoint = torch.load(ckpt_path, map_location=device)

                        # print(checkpoint)

                        # The process
                        
                        cfg.data.samples_per_gpu = batch_size
                        cfg.data.workers_per_gpu = 1
                        # Sample Data
                        
                        sampled_data = sample_data(sample_size, resized_size, cfg)
                        

                        # Construct model
                        models = build_models(cfg.model)  # models['cls']
                        models['cls'].load_state_dict(checkpoint['models']['cls'])

                        # Get features
                        message, feats, labels, mean_loss_val = get_features(is_odenet, feat_type, layer_idxs, t_list, pooled, add_fc, models['cls'], sampled_data, device)
                        log_file.write(message + '\n')

                        # Compute fuzziness
                        fuzzs_out = []
                        for layer_i, c_feat in enumerate(feats):

                            print(f'Starting {layer_i} fuzziness.')
                            fuzzs_out.append(fuzziness(c_feat, labels, num_classes=num_classes))

                        fuzzs_val = [out[0] for out in fuzzs_out]
                        fuzz_bs = [out[1] for out in fuzzs_out]
                        fuzz_ws = [out[2] for out in fuzzs_out]

                        fuzzs_val = torch.tensor(fuzzs_val)

                        log_file.write(f'Fuzzs:{fuzzs_val} \n')

                        fuzzs_log_val =  fuzzs_val.abs().log()
                        log_file.write(f'Log Fuzzs :{fuzzs_log_val}\n')

                        axs[j][i].set_title(f'Iter {iter_num} Loss {mean_loss_val:.4f}')
                        
                        axs[j][i].plot(layer_idxs[:len(fuzzs_log_val)], fuzzs_log_val, 'ro')
                        # axs[j, i].plot(layer_idxs[:5], layer_idxs[:5], 'ro')
                        log_file.flush()
                    if ckpt_not_exist:
                        break
                if not ckpt_not_exist:
                    for ax in axs.flat:
                        ax.set(xlabel='depth', ylabel='log D')
                        ax.label_outer()
                    # plt.ylim(-1, 10)
                    # plt.axis('equal')
                    plt.suptitle(f'{model_name}-{ep_name}_{seed}-{feat_tag}')
                    os.makedirs(f'./figures/{model_name}/{feat_tag}{pooled_tag}', exist_ok=True)
                    plt.savefig(f'./figures/{model_name}/{feat_tag}{pooled_tag}/{ep_name}_{seed}_{feat_tag}{pooled_tag}.png')
                # plt.savefig(f'./figures/pool_{model_name}_{ep_name}_{seed}_{feat_type}_l{min(layer_idxs)}-{max(layer_idxs)}_sum.png')
                


    # ori_fuzzs_out = [get_variation(c_feat.view(c_feat.size(0), -1), labels, {'class_num':10}) for c_feat in feats]
    # ori_fuzzs_val = [out[0] for out in ori_fuzzs_out]
    # ori_fuzz_bs = [out[2] for out in ori_fuzzs_out]
    # ori_fuzz_ws = [out[3] for out in ori_fuzzs_out]


    # ori_fuzzs_val = torch.tensor(ori_fuzzs_val)
    # print('Ori Fuzzs :',ori_fuzzs_val)

    # # print('ws', torch.equal(fuzz_ws[0], ori_fuzz_ws[0]), (fuzz_ws[0] - ori_fuzz_ws[0]).abs().sum())

    # # print('bs', torch.equal(fuzz_bs[0], ori_fuzz_bs[0]), (fuzz_bs[0] - ori_fuzz_bs[0]).abs().sum())


    # ori_fuzzs_log_val =  ori_fuzzs_val.abs().log()
    # print('Ori Log Fuzzs :',ori_fuzzs_log_val)


    # # ori_fuzzs_np_val = [get_variation_np(c_feat.view(c_feat.size(0), -1).cpu().numpy(), labels.cpu().numpy(), {'class_num':10})[0] for c_feat in feats]
    # # ori_fuzzs_np_val = torch.tensor(ori_fuzzs_np_val)
    # # print('NP Ori Fuzzs :',ori_fuzzs_np_val)

    # # ori_fuzzs_np_log_val =  ori_fuzzs_np_val.abs().log()
    # # print('NP Ori Log Fuzzs :',ori_fuzzs_np_log_val)




