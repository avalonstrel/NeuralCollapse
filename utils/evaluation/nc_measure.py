import copy
import cvxpy as cvx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, LBFGS

def get_dual_weight(feats, labels, num_classes, C_intermediate, after_settings, 
                    net_type='fc', optim_method='adam', 
                    lr=0.1, batch_num=1, epoch_num=500, device=None):
    if device is None:
        device = labels.device

    feat_size = feats.size()
    if len(feat_size) == 4:
        B, C, H, W = feat_size
    else:
        B, C = feat_size
        H, W = 1, 1
    assert B % batch_num == 0, f'batch number should be divided by whole number {B} / {batch_num}'
    batch_size = B // batch_num
    # out_layer = nn.Linear(C, num_classes).to(device)
    kernel_size, stride, padding = after_settings
    print(after_settings)
    # print('C intermediate', C_intermediate)
    if net_type == 'fc':
        out_layer1 = nn.Conv2d(C, C_intermediate, kernel_size, stride=stride, padding=padding).to(device)
        out_layer2 = nn.Linear(C_intermediate * H * W, num_classes).to(device)
        # nn.init.kaiming_normal_(out_layer1.weight)
    elif net_type == 'conv':
        out_layer1 = nn.Conv2d(C, C_intermediate, kernel_size, stride=stride, padding=padding).to(device)
        out_layer2 = nn.Conv2d(C_intermediate, num_classes, (H, W) , stride=(H, W), padding=0).to(device)
        # nn.init.kaiming_normal_(out_layer1.weight)
    elif net_type == 'downfc':
        # down_size
        down_size = H // 8
        out_layer1 = nn.Sequential(*[
                            nn.Conv2d(C, C_intermediate, kernel_size, stride=stride, padding=padding),
                            nn.AvgPool2d(kernel_size=down_size, stride=down_size),
                      ]).to(device)
        out_layer2 = nn.Linear(C_intermediate * H * W // down_size**2, num_classes).to(device)
    elif net_type == 'downstridefc':
        # down_size
        down_size = H // 8
        out_layer1 = nn.Conv2d(C, C_intermediate, kernel_size, stride=stride*down_size, padding=padding).to(device)
        out_layer2 = nn.Linear(C_intermediate * H * W // down_size**2, num_classes).to(device)
    elif net_type == 'downconv':
        # down_size
        down_size = H // 8
        out_layer1 = nn.Sequential(*[
                            nn.Conv2d(C, C_intermediate, kernel_size, stride=stride, padding=padding),
                            nn.AvgPool2d(kernel_size=down_size, stride=down_size),
                      ]).to(device)
        out_layer2 = nn.Conv2d(C_intermediate, num_classes, (H // down_size, W // down_size) , stride=(H // down_size, W // down_size), padding=0).to(device)

    
    def forward(tmp_feats):
        tmp_feats = out_layer1(tmp_feats)
        if 'fc' in net_type:
            tmp_feats = torch.flatten(tmp_feats, 1)
        tmp_preds = out_layer2(tmp_feats)
        if 'conv' in net_type:
            tmp_preds = torch.flatten(tmp_preds, 1)
        return torch.flatten(tmp_feats, 1), tmp_preds

    loss_func = nn.CrossEntropyLoss()
    params = [*(out_layer1.parameters()), *(out_layer2.parameters())]
    if optim_method == 'nmsgd':
        optim = SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    elif optim_method == 'msgd':
        optim = SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    elif optim_method == 'adam':
        optim = Adam(params, lr=lr)
    mean_loss_vals = []
    for epoch_i in range(epoch_num):
        mean_loss_val = 0
        for batch_i in range(batch_num):
            tmp_feats = feats[batch_i*batch_size:(batch_i+1)*batch_size, :].detach().to(device)
            tmp_labels = labels[batch_i*batch_size:(batch_i+1)*batch_size].to(device)
            
            optim.zero_grad()
            tmp_feats, tmp_preds = forward(tmp_feats)
            loss_val = loss_func(tmp_preds, tmp_labels)

            loss_val.backward()
            optim.step()
            
            mean_loss_val = mean_loss_val + loss_val.cpu().item()
        mean_loss_val = mean_loss_val / batch_num
        mean_loss_vals.append(mean_loss_val)

        # print(f'Epoch {epoch_i}: mean loss val: {mean_loss_val}')
    preds = []
    out_feats = []
    with torch.no_grad():
        for batch_i in range(batch_num):
            tmp_feats = feats[batch_i*batch_size:(batch_i+1)*batch_size, :].detach().to(device)
            tmp_feats, tmp_preds = forward(tmp_feats)
            out_feats.append(tmp_feats)
            preds.append(tmp_preds)
    out_feats = torch.cat(out_feats, dim=0)
    preds = torch.cat(preds, dim=0)
    return out_feats, preds, mean_loss_vals

def separation_loss(feats, labels, num_classes, C_intermediate, after_settings, 
                   net_type='fc', optim_method='adam', lr=0.01,
                   batch_num=1, epoch_num=100, repeat_num=3, device=None):
    """
    Compute minimal according to features and corresponding labels.
    Args:
        feats[torch.tensor, N x [*]]: features used to compute margin
        labels[torch.tensor, N]: labels used to compute the margin
    Return:
        minimal margin[float]
    """
    repeat_loss_vals = []
    for r_i in range(repeat_num):
        out_feats, preds, mean_loss_vals = get_dual_weight(feats, labels, num_classes, C_intermediate, after_settings, 
                                                           net_type=net_type, optim_method=optim_method,
                                                           lr=lr, batch_num=batch_num, epoch_num=epoch_num, device=device)
        # print('mean_loss_vals', len(mean_loss_vals),  mean_loss_vals[-5:])
        repeat_loss_vals.append(mean_loss_vals[-1])
    print('repeat loss vals', repeat_loss_vals[-5:])
    # margin_val = 0
    # for i in range(preds.size(0)):
    #     pred, label = preds[i], labels[i]
    #     tmp_val = (pred[label]-pred).sum() / (num_classes - 1)
    #     margin_val += tmp_val
    # margin_val = margin_val / preds.size(0)

    # mininal marginal
    # preds_max_val = preds.max(dim=1, keepdim=True)[0]
    # margin_val = (-(preds - preds_max_val).topk(2, dim=1)[0][:, -1]).min()
    return np.mean(repeat_loss_vals)
    # return np.log(np.mean(repeat_loss_vals))


def separation_stn(feats, labels, num_classes, C_intermediate, after_settings, 
                   net_type='fc', optim_method='adam', lr=0.01,
                   batch_num=1, epoch_num=100, repeat_num=3, device=None):
    """
    Compute minimal according to features and corresponding labels.
    Args:
        feats[torch.tensor, N x [*]]: features used to compute margin
        labels[torch.tensor, N]: labels used to compute the margin
    Return:
        minimal margin[float]
    """
    repeat_stn_vals = []
    for rep_i in range(repeat_num):
        out_feats, preds, mean_loss_vals = get_dual_weight(feats, labels, num_classes, C_intermediate, after_settings, 
                                                           net_type=net_type, optim_method=optim_method, 
                                                           lr=lr, batch_num=batch_num, epoch_num=epoch_num, device=device)
        stn, signal_b, signal_w = signal_to_noise(out_feats, labels, num_classes)
        repeat_stn_vals.append(stn.cpu().item())
    
    return np.log(np.mean(repeat_stn_vals))
    # return np.log(np.mean(repeat_loss_vals))

def separation_stn_loss(feats, labels, num_classes, C_intermediate, after_settings, 
                   net_type='fc', optim_method='adam', lr=0.01,
                   batch_num=1, epoch_num=100, repeat_num=3, device=None):
    """
    Compute minimal according to features and corresponding labels.
    Args:
        feats[torch.tensor, N x [*]]: features used to compute margin
        labels[torch.tensor, N]: labels used to compute the margin
    Return:
        minimal margin[float]
    """
    repeat_stn_vals, repeat_loss_vals = [], []

    for rep_i in range(repeat_num):
        out_feats, preds, mean_loss_vals = get_dual_weight(feats, labels, num_classes, C_intermediate, after_settings, 
                                                           net_type=net_type, optim_method=optim_method, 
                                                           lr=lr, batch_num=batch_num, epoch_num=epoch_num, device=device)
        stn, signal_b, signal_w = signal_to_noise(out_feats, labels, num_classes)
        repeat_stn_vals.append(stn.cpu().item())
        repeat_loss_vals.append(mean_loss_vals[-1])
    
    return np.log(np.mean(repeat_stn_vals)), np.mean(repeat_loss_vals)
    
def reshape_conv_feats(feats, kernel_size, stride, padding):
    """
    Reshape convolution features N x C x H x W into N x (Patch_num) * (kernel_size**2 * C)
    Args:
        feats[torch.tensor]: features of convolution network, N C H W
        kernel_size, stride, padding: the parameters of the last convolution layer
    Return:
        patch_feats[torch.tensor]: features in shape N x (Patch_num) * (kernel_size**2 * C)
    Examples:
        if kernel_size=1, stride=1, padding=0, the it is the same as reshape the conv feats into N x HW x C
    """
    N, C, H, W = feats.size()
    # print('ori feature size', N, C, H, W)
    padded_feats = F.pad(feats, (padding, padding, padding, padding), 'constant', 0)  # N C (H+2d) (W+2d)
    P_H = H + 2 * padding - kernel_size + 1  # patch number
    P_W = W + 2 * padding - kernel_size + 1  # patch number
    patch_feats = []
    for i in range(P_H):
        for j in range(P_W):
            tmp_feats = padded_feats[:, :, i:i+kernel_size, j:j+kernel_size]  # N C 3 3 
            patch_feats.append(tmp_feats.reshape(N, 1, kernel_size**2 * C))
    patch_feats = torch.cat(patch_feats, dim=1)
    return patch_feats

def project_cvx_func(A, Bs, norm_type=2):
    device = A.device
    A = A.cpu().numpy()
    num_classes = len(Bs)
    Bs = [B.cpu().numpy() for B in Bs]
    proj_c = cvx.Variable(num_classes)
    sum_B = sum([proj_c[k] * Bs[k] for k in range(num_classes)])
    objective = cvx.Minimize(cvx.norm(A - sum_B, norm_type))
    prob = cvx.Problem(objective)
    prob.solve()
    projected_A = sum([proj_c.value[k] * Bs[k] for k in range(num_classes)])
    return torch.from_numpy(projected_A).to(device), prob.value

def project_cvx_sdp_func(A, Bs, norm_type=2):
    device = A.device
    # SDP form
    A = A.cpu().numpy()
    num_classes = len(Bs)
    Bs = [B.cpu().numpy() for B in Bs]
    proj_c = cvx.Variable(num_classes)
    t = cvx.Variable(1)
    sum_B = sum([proj_c[k] * Bs[k] for k in range(num_classes)])
    objective = cvx.Minimize(t)
    constraints = [(A - sum_B).T @ (A - sum_B) << t * np.eye(sum_B.shape[1])]
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    projected_A = sum([proj_c.value[k] * Bs[k] for k in range(num_classes)])
    return torch.from_numpy(projected_A).to(device), prob.value

def project_pytorch_func(A, Bs, 
                         norm_type=2, optim_method='adam', epoch_num=100, lr=0.1, device=None):
    if device is None:
        device = A.device
    num_classes = len(Bs)
    def fro_initialization(A, Bs):
        Y = A.reshape(-1, 1)  # {#i * #j, 1}
        X = torch.cat([B.reshape(-1, 1) for B in Bs], dim=1)  # {#i * #j, K} 
        return torch.linalg.lstsq(X, Y).solution.reshape(-1)
    
    # proj_c = torch.randn(num_classes).to(device)
    proj_c = fro_initialization(A, Bs).detach().to(device)
    proj_c.requires_grad=True
    params = [proj_c]
    A = A.to(device)
    Bs = [Bs[k].to(device) for k in range(len(Bs))]
    # print(device)
    def forward(proj_c):
        sum_B = sum([proj_c[k] * Bs[k] for k in range(num_classes)])
        # print(torch.any(torch.isnan(A-sum_B)),torch.isnan(A-sum_B).sum(), torch.isinf(A-sum_B).sum() )
        loss = torch.linalg.matrix_norm(A - sum_B, ord=norm_type)
        return loss
    
    if optim_method in ['adam', 'nmsgd']:
        if optim_method == 'nmsgd':
            optim = SGD(params, lr=lr, momentum=0.9, weight_decay=0, nesterov=True)
        elif optim_method == 'adam':
            optim = Adam(params, lr=lr)
        for epoch_i in range(epoch_num):
            optim.zero_grad()
            loss = forward(proj_c)
            # if epoch_i >= epoch_num - 1:
            #     print(f'Final {epoch_i}', loss)
            loss.backward()
            optim.step()
    elif optim_method == 'lbfgs':
        optim = LBFGS(params, lr=lr, history_size=100, max_iter=40, tolerance_change=1e-5, tolerance_grad=1e-5)
        def closure():
            optim.zero_grad()
            loss = forward(proj_c)
            loss.backward()
            return loss
        for epoch_i in range(epoch_num):
            optim.step(closure)
        # print(f'Final ', forward(proj_c))
    
    projected_A = sum([proj_c[k] * Bs[k] for k in range(num_classes)])
    return projected_A.detach().cpu(), forward(proj_c).detach().cpu().item()

def project_func(A, Bs,  method='pytorch', norm_type=2, optim_method='adam', epoch_num=100, lr=0.01, device=None):
    assert method in ['cvx', 'pytorch', 'cvx_sdp'], 'Projection method not supported.'
    if method == 'cvx':
        return project_cvx_func(A, Bs, norm_type=norm_type)
    elif method == 'cvx_sdp':
        return project_cvx_sdp_func(A, Bs, norm_type=norm_type)
    elif method == 'pytorch':
        return project_pytorch_func(A, Bs, norm_type=norm_type, 
                                    optim_method=optim_method, epoch_num=epoch_num, lr=lr, device=device)
    
def projection(feats, labels, num_classes, kernel_size, stride, padding, 
               method='pytorch', norm_type=2, optim_method='lbfgs', epoch_num=100, lr=0.01, device=None):
    # N C H W -> N (patch num) (kernel_size**2 * C)
    
    if len(feats.size()) == 4:
        patch_feats = reshape_conv_feats(feats, kernel_size, stride, padding)
    elif len(feats.size()) == 2:
        patch_feats = feats.reshape(feats.size(0), feats.size(1), 1)
    print('patch_feats', patch_feats.size())
    # Compute the feat means
    class_idxs = list(range(num_classes))
    patch_feats_dict = [[] for _ in class_idxs]
    
    truncated_num = 100
    for idx, label in enumerate(labels):
        label_ = int(label.item())
        if len(patch_feats_dict[label_]) >= truncated_num:
            continue
        patch_feats_dict[label_].append(patch_feats[idx])
        
    # K * (patch num) (kernel_size**2 * C)
    patch_feats_means = [torch.stack(patch_feats_dict[cls_idx]).mean(dim=0) for cls_idx in class_idxs]

    # Projection by minimize || A - c_1 B_1 - c_2 B_2 ... c_K B_K ||_S
    # A: [N x P x CK**2], B_i: [P x CK**2]
    truncated_classes = 10
    As = []
    truncated_labels = []
    for cls_idx in range(truncated_classes):
        As = As + patch_feats_dict[cls_idx]
        truncated_labels = truncated_labels + [cls_idx for _ in range(len(patch_feats_dict[cls_idx]))]
    Bs = patch_feats_means

    # [patch_feats_mean.cpu().numpy() for patch_feats_mean in patch_feats_means]
    projected_As = []
    for n in range(len(As)):
        A = As[n]
        projected_A, min_val = project_func(A, Bs, method=method, norm_type=norm_type, optim_method=optim_method,
                                   epoch_num=epoch_num, lr=lr, device=device)
        projected_As.append(projected_A)
        if n % 100 == 0:
            print(f'Projection Number {n}, min value:{min_val}')
    projected_feats_dict = [[] for _ in class_idxs]
    for i, label_ in enumerate(truncated_labels):
        projected_feats_dict[label_].append(projected_As[i])
    projected_feats_means = [torch.stack(projected_feats_dict[cls_idx]).mean(dim=0) for cls_idx in class_idxs]

    return projected_As, truncated_labels, projected_feats_means

def sigma_to_mu(projected_feats, projected_labels, projected_feats_means, dist_func):
    # within distances
    dist_within = 0
    N = len(projected_feats)
    for n in range(N):
        label_ = int(projected_labels[n])
        feats_mean = projected_feats_means[label_]
        dist_within += dist_func(projected_feats[n], feats_mean)
        
    dist_within = dist_within / N

    K = len(projected_feats_means)
    dist_between = 0
    for i in range(K-1):
        for j in range(i+1, K):
            dist_between += dist_func(projected_feats_means[i], projected_feats_means[j])
    dist_between = dist_between / ( K * (K-1) / 2)
    return dist_within, dist_between

def projected_distance(feats, labels, num_classes, kernel_size, stride, padding, 
                       method='cvx', proj_norm_type=2, dist_norm_type=2, dist_type='mean',
                       optim_method='lbfgs', epoch_num=100, lr=0.1, device=None):
    # proj As:[N x P x CK**2], B_i: [P x CK**2]
    projected_feats, projected_labels, projected_feats_means = projection(feats, labels, num_classes, kernel_size, stride, padding, 
                                                                method=method, norm_type=proj_norm_type, optim_method=optim_method,
                                                                epoch_num=epoch_num, lr=lr, device=device)
    # print("shape", len(projected_feats), len(projected_feats_means))
    if device is None:
        device = torch.device('cpu')
    
    def dist_func(x, y):
        x, y = x.to(device), y.to(device)
        if len(x.size()) == 1:
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
        return torch.linalg.matrix_norm(x-y, dist_norm_type).detach().cpu().item()
        
    # within distances
    dist_within = 0
    dist_within2 = 0
    N = len(projected_feats)
    for n in range(N):
        label_ = int(projected_labels[n])
        feats_mean = projected_feats_means[label_]
        dist_within += dist_func(projected_feats[n], feats_mean)
        dist_within2 += dist_func(projected_feats[n], feats_mean)**2
    dist_within = dist_within / N

    # between distances
    # if dist_type == 'mean':
    K = len(projected_feats)
    whole_feat_mean = sum(projected_feats) / len(projected_feats)
    dist_between = 0
    for i in range(K):
        dist_between += dist_func(projected_feats[i], whole_feat_mean)
    dist_between = dist_between / K
    mean_dist_between = dist_between
    # else:
    K = len(projected_feats_means)
    dist_between = 0
    for i in range(K-1):
        for j in range(i+1, K):
            dist_between += dist_func(projected_feats_means[i], projected_feats_means[j])
    dist_between = dist_between / ( K * (K-1) / 2)
    
    print(dist_within, dist_between, mean_dist_between)
    return dist_within, dist_between, mean_dist_between, (projected_feats, projected_labels, projected_feats_means)

def norm_distance(feats, labels, num_classes, kernel_size, stride, padding,
                  norm_type=2, dist_type='mean',
                  device=None):
    """
    Compute fuzziness according to features and corresponding labels.
    Args:
        feats[torch.tensor, N x [*]]: features used to compute fuzziness
        labels[torch.tensor, N]: labels of each sample
    Return:
        fuzziness[float]
    """
    if device is None:
        device = torch.device('cpu')
    # print(feats.size(), kernel_size, stride, padding)
    patch_feats = feats
    if len(feats.size()) == 4:
        if norm_type == 'orifro':
            N, C, H, W = feats.size()
            patch_feats = feats.reshape(N, C, H*W)
            norm_type = 'fro'
        else:
            patch_feats = reshape_conv_feats(feats, kernel_size, stride, padding)

    # Compute the feat means
    class_idxs = list(range(num_classes))
    patch_feats_dict = [[] for _ in class_idxs]
    
    for idx, label in enumerate(labels):
        label_ = int(label.item())
        patch_feats_dict[label_].append(patch_feats[idx])
        
    # K * (patch num) (kernel_size**2 * C)
    patch_feats_means = [torch.stack(patch_feats_dict[cls_idx]).mean(dim=0) for cls_idx in class_idxs]

    def dist_func(x, y):
        x, y = x.to(device), y.to(device)
        if len(x.size()) == 1:
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
        return (torch.linalg.matrix_norm(x-y, norm_type)).detach().cpu().item()
    
    # within distances
    dist_within = 0
    N = len(patch_feats)
    for n in range(N):
        label_ = int(labels[n])
        feats_mean = patch_feats_means[label_]
        # dist_within += dist_func(patch_feats[n], feats_mean)
        dist_within = dist_func(patch_feats[n], feats_mean)
    dist_within = dist_within / N
    # between distances
    
    K = len(patch_feats_means)
    whole_feat_mean = sum(patch_feats_means) / len(patch_feats_means)
    dist_between = 0
    for i in range(K):
        dist_between += dist_func(patch_feats_means[i], whole_feat_mean)
    dist_between = dist_between / K
    mean_dist_between = dist_between
    
    dist_between = 0
    for i in range(K-1):
        for j in range(i+1, K):
            dist_between += dist_func(patch_feats_means[i], patch_feats_means[j])
    dist_between = dist_between / ( K * (K-1) / 2)

    print(dist_within, dist_between, mean_dist_between)
    return dist_within, dist_between, mean_dist_between

def distance_WB(feats, labels, num_classes=10):
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
    # feats = feats[:,:,0,0]
    feats = feats.reshape(feats.size(0), -1).to(device)  # [N x C]
    C = feats.size(1)

    feats = feats.to(device)

    whole_feat_mean = feats.mean(dim=0)  # [C]

    feats = feats - whole_feat_mean.view(1, -1)
    
    # compute mean
    class_idxs = list(range(num_classes))
    feats_dict = [[] for _ in class_idxs]  #{0:[C], 1:[C], 2:[C], ...}
    
    for idx, label in enumerate(labels):
        label_ = int(label.item())
        feats_dict[label_].append(feats[idx])
    
    feats_means = [torch.stack(feats_dict[cls_idx]).mean(dim=0) for cls_idx in class_idxs]

    # Compute the cos angle, between class variation
    d_bs = torch.zeros(num_classes, num_classes)
    for i in class_idxs:
        for j in class_idxs:
            d_bs[i, j] = F.cosine_similarity(feats_means[i], feats_means[j], dim=0)
    d_b = d_bs.mean()

    # Compute the within class variation
    d_ws = torch.zeros(num_classes)
    for k in class_idxs:
        feats_k = torch.stack(feats_dict[k])  #N_k, C
        feat_mean_k = feats_means[k]  # C
        feats_k = F.pairwise_distance(feats_k, feat_mean_k) / feat_mean_k.size(0)
        d_ws[k] = feats_k.mean()
    d_w = d_ws.mean()

    return d_b, d_w

def signal_to_noise(feats, labels, num_classes):
    """
    Compute signal_to_noise according to features and corresponding labels.
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
        
        diff_b = feat_mean 
        signal_b = signal_b + \
                      torch.matmul(diff_b.reshape(-1, 1), diff_b.reshape(1, -1)) * len(feat_means[k])
        
        diff_w =  class_feats - feat_mean.view(1, -1)
        
        signal_w = signal_w + torch.matmul(torch.transpose(diff_w, 0, 1), diff_w)

    signal_b = signal_b / float(feats.size(0))
    signal_w = (signal_w / float(feats.size(0))).cpu()
    try:
        inv_signal_b = torch.linalg.pinv(signal_b, rcond=1e-5)
        signal_b = signal_b.cpu()

        D_seperation = torch.trace(torch.matmul(signal_w.to(device), inv_signal_b)) 
        return D_seperation, signal_b.cpu(), signal_w.cpu()
    except:
        return torch.tensor(1000), signal_b.cpu(), signal_w.cpu()

def svd_max_cov(x):
    # U, S, Vh = torch.linalg.svd(x)
    # print(S)
    return torch.matmul(x, x.transpose(0,1))
    
    # return (U[:,0] * S[0]**2 * U[:,0]).sum()

def general_signal_to_noise(feats, labels, num_classes, dim, kernel_size, stride, padding):
    """
    Try to use svd
    Compute signal_to_noise according to features and corresponding labels.
    Args:
        feats[torch.tensor, N x [C H W]]: features used to compute fuzziness
        labels[torch.tensor, N]: labels of each sample
        dim[torch.tensor]: 1, 2
    Return:
        fuzziness[float]
    """
    device = labels.device
    feats = copy.deepcopy(feats)
    labels = copy.deepcopy(labels)
    # feats = feats.reshape(feats.size(0), -1).to(device)  # [N x C]
    patch_feats = reshape_conv_feats(feats, kernel_size, stride, padding)  # N x [Patch num, kernel_size**2 * C]
    # patch_feats = feats.reshape(feats.size(0), feats.size(1), -1)
    feats = patch_feats.to(device)
    C1, C2 = feats.size(1), feats.size(2)
    
    whole_feat_mean = feats.mean(dim=0)  # [Patch_num,  Kernel_size**2 * C]

    feats = feats - whole_feat_mean
    
    # compute mean
    class_idxs = list(range(num_classes))
    feat_means = [[] for _ in class_idxs]  #{0:[C], 1:[C], 2:[C], ...}
    
    for idx, label in enumerate(labels):
        label_ = int(label.item())
        feat_means[label_].append(feats[idx])
    feats = feats.cpu()
    C = C1 if dim == 1 else C2
    signal_b = torch.zeros(C, C).to(device)
    signal_w = torch.zeros(C, C).to(device)

    for k in class_idxs:
        class_feats = torch.stack(feat_means[k])  # N_k P KKC
        feat_mean = class_feats.mean(dim=0)  # P KKC
        
        diff_b = feat_mean  # [P, KKC]
        
        if dim == 2:
            diff_b = diff_b.transpose(0, 1)
        # print(diff_b.size())
        signal_b = signal_b + svd_max_cov(diff_b)
                    #   torch.matmul(diff_b, diff_b.transpose(0, 1)) * len(feat_means[k])
                    
    diff_w =  class_feats - feat_mean.view(1, C1, C2)
    
    if dim == 2:
        diff_w = diff_w.transpose(1, 2)
    # print(diff_w.size())
    for tp in range(diff_w.size(0)):
        signal_w = signal_w + svd_max_cov(diff_w[tp])
        # torch.matmul(diff_w[tp], diff_w[tp].transpose(0, 1))

    signal_b = signal_b / float(feats.size(0))
    signal_w = (signal_w / float(feats.size(0))).cpu()
    try:
        inv_signal_b = torch.linalg.pinv(signal_b, rcond=1e-5)
        signal_b = signal_b.cpu()

        D_seperation = torch.trace(torch.matmul(signal_w.to(device), inv_signal_b)) 
        return D_seperation, signal_b.cpu(), signal_w.cpu()
    except:
        return torch.tensor(1000), signal_b.cpu(), signal_w.cpu()
