import os
import numpy as np
import torch
import torch.nn as nn


# if args.adjoint:
from torchdiffeq import odeint_adjoint as odeint
# else:
#     from torchdiffeq import odeint


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out

class LODEfunc(nn.Module):

    def __init__(self, dim):
        super(LODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, 2*dim, 3, 1, 1)
        self.norm2 = norm(2*dim)
        self.conv2 = ConcatConv2d(2*dim, 2*dim, 3, 1, 1)
        self.norm3 = norm(2*dim)
        self.conv3 = ConcatConv2d(2*dim, dim, 3, 1, 1)
        self.norm4 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(t, out)
        out = self.norm4(out)
        return out

class ODEBlock(nn.Module):
    def __init__(self, odefunc, atol=1e-3, rtol=1e-3):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()  # default 0, 1 
        self.atol = atol
        self.rtol = rtol
    def forward(self, x, t=None, atol=None, rtol=None, return_all=False):
        if t is None:
            integration_time = self.integration_time.type_as(x)
        else:
            integration_time = torch.tensor(t).type_as(x)  # t=[0,1]; [0,0.5,1]
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        out = odeint(self.odefunc, x, integration_time, rtol=rtol, atol=atol)
        if return_all:
            return out
        else:
            return out[-1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

ODEFUNC_DICT = {
    'odenet':ODEfunc,
    'lodenet':LODEfunc,
}

class ODEClassifier(nn.Module):
    def __init__(self, num_classes, model_type, atol=1e-3, rtol=1e-3, t_list=[0,1]):
        super().__init__()

        # Input / Downsampling Layers
        downsampling_layers = [
            nn.Conv2d(3, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
        ]
        # downsampling_layers = [
        #     nn.Conv2d(1, 64, 3, 1),
        #     ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        #     ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        # ]
        self.downsampling_layers = nn.Sequential(*downsampling_layers)
        # Features Layers(ODE)
        # feature_layers = ODEBlock(ODEfunc(64), t_list)
        self.feature_layers = ODEBlock(ODEFUNC_DICT[model_type](64), atol=atol, rtol=rtol)
        fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, num_classes)]
        self.fc_layers = nn.Sequential(*fc_layers)
        self.t_list = t_list

    def get_features(self, x, feat_type=None, t_list=None):
        outputs = []
        for l_i, layer in enumerate(self.downsampling_layers):
            x = layer(x)
            if (l_i + 1) % 3 == 0:
                outputs.append(x.detach().cpu())
        if t_list is None:
            t = [0, 1]
        else:
            t = t_list
        ode_feats = []
        # xs = self.feature_layers(x, t, return_all=True)
        # ode_feats = [x_.detach().cpu() for x_ in  xs[1:]]
        for t_i in range(len(t_list)-1):
            x = self.feature_layers(x, t_list[t_i:t_i+2])
            ode_feats.append(x.detach().cpu())
        # print('odefeat', [feat[:2,0,0] for feat in ode_feats])
        outputs.extend(ode_feats)
        x = self.fc_layers(x)
        outputs.append(x.detach().cpu())
        return outputs

    def forward(self, x, t=None):
        if t is None:
            t = self.t_list
        x = self.downsampling_layers(x)
        x = self.feature_layers(x, t)
        # print(x.size())
        x = self.fc_layers(x)

        return x

    @property
    def nfe(self):
        return self.feature_layers.nfe

    @nfe.setter
    def nfe(self, value):
        self.feature_layers.nfe = value


from .vggnet import make_layers as vgg_layers
# make_layers(cfg: List[Union[str, int]], input_size: List[int], norm_type: str = 'bn') -> nn.Sequential:

from .resnet import BasicBlock, Bottleneck

from .inception import InceptionA, InceptionB, InceptionC, InceptionD

from .vit import EncoderBlock

from .swintransformer import SwinTransformerBlock, PatchMerging

class HybridODEClassifier(nn.Module):
    def __init__(self, num_classes, model_type, atol=1e-3, rtol=1e-3, t_list=[0,1]):
        super().__init__()

        # Input / Downsampling Layers
        downsampling_layers = [
            nn.Conv2d(3, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
        ]
        # downsampling_layers = [
        #     nn.Conv2d(1, 64, 3, 1),
        #     ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        #     ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        # ]
        self.downsampling_layers = nn.Sequential(*downsampling_layers)
        # Features Layers(ODE)
        # feature_layers = ODEBlock(ODEfunc(64), t_list)
        self.feature_layers = ODEBlock(ODEFUNC_DICT[model_type](64), atol=atol, rtol=rtol)
        fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, num_classes)]
        self.fc_layers = nn.Sequential(*fc_layers)
        self.t_list = t_list

    def get_features(self, x, feat_type=None, t_list=None):
        outputs = []
        for l_i, layer in enumerate(self.downsampling_layers):
            x = layer(x)
            if (l_i + 1) % 3 == 0:
                outputs.append(x.detach().cpu())
        if t_list is None:
            t = [0, 1]
        else:
            t = t_list
        ode_feats = []
        # xs = self.feature_layers(x, t, return_all=True)
        # ode_feats = [x_.detach().cpu() for x_ in  xs[1:]]
        for t_i in range(len(t_list)-1):
            x = self.feature_layers(x, t_list[t_i:t_i+2])
            ode_feats.append(x.detach().cpu())
        # print('odefeat', [feat[:2,0,0] for feat in ode_feats])
        outputs.extend(ode_feats)
        x = self.fc_layers(x)
        outputs.append(x.detach().cpu())
        return outputs

    def forward(self, x, t=None):
        if t is None:
            t = self.t_list
        x = self.downsampling_layers(x)
        x = self.feature_layers(x, t)
        # print(x.size())
        x = self.fc_layers(x)

        return x

    @property
    def nfe(self):
        return self.feature_layers.nfe

    @nfe.setter
    def nfe(self, value):
        self.feature_layers.nfe = value






