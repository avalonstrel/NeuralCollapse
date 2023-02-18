from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

CFGS = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # vgg13
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],  # vgg16
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],  # vgg19
    "vgg13_bn": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # vgg13
    "vgg16_bn": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],  # vgg16
    "vgg19_bn": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],  # vgg19
}


class NormLayer(nn.Module):
    def __init__(self, input_size, norm_type='bn', no_mean=False, no_var=False):
        super().__init__()
        self.norm_type = norm_type
        self.norm = None
        if norm_type == 'bn':
            if no_mean or no_var:
                self.norm = nn.BatchNorm2d(input_size[0], affine=False)
            else:
                self.norm = nn.BatchNorm2d(input_size[0])
        elif norm_type == 'ln':
            self.norm = nn.LayerNorm(input_size)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(input_size[0])
        self.no_mean = no_mean
        self.no_var = no_var

    def forward(self, x):
        
        if self.norm_type == 'bn':

            x = self.norm(x)
            if self.no_mean or self.no_var:
                r_m = self.norm.running_mean.view(1, -1, 1, 1)
                r_v = self.norm.running_var.view(1, -1, 1, 1)
                if self.no_var:
                    x = x * r_v
                    
                if self.no_mean:
                    x = x * r_v + r_m
                    x = x / (r_v + self.norm.eps)
                    
        else:
            if self.norm is not None:
                x = self.norm(x)
        return x

def make_layers(cfg: List[Union[str, int]], input_size: List[int], norm_type: str = 'bn', no_mean=False, no_var=False) -> nn.Sequential:
    layers: List[nn.Module] = []
    tmp_inputs = torch.rand(1, 3, *input_size)
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            tmp_inputs = layers[-1](tmp_inputs)
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            tmp_inputs = conv2d(tmp_inputs)
            if norm_type in ['bn', 'ln', 'in']:
                layers += [conv2d, NormLayer(tmp_inputs.size()[1:], norm_type=norm_type, no_mean=no_mean, no_var=no_var), nn.ReLU(inplace=True)]
                tmp_inputs = layers[-2](tmp_inputs)
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            # if norm_type == 'bn':
            #     layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            # elif norm_type == 'ln':
            #     layers += [conv2d, nn.LayerNorm(tmp_inputs.size()[1:]), nn.ReLU(inplace=True)]
            # elif norm_type == 'in':
            #     layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
            # else:
            #     layers += [conv2d, nn.ReLU(inplace=True)]
            # if norm_type in ['bn', 'ln', 'in']:
            #     tmp_inputs = layers[-2](tmp_inputs)

            in_channels = v
    return tmp_inputs, nn.Sequential(*layers)

class VGGClassifier(nn.Module):
    def __init__(
        self, num_classes, norm_type='bn', dropout=0., image_size=32, model_type='vgg16', 
        no_mean=False, no_var=False, pretrained=False
    ) -> None:
        super().__init__()
        tmp_inputs, features = make_layers(CFGS[model_type], (image_size, image_size), norm_type=norm_type, no_mean=no_mean, no_var=no_var)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        tmp_inputs = self.avgpool(tmp_inputs)
        tmp_inputs = torch.flatten(tmp_inputs, 1)
        self.classifier = nn.Sequential(
            nn.Linear(tmp_inputs.size(1), 4096),
            nn.ReLU(True),
            # nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        self.norm_type = norm_type
        self.model_type = model_type
        self.no_mean, self.no_var = no_mean, no_var
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) and not (self.no_mean or self.no_var):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def layer_idxs(self, model_type, layer_type):
        cfg = CFGS[model_type]
        layer_offset = {
                'conv': 0, 
                'relu': 1
            }
        if 'bn' in model_type:
            layer_offset = {
                'conv': 0,
                'bn': 1, 
                'relu': 2
            }
        
        layer_idxs = []
        layer_idx = 0
        for channel in cfg:
            if channel == 'M':
                layer_idx += 1
            else:
                layer_idxs.append(layer_idx + layer_offset[layer_type])
                layer_idx += len(layer_offset)

        return layer_idxs

    def get_features(self, x, feat_type='relu'):
        check_layer_idxs = self.layer_idxs(self.model_type, 'relu')
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            if i in check_layer_idxs:
                # print(layer)
                # outputs[lt].append(x.reshape(x.size(0),-1)[:,:256].detach().cpu())
                outputs.append(x.detach().cpu())
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i in [1,3]:
                outputs.append(x.detach().cpu())
        outputs.append(x.detach().cpu())
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# class VGGClassifier(nn.Module):
#     """Take the feature embedding as input, output the feature for classification
#     """
#     def __init__(self, num_classes, dropout=0.5, model_type='vgg16', pretrained=False):
#         super().__init__()
#         _vgg = VGG_DICT[model_type](pretrained=pretrained)
#         self.model_type = model_type
#         self.features = _vgg.features
#         self.avgpool = _vgg.avgpool
#         self.classifier = nn.Sequential(
#             # nn.Linear(512 * 7 * 7, 4096),
#             nn.Linear(512, 4096),
#             nn.ReLU(True),
#             # nn.Dropout(p=dropout),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             # nn.Dropout(p=dropout),
#             nn.Linear(4096, num_classes),
#         )
        
#         if not pretrained:
#             self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
        
#     def layer_idxs(self, model_type, layer_type):
#         cfg = CFGS[model_type]
#         layer_offset = {
#                 'conv': 0, 
#                 'relu': 1
#             }
#         if 'bn' in model_type:
#             layer_offset = {
#                 'conv': 0,
#                 'bn': 1, 
#                 'relu': 2
#             }
        
#         layer_idxs = []
#         layer_idx = 0
#         for channel in cfg:
#             if channel == 'M':
#                 layer_idx += 1
#             else:
#                 layer_idxs.append(layer_idx + layer_offset[layer_type])
#                 layer_idx += len(layer_offset)

#         return layer_idxs

#     def get_features(self, x, feat_type='relu'):
#         check_layer_idxs = self.layer_idxs(self.model_type, feat_type)
#         outputs = []
#         for i, layer in enumerate(self.features):
#             x = layer(x)
            
#             if i in check_layer_idxs:
#                 # print(layer)
#                 # outputs[lt].append(x.reshape(x.size(0),-1)[:,:256].detach().cpu())
#                 outputs.append(x.detach().cpu())
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         outputs.append(x.detach().cpu())
#         return outputs

#     def forward(self, x):
#         x = self.features(x)
#         # print('1',x.size())
#         # x = self.avgpool(x)
#         # print(x.size())
#         x = torch.flatten(x, 1)

#         x = self.classifier(x)
        
#         return x
