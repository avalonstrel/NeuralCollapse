import math
from functools import partial
from collections import Iterator, Iterable
from typing import Any, cast, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops import MLP, Permute
from torchvision.ops.stochastic_depth import StochasticDepth


# from .vggnet import make_layers as vgg_layers
# make_layers(cfg: List[Union[str, int]], input_size: List[int], norm_type: str = 'bn') -> nn.Sequential:

from .resnet import BasicBlock, Bottleneck

from .densenet import _DenseBlock, _Transition

from .inception import InceptionA, InceptionB, InceptionC, InceptionD

from .vit import EncoderBlock

from .swintransformer import SwinTransformerBlock, PatchMerging

from .odenet import ODEBlock

NORM_LAYERS = {
    'bn':nn.BatchNorm2d,
    'ln':nn.LayerNorm,
    'in':nn.InstanceNorm2d,

}
class PreViTTrans(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, patch_size, dropout=0.):
        super().__init__()
        self.patch_size = patch_size
        self.conv_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size, stride=patch_size)
        seq_length = (in_size // patch_size)**2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, out_channels).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, c, h, w = x.size()
        p = self.patch_size
        n_h = h // p
        n_w = w // p
        
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # print(n,c,h,w,x.size())
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, -1, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        # print(x.size(), )
        return x + self.pos_embedding

class PostViTTrans(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.):
        super().__init__()
        self.conv_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        n, hw, c = x.size()
        h = w = int(hw**(0.5))
        x = x.permute(0, 2, 1)
        x = x.reshape(n, c, h, w)

        x = self.conv_proj(x)
        # (n, c, h, w) -> (n, c, h, w)
        return x

class PreSwinTrans(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.permute(0, 2, 3, 1)

class PostSwinTrans(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class PostFCTrans(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):

        b, whole_channels = x.size()
        hw = whole_channels // self.channels
        h = w = int(hw**(0.5))
        return x.reshape(b, self.channels, h, w)

class FCLayer(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        norm = None
        act = True
        if len(cfg) == 2:
            in_channels, out_channels = cfg
        elif len(cfg) == 3:
            in_channels, out_channels, norm = cfg
        elif len(cfg) == 4:
            in_channels, out_channels, norm, act = cfg
            if isinstance(act, str):
                act = eval(act)
        self.fc = nn.Linear(in_channels, out_channels)
        if norm is not None and norm == 'bn':
            self.norm = nn.BatchNorm1d(out_channels)
        if act:
            self.act = nn.ReLU(True)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        if hasattr(self, 'act'):
            x = self.act(x)
        return x

class NormLayer(nn.Module):
    def __init__(self, input_size, norm_type='bn'):
        super().__init__()
        self.norm_type = norm_type
        self.norm = None
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(input_size[0])
        elif norm_type == 'ln':
            self.norm = nn.LayerNorm(input_size)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(input_size[0])

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        return x

class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.flatten(x, 1)
        
def vgg_layers(cfg, norm_type='bn', tmp_inputs=None):
    layers: List[nn.Module] = []
    input_size = tmp_inputs.size()[1:]
    in_channels = input_size[0]
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            tmp_inputs = layers[-1](tmp_inputs)
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            tmp_inputs = layers[-1](tmp_inputs)
        elif v == 'DNC':  # Down Conv
            layers += [nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)]
            tmp_inputs = layers[-1](tmp_inputs)
        elif v == 'UPC':  # Up Conv
            layers += [nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)]
            tmp_inputs = layers[-1](tmp_inputs)
        elif v == 'UPI':  # Up Interpolation
            layers += [nn.Upsample(scale_factor=2)]
            tmp_inputs = layers[-1](tmp_inputs)
        else:
            kernel_size = 3
            stride = 1
            padding = 1
            activation = True
            
            if isinstance(v, Iterable):
                if len(v) == 3:
                    v, kernel_size, padding = [cast(int, v_) for v_ in v]
                elif len(v) == 4:
                    activation = eval(v[3])
                    v, kernel_size, padding = [cast(int, v_) for v_ in v[:3]]
                elif len(v) == 5:
                    activation = eval(v[4])
                    v, kernel_size, stride, padding = [cast(int, v_) for v_ in v[:4]]
            else: 
                v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, stride=stride, padding=padding)
            tmp_inputs = conv2d(tmp_inputs)
            if norm_type == 'bn':
                layers += [conv2d, nn.BatchNorm2d(v)]
            elif norm_type == 'ln':
                layers += [conv2d, nn.LayerNorm(tmp_inputs.size()[1:])]
            elif norm_type == 'in':
                layers += [conv2d, nn.InstanceNorm2d(v)]
            else:
                layers += [conv2d]

            if activation:
                layers += [nn.ReLU(inplace=True)]

            # if norm_type in ['bn', 'ln', 'in']:
            #     if activation:
            #         tmp_inputs = layers[-2](tmp_inputs)
            #     else:
            #         tmp_inputs = layers[-1](tmp_inputs)

            in_channels = v
    return nn.Sequential(*layers)

BLOCKS = {
    'VGGBlock':vgg_layers,
    'ResBlock':BasicBlock,
    'ResBottleneck':Bottleneck,
    'InceptionA':InceptionA,
    'InceptionB':InceptionB,
    'InceptionC':InceptionC,
    'InceptionD':InceptionD,
    'ViTBlock':EncoderBlock,
    'SwinBlock':SwinTransformerBlock,
    'PatchMerging':PatchMerging,
    'PreViTTrans':PreViTTrans,
    'PostViTTrans':PostViTTrans,
    'PreSwinTrans':PreSwinTrans,
    'PostSwinTrans':PostSwinTrans,
    'FC':FCLayer,
    'PostFCTrans':PostFCTrans,
    'DenseBlock':_DenseBlock,
    'Transition':_Transition,
    'Flatten':Flatten,
}

# TEMPLATE_BLOCKS = [
#     'vgg_layers': (cfg: [64,'M'],  norm_type: 'bn', tmp_inputs),  # Simple CNN Layers
#     'BasicBlock': (inplanes, planes, stride = 1, norm_type = 'bn', tmp_inputs = None),
#     'Bottleneck': (inplanes, planes, stride = 1, norm_type = 'bn', tmp_inputs = None),
#     'InceptionA': (in_channels, pool_features),
#     'InceptionB': (in_channels),
#     'InceptionC': (in_channels, channels_7x7),
#     'InceptionD': (in_channels),
#     'ViTBlock': (num_heads, hidden_dim, mlp_dim,),
#     'SwinBlock': (dim, num_heads, window_size: List[int], shift_size: List[int],),
#     'PatchMerging': (dim:input C),
#     '_DenseBlock': (num_layers, num_input_features, bn_size, growth_rate, drop_rate,)
#     '_Transition': (num_input_features, num_output_features)
# ]

class HybridNetClassifier(nn.Module):
    '''
    Construct Classifier with different kinds of blocks. 
    Args:
        num_classes[int]: the number of classes
        structures[list]: e.g., [(block_type, args:(a,b,c)), ]
    '''
    def __init__(self,
        num_classes,
        structures, 
        image_size=128,
        norm_type='bn',
        ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.norm_type = norm_type
        # for transformer
        
        self.build_models(structures)

        self.init_weights()

    def build_models(self, structures):
        # input conv
        x = torch.rand(2, 3, self.image_size, self.image_size)
        tmp_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        x = tmp_conv(x)
        self.in_conv = nn.Sequential(*[
                tmp_conv, 
                NormLayer(x.size()[1:], 'bn'),
                nn.ReLU(inplace=True)
            ])
        # N 64 H W
        blocks = []
        for (block, block_args) in structures:
            kwargs = {}
            if block in ['VGGBlock', 'ResBlock', 'ResBottleneck']:
                kwargs['tmp_inputs'] = x
            if block in ['VGGBlock', 'ResBlock', 'ResBottleneck']:
                kwargs['norm_type'] = self.norm_type
            tmp_block = BLOCKS[block](*block_args, **kwargs)
            
            blocks.append(tmp_block)
            x = tmp_block(x)

        self.blocks = nn.Sequential(*blocks)
        
        x = torch.flatten(x, 1)
        self.fc = nn.Linear(x.size(1), self.num_classes)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)    

    def get_features(self, x, feat_type='relu'):
        outputs = []
        x = self.in_conv(x)
        outputs.append(x.detach().cpu())
        for block in self.blocks:
            

            if isinstance(block, (PreViTTrans, EncoderBlock)):
                x = block(x)
                n, hw, c = x.size()
                h = w = int(hw**(0.5))
                outputs.append(x.reshape(n, h, w, c).permute(0,3,1,2).detach().cpu())    
            elif isinstance(block, (PreSwinTrans, SwinTransformerBlock)):
                x = block(x)
                outputs.append(x.permute(0,3,1,2).detach().cpu())
            else:
                # # For BatchNorm Now
                # if isinstance(block, nn.Sequential):
                #     for layer in block:
                #         x = layer(x)
                #         if isinstance(layer, nn.ReLU):
                #             outputs.append(x.detach().cpu())
                # else:
                x = block(x)
                outputs.append(x.detach().cpu())
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return outputs

    def forward(self, x, onnx=False):
        x = self.in_conv(x)
        for block in self.blocks:
            if isinstance(block, (EncoderBlock, SwinTransformerBlock)):
                x = block(x, onnx=onnx)
            else:
                x = block(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class PureHybridNetClassifier(nn.Module):
    '''
    Construct Classifier with different kinds of blocks. 
    Args:
        num_classes[int]: the number of classes
        structures[list]: e.g., [(block_type, args:(a,b,c)), ]
    '''
    def __init__(self,
        num_classes,
        structures, 
        image_size=128,
        norm_type='bn',
        ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.norm_type = norm_type
        
        self.build_models(structures)

        self.init_weights()

    def build_models(self, structures):
        # input conv
        x = torch.rand(2, 3, self.image_size, self.image_size)
        # N 64 H W
        blocks = []
        for (block, block_args) in structures:
            kwargs = {}
            if block in ['VGGBlock', 'ResBlock', 'ResBottleneck']:
                kwargs['tmp_inputs'] = x
            if block in ['VGGBlock', 'ResBlock', 'ResBottleneck']:
                kwargs['norm_type'] = self.norm_type
            tmp_block = BLOCKS[block](*block_args, **kwargs)
            blocks.append(tmp_block)
            # print(tmp_block, x.size())
            x = tmp_block(x)

        self.blocks = nn.Sequential(*blocks)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)    

    def get_features(self, x, feat_type='relu'):
        outputs = []
        # do not contain the last layer
        for block in self.blocks[:-1]:
            if isinstance(block, (PreViTTrans, EncoderBlock)):
                x = block(x)
                n, hw, c = x.size()
                h = w = int(hw**(0.5))
                outputs.append(x.reshape(n, h, w, c).permute(0,3,1,2).detach().cpu())    
            elif isinstance(block, (PreSwinTrans, SwinTransformerBlock)):
                x = block(x)
                outputs.append(x.permute(0,3,1,2).detach().cpu())
            else:
                if isinstance(block, (Flatten)):
                    x = block(x)
                else:
                    x = block(x)
                    outputs.append(x.detach().cpu())

        return outputs

    def forward(self, x, onnx=False):
        for block in self.blocks:
            if isinstance(block, (EncoderBlock, SwinTransformerBlock)):
                x = block(x, onnx=onnx)
            else:
                x = block(x)
        return x


class ODEAddWarpper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, t, x):
        x_t = torch.ones_like(x) * t
        return self.block(x + x_t)



class HybridODEClassifier(nn.Module):
    '''
    Construct Classifier with different kinds of blocks. 
    Args:
        num_classes[int]: the number of classes
        structures[list]: e.g., [(block_type, args:(a,b,c)), ]
    '''
    def __init__(self,
        num_classes,
        structures, 
        image_size=128,
        norm_type='bn',
        atol=1e-3, 
        rtol=1e-3,
        ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.norm_type = norm_type
        self.atol = atol
        self.rtol = rtol
        # for transformer
        
        self.build_models(structures)

        self.init_weights()

    def build_models(self, structures):
        # input conv
        x = torch.rand(2, 3, self.image_size, self.image_size)
        tmp_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        x = tmp_conv(x)
        self.in_conv = nn.Sequential(*[
                tmp_conv, 
                NormLayer(x.size()[1:], 'bn'),
                nn.ReLU(inplace=True)
            ])
        # N 64 H W
        blocks = []
        pre_transformer = False  # whether the previous block is transformer based block
        for (is_ode, block, block_args) in structures:
            kwargs = {}
            if block in ['VGGBlock', 'ResBlock', 'ResBottleneck']:
                kwargs['tmp_inputs'] = x
            if block in ['VGGBlock', 'ResBlock', 'ResBottleneck']:
                kwargs['norm_type'] = self.norm_type
            tmp_block = BLOCKS[block](*block_args, **kwargs)
            if is_ode:
                tmp_block = ODEBlock(ODEAddWarpper(tmp_block), atol=self.atol, rtol=self.rtol)
            blocks.append(tmp_block)
            x = tmp_block(x)

        self.blocks = nn.Sequential(*blocks)
        
        x = torch.flatten(x, 1)
        self.fc = nn.Linear(x.size(1), self.num_classes)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)    

    def get_features(self, x, feat_type='relu', t_list=None):
        outputs = []
        x = self.in_conv(x)
        outputs.append(x.detach().cpu())
        for block in self.blocks:
            if isinstance(block, ODEBlock):
                for t_i in range(len(t_list)-1):
                    x = block(x, t_list[t_i:t_i+2])
                    outputs.append(x.detach().cpu())
            else:    
                x = block(x)

                if isinstance(block, (PreViTTrans, EncoderBlock)):
                    n, hw, c = x.size()
                    h = w = int(hw**(0.5))
                    outputs.append(x.reshape(n, h, w, c).permute(0,3,1,2).detach().cpu())    
                elif isinstance(block, (PreSwinTrans, SwinTransformerBlock)):
                    outputs.append(x.permute(0,3,1,2).detach().cpu())    
                else:
                    outputs.append(x.detach().cpu())

        x = torch.flatten(x, 1)
        x = self.fc(x)
        outputs.append(x.detach().cpu())
        return outputs

    def forward(self, x, onnx=False):
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x





