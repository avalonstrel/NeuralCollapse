import torch
import torch.nn as nn

from models.modules import ConvModule

def simple_conv_layer(
    in_channels, out_channels,
    kernel_size, stride):
    return ConvModule(
             in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=kernel_size,
             stride=stride,
             norm_cfg=dict(type='BN'),
             act_cfg=dict(type='ReLU'),
             order=('conv', 'norm', 'act')
        )

class ConvEmbedding(nn.Module):
    """An embedding network which transforms the input data into a shared feature space.
       Args:
            structures (Tuple[Tuple[int, int, int]]): [[channel, kernel_size, stride,],...] 
    """
    def __init__(
        self, 
        structures):
        super().__init__()
        self.conv_layers = self.build_model(structures)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.init_weights()

    def build_model(self, structures):
        conv_layers = []
        for i in range(len(structures)-1):
            kernel_size, stride = structures[i][1], structures[i][2]
            in_channels, out_channels = structures[i][0], structures[i + 1][0]
            conv_layer = simple_conv_layer(in_channels, out_channels, kernel_size, stride)
            conv_layers.append(conv_layer)

        return nn.ModuleList(conv_layers)

    def init_weights(self):
        # Do not need to initialize for embeddings (init when contructs.)
        pass

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        return self.maxpool(x)