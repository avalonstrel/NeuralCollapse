
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import resnet50


class ResnetBackbone(nn.Module):
    """Take the feature embedding as input, output the feature for classification
    """
    def __init__(self):
        super().__init__()
        _resnet50 = resnet50(pretrained=True)
        self.maxpool = _resnet50.maxpool
        self.layer1 = _resnet50.layer1
        self.layer2 = _resnet50.layer2
        self.layer3 = _resnet50.layer3
        self.layer4 = _resnet50.layer4
        self.avgpool = _resnet50.avgpool

    def init_weights(self):
        pass # now need
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
