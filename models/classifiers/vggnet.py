import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.vgg import vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn

VGG_DICT = {
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19': vgg19,
    'vgg19_bn': vgg19_bn,
}

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

CFGS = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # vgg13
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],  # vgg16
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],  # vgg19
    "vgg13_bn": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # vgg13
    "vgg16_bn": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],  # vgg16
    "vgg19_bn": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],  # vgg19
}

class VGGClassifier(nn.Module):
    """Take the feature embedding as input, output the feature for classification
    """
    def __init__(self, num_classes, dropout=0.5, model_type='vgg16', pretrained=False):
        super().__init__()
        _vgg = VGG_DICT[model_type](pretrained=pretrained)
        self.model_type = model_type
        self.features = _vgg.features
        self.avgpool = _vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        
        if not pretrained:
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
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

    def forward(self, x, layer_types=['conv', 'bn', 'relu']):

        check_layer_idxs = {lt:self.layer_idxs(self.model_type, lt) for lt in layer_types}
        outputs = {lt:[] for lt in layer_types}
        for i, layer in enumerate(self.features):
            x = layer(x)
            for lt in layer_types:
                if i in check_layer_idxs[lt]:
                    outputs[lt].append(x.reshape(x.size(0),-1)[:,:256].detach().cpu())
                    # outputs[lt].append(x.detach().cpu())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x, outputs
