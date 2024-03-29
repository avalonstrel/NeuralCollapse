import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.inception import inception_v3
import torch.nn.functional as F
import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

INCEPTION_DICT = {
    'inception_v3':inception_v3,
}

class InceptionClassifier(nn.Module):
    def __init__(
        self,
        num_classes = 1000,
        pretrained = False,
        aux_logits = False,
        transform_input = False,
        inception_blocks = None,
        init_weights = True,
        dropout: float = 0.5,
        image_size = 32,
        add_settings = (0,0),
    ) -> None:
        super().__init__()
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux]
        if len(inception_blocks) != 7:
            raise ValueError(f"length of inception_blocks should be 7 instead of {len(inception_blocks)}")
        tmp_inputs = torch.randn(2, 3, image_size, image_size)
        self.image_size = image_size
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]
        self.add_settings = add_settings
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        tmp_inputs = self.Conv2d_1a_3x3(tmp_inputs)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        tmp_inputs = self.Conv2d_2a_3x3(tmp_inputs)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        tmp_inputs = self.Conv2d_2b_3x3(tmp_inputs)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        tmp_inputs = self.maxpool1(tmp_inputs)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        tmp_inputs = self.Conv2d_3b_1x1(tmp_inputs)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        tmp_inputs = self.Conv2d_4a_3x3(tmp_inputs)

        
        if add_settings[0] > 0:
            self.add_convs = nn.Sequential(*[
                    conv_block(192, 192, kernel_size=1)
                    for _ in range(add_settings[0])
                ])
            tmp_inputs = self.add_convs(tmp_inputs)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        tmp_inputs = self.maxpool2(tmp_inputs)
        self.Mixed_5b = inception_a(192, pool_features=32)
        tmp_inputs = self.Mixed_5b(tmp_inputs)
        self.Mixed_5c = inception_a(256, pool_features=64)
        tmp_inputs = self.Mixed_5c(tmp_inputs)
        self.Mixed_5d = inception_a(288, pool_features=64)
        tmp_inputs = self.Mixed_5d(tmp_inputs)
        if add_settings[1] > 0:
            self.add_inceptions = nn.Sequential(*[
                    inception_a(288, pool_features=64)
                    for i in range(add_settings[1])
                ])
            tmp_inputs = self.add_inceptions(tmp_inputs)
        self.Mixed_6a = inception_b(288)
        tmp_inputs = self.Mixed_6a(tmp_inputs)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        tmp_inputs = self.Mixed_6b(tmp_inputs)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        tmp_inputs = self.Mixed_6c(tmp_inputs)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        tmp_inputs = self.Mixed_6d(tmp_inputs)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        tmp_inputs = self.Mixed_6e(tmp_inputs)

        self.Mixed_7a = inception_d(768)
        tmp_inputs = self.Mixed_7a(tmp_inputs)
        self.Mixed_7b = inception_e(1280)
        tmp_inputs = self.Mixed_7b(tmp_inputs)
        self.Mixed_7c = inception_e(2048)
        tmp_inputs = self.Mixed_7c(tmp_inputs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        tmp_inputs = self.avgpool(tmp_inputs)
        self.dropout = nn.Dropout(p=dropout)
        tmp_inputs = self.dropout(tmp_inputs)
        
        tmp_inputs = torch.flatten(tmp_inputs, 1)
        self.fc = nn.Linear(tmp_inputs.size(1), num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1  # type: ignore
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        if pretrained:
            _inceptionv3 = inception_v3(pretrained=True)
            self.load_state_dict(_inceptionv3.state_dict())
            self.fc = nn.Linear(2048, num_classes)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        if self.add_settings[0] > 0:
            x = self.add_convs(x)
        
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        if self.add_settings[1] > 0:
            x = self.add_inceptions(x)
        if self.image_size > 32:
            # N x 288 x 35 x 35
            x = self.Mixed_6a(x)
            # N x 768 x 17 x 17
            x = self.Mixed_6b(x)
            # N x 768 x 17 x 17
            x = self.Mixed_6c(x)
            # N x 768 x 17 x 17
            x = self.Mixed_6d(x)
            # N x 768 x 17 x 17
            x = self.Mixed_6e(x)
            # N x 768 x 17 x 17
        if self.image_size > 64:
            # N x 768 x 17 x 17
            x = self.Mixed_7a(x)
            # N x 1280 x 8 x 8
            x = self.Mixed_7b(x)
            # N x 2048 x 8 x 8
            x = self.Mixed_7c(x)
        
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]):
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def get_features(self, x, feat_type='relu'):
        outputs = []
        x = self._transform_input(x)
        outputs.append(x.detach().cpu())
        x = self.Conv2d_1a_3x3(x)
        outputs.append(x.detach().cpu())
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        outputs.append(x.detach().cpu())
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        outputs.append(x.detach().cpu())
        # N x 64 x 147 x 147
        x = self.maxpool1(x)

        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        outputs.append(x.detach().cpu())
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        outputs.append(x.detach().cpu())
        if self.add_settings[0] > 0:
            for layer in self.add_convs:
                x = layer(x)
                outputs.append(x.detach().cpu())
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        outputs.append(x.detach().cpu())
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        outputs.append(x.detach().cpu())
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        outputs.append(x.detach().cpu())
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        outputs.append(x.detach().cpu())
        if self.add_settings[1] > 0:
            for layer in self.add_inceptions:
                x = layer(x)
                outputs.append(x.detach().cpu())
        if self.image_size > 32:
            # N x 288 x 35 x 35
            x = self.Mixed_6a(x)
            outputs.append(x.detach().cpu())
            # N x 768 x 17 x 17
            x = self.Mixed_6b(x)
            outputs.append(x.detach().cpu())
            # N x 768 x 17 x 17
            x = self.Mixed_6c(x)
            outputs.append(x.detach().cpu())
            # N x 768 x 17 x 17
            x = self.Mixed_6d(x)
            outputs.append(x.detach().cpu())
            # N x 768 x 17 x 17
            x = self.Mixed_6e(x)
            outputs.append(x.detach().cpu())
            # N x 768 x 17 x 17
        if self.image_size > 64:
            # N x 768 x 17 x 17
            x = self.Mixed_7a(x)
            outputs.append(x.detach().cpu())
            # N x 1280 x 8 x 8
            x = self.Mixed_7b(x)
            outputs.append(x.detach().cpu())
            # N x 2048 x 8 x 8
            x = self.Mixed_7c(x)
            outputs.append(x.detach().cpu())
        x = self.avgpool(x)
        outputs.append(x.detach().cpu())
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 288 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        outputs.append(x.detach().cpu())
        return outputs
    def forward(self, x: Tensor):
        x = self._transform_input(x)
        return self._forward(x)
       

        


class InceptionA(nn.Module):
    def __init__(
        self, in_channels: int, pool_features: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(
        self, in_channels: int, channels_7x7: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

