__author__ = 'Qi'
# Created by on 1/10/22.

import torch
import torch.nn as nn
from torch import Tensor


__all__ = ['ResNet50FC', 'ResNet50FCLB']


class ResNet50FC(nn.Module):
    def __init__(self, num_classes= 1000, data = None, pretrained = True):
        super(ResNet50FC, self).__init__()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
       out = self.fc(x)

       return out, None



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )



class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: bool = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: bool = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet50FCLB(nn.Module):

     def __init__(self, planes = 512, block = Bottleneck, num_classes = 1000, data = None, pretrained = True):
         super(ResNet50FCLB, self).__init__()
         self.inplanes = 64
         self.dilation = 1
         self.groups = 1
         self.base_width = 64

         self.inplanes = planes * block.expansion
         last_block_layers = []
         last_block_layers.append(
             block(
                 self.inplanes,
                 planes,
                 groups=self.groups,
                 base_width=self.base_width,
                 dilation=self.dilation,
                 norm_layer=None,
                 )
             )
         self.lb_layers = nn.Sequential(*last_block_layers)
         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
         self.fc = nn.Linear(512 * block.expansion, num_classes)


     def forward(self, x):
         x = self.lb_layers(x)
         x = self.avgpool(x)
         x = self.fc(x)

         return x

