# This implementation is based on the DenseNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import math
import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        def conv3x3(inplanes, planes, stride=1):
            if stride == 2:
                return nn.Conv2d(inplanes, planes, kernel_size=3, padding=0, stride=stride, bias=False)
            else:
                return nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)

        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv2 = conv3x3(planes, planes)
        self.stride = stride

    def forward(self, x):
        residual = x
        # TODO: fix the bug of original Stochatic depth
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        if self.stride > 1:
            residual = torch.nn.functional.pad(residual, (0, 2, 0, 2), value=0.)
        residual = self.conv1(residual)
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)
        if self.downsample is not None:
            x = self.downsample(x)

        x = x + residual
        return x


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.downsample = nn.Conv2d(nIn, nOut, kernel_size=1, padding=0, stride=stride, bias=False)

    def forward(self, x):
        return self.downsample(x)


class ResNet(nn.Module):
    '''Small ResNet for CIFAR & SVHN '''

    def __init__(self, depth=32, width_multiplier=1, block=BasicBlock, num_classes=10):
        assert ((depth - 2) % 6 == 0 or (depth - 4) % 6 == 0), 'depth should be one of 6N+2 or 6N+4'
        super(ResNet, self).__init__()
        n = (depth - 2) // 6
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16 * width_multiplier, n)
        self.layer2 = self._make_layer(block, 32 * width_multiplier, n, stride=2)
        self.layer3 = self._make_layer(block, 64 * width_multiplier, n, stride=2)
        self.final_bn = nn.BatchNorm2d(self.inplanes)
        self.final_relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * width_multiplier, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = DownsampleB(self.inplanes, planes, stride)
        layers = [block(self.inplanes, planes, stride, downsample=downsample)]

        self.inplanes = planes
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, stride=1))

        return nn.Sequential(*layers)

    @property
    def classifier(self):
        return self.fc

    @property
    def num_classes(self):
        return self.fc.weight.size(-2)

    @property
    def num_features(self):
        return self.fc.weight.size(-1)

    def extract_features(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return x

    def classifier(self, x):
        return self.fc(x)

    def forward(self, x):
        features = self.extract_features(x)
        res = self.fc(features)
        return res


def wideresnet28_10(num_classes=10):
    return ResNet(depth=28, width_multiplier=10, num_classes=num_classes)
