from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import random
import numpy as np
import numpy.random as npr

__all__ = ['ResNet', 'resnet50', 'resnet18', 'resnet34', 'resnet101', 'resnet152']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


def conv1x1(input_channel, output_channel, bias=False):
    return nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=bias)


def conv3x3(in_channel, out_channel, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias=bias)


def random_sample(prob, sampling_num):
    batch_size, channels, h, w = prob.shape
    return torch.multinomial((prob.view(batch_size * channels, -1) + 1e-8), sampling_num, replacement=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channel, output_channel, stride=1, downsample=None, track_running_stats=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(input_channel, output_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(output_channel, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(output_channel, output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel, track_running_stats=track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        residual = input
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_channel, channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(input_channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(channel, channel, stride=stride)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = conv1x1(channel, channel * 4)
        self.bn3 = nn.BatchNorm2d(channel * 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        residual = input  # skip path
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual

        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, backbone=False, feature_norm=False):
        self.feature_norm = feature_norm
        self.backbone = backbone
        super(ResNet, self).__init__()
        self.input_channel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, padding=0, stride=1)
        self.cls = nn.Linear(512 * block.expansion, num_classes)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4, self.cls]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.input_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.input_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = []
        layers.append(block(self.input_channel, channel, stride=stride, downsample=downsample))
        self.input_channel = channel * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.input_channel, channel))

        return nn.Sequential(*layers)

    def features(self, x):
        if x.size(-1) == 224:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
        elif x.size(-1) == 56:
            x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        feature_out = x
        return feature_out

    def forward(self, x):
        feature_out = self.features(x)
        x = self.cls(feature_out)
        return x

    def classifier(self, x):
        out = self.cls(x)
        return out


def resnet18_pretrained(cfg):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=cfg.DATASET.n_classes)
    if cfg.DATASET.pretrained:
        print('load success')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet50_pretrained(cfg):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=cfg.DATASET.n_classes)
    if cfg.DATASET.pretrained:
        print('load success')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
