import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
import torch.utils.model_zoo as model_zoo
import torchvision.models.resnet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,downsample=None) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
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
        self.conv3 = conv1x1(channel, channel * self.expansion)
        self.bn3 = nn.BatchNorm2d(channel * self.expansion)
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
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf=64) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       nn.ReLU(),
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4)
        self.cls = self.linear

        # self.encoder = nn.Sequential(
        #     nn.Linear(nf * 8 * block.expansion, nf * 8 * block.expansion),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(nf * 8 * block.expansion, 512)
        # )

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        else:
            downsample=None
        layers.append(block(self.in_planes, planes, strides[0], downsample))
        self.in_planes = planes * block.expansion
        for i in range(1,len(strides)):
            layers.append(block(self.in_planes, planes, strides[i]))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    # def encoders(self, x: torch.Tensor) -> torch.Tensor:
    #     out = self._features(x)
    #     out = avg_pool2d(out, out.shape[2])
    #     feat = out.view(out.size(0), -1)
    #     feat = self.encoder(feat)
    #     return feat

    def classifier(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cls(x)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.conv1(x)))  # 64, 32, 32
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        out = self.layer1(out)  # -> 64, 32, 32
        out = self.layer2(out)  # -> 128, 16, 16
        out = self.layer3(out)  # -> 256, 8, 8
        out = self.layer4(out)  # -> 512, 4, 4
        out = avg_pool2d(out, out.shape[2])  # -> 512, 1, 1
        feature = out.view(out.size(0), -1)  # 512
        out = self.cls(feature)
        return out


def resnet10(cfg) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], cfg.DATASET.n_classes)
    return model


def resnet12(cfg) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    model = ResNet(BasicBlock, [2, 1, 1, 1], cfg.DATASET.n_classes)
    return model


def resnet18(cfg) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], cfg.DATASET.n_classes)

    return model


def resnet20(cfg) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    model = ResNet(BasicBlock, [1, 3, 3, 3], cfg.DATASET.n_classes)
    return model


def resnet34(cfg) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], cfg.DATASET.n_classes)
    return model

def resnet50(cfg) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], cfg.DATASET.n_classes)
    return model
