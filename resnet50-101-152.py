# 2023/11/23

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, downsample=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if downsample:
            self.downsampling = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsampling = None

    def forward(self, X):
        Y = self.bottleneck(X)
        if self.downsampling:
            X = self.downsampling(X)
        Y += X
        return F.relu(Y)


def make_layer(in_channels, out_channels, mid_channels, num_bottleneck, stride):
    layer = list()
    for i in range(num_bottleneck):
        if i == 0:
            layer.append(ResidualBlock(in_channels, out_channels, mid_channels, downsample=True, stride=stride))
        else:
            layer.append(ResidualBlock(out_channels, out_channels, mid_channels))
    return layer


class Resnet(nn.Module):
    def __init__(self, block, num_classes=1000):
        super(Resnet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.AvgPool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2048, num_classes)
        self.layer1 = nn.Sequential(*make_layer(64, 256, 64, block[0], 1))
        self.layer2 = nn.Sequential(*make_layer(256, 512, 128, block[1], 2))
        self.layer3 = nn.Sequential(*make_layer(512, 1024, 256, block[2], 2))
        self.layer4 = nn.Sequential(*make_layer(1024, 2048, 512, block[3], 2))

    def forward(self, X):
        Y = self.pre(X)
        Y = self.layer1(Y)
        Y = self.layer2(Y)
        Y = self.layer3(Y)
        Y = self.layer4(Y)
        Y = self.AvgPool(Y)
        Y = Y.view(Y.size(0), -1)
        Y = self.fc(Y)
        return Y


def ResNet50():
    return Resnet([3, 4, 6, 3])


def ResNet101():
    return Resnet([3, 4, 23, 3])


def ResNet152():
    return Resnet([3, 8, 36, 3])



X = torch.rand(size=(1, 3, 224, 224))
# net = ResNet50()
# net = ResNet101()
net = ResNet152()
print(net(X))
