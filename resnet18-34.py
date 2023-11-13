# 2023/11/23

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.BasicBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
        Y = self.BasicBlock(X)
        if self.downsampling:
            X = self.downsampling(X)
        Y += X
        return F.relu(Y)


def make_layer(in_channels, out_channels, num_Residuals, first_block=False):
    layer = list()
    for i in range(num_Residuals):
        if i == 0 and not first_block:
            layer.append(ResidualBlock(in_channels, out_channels, downsample=True, stride=2))
        else:
            layer.append(ResidualBlock(out_channels, out_channels))
    return layer


class Resnet(nn.Module):
    def __init__(self, block):
        super(Resnet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(*make_layer(64, 64, block[0], first_block=True))
        self.layer2 = nn.Sequential(*make_layer(64, 128, block[1]))
        self.layer3 = nn.Sequential(*make_layer(128, 256, block[2]))
        self.layer4 = nn.Sequential(*make_layer(256, 512, block[3]))
        self.AvgPool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, 1000)

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


def ResNet18():
    return Resnet([2, 2, 2, 2])


def ResNet34():
    return Resnet([3, 4, 6, 3])


X = torch.rand(size=(1, 3, 224, 224))
# net = ResNet18()
net = ResNet34()
print(net(X))
