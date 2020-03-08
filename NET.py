import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, 5, 1, 2)
        self.bn0 = nn.BatchNorm2d(32)
        self.res_block = self.res_layers(BasicBlock, 32, 32, 8, stride=1)
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)

    def res_layers(self, block, in_planes, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for step in strides:
            layers.append(block(in_planes, planes, step))
        return nn.Sequential(*layers)

    def forward(self, imgL, imgR):
        imgL = F.relu(self.bn0(self.conv0(imgL)))
        imgR = F.relu(self.bn0(self.conv0(imgR)))

        imgL_block = self.res_block(imgL)
        imgR_block = self.res_block(imgR)

        imgL = self.conv1(imgL_block)
        imgR = self.conv1(imgR_block)

        return imgL, imgR
