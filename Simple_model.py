#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/7 19:39
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : Simple_model.py
# @Software: PyCharm

import torch
from torch import nn
from torchsummary import summary
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import random
from utils import *
# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)

class SM(nn.Module):
    def __init__(self, size=(28, 28), num_classes=10):
        super(SM, self).__init__()
        self.size = size
        self.num_classes = num_classes

        self.linear = nn.Sequential(
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = x.flatten(1)
        return self.linear(x)


class ConvModel(nn.Module):
    def __init__(self, size=(28, 28), num_classes=10):
        super(ConvModel, self).__init__()
        self.size = size
        self.num_classes = num_classes

        fm1 = 10
        fm2 = 50
        fm3 = 100

        self.feature = nn.Sequential(
            nn.Conv2d(1, fm1, (3, 3), padding=1, stride=1),
            nn.BatchNorm2d(fm1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fm1, fm2, (3, 3), padding=1, stride=2),
            nn.BatchNorm2d(fm2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fm2, fm3, (3, 3), padding=1, stride=2),
            nn.BatchNorm2d(fm3),
            nn.ReLU(inplace=True),
#             nn.AdaptiveMaxPool2d((1, 1)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.linear = nn.Linear(fm3, 10)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class ConvModel2(nn.Module):
    def __init__(self, size=(28, 28), num_classes=10):
        super(ConvModel2, self).__init__()
        self.size = size
        self.num_classes = num_classes

        fm1 = 64
        fm2 = 128
        fm3 = 256

        self.feature = nn.Sequential(
            nn.Conv2d(3, fm1, (3, 3), padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(fm1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fm1, fm2, (3, 3), padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(fm2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fm2, fm3, (3, 3), padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(fm3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(fm3, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.feature( x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (3, 3), stride=1, padding=1)
        resnet18 = models.resnet18(pretrained=True)
        self.extract = nn.Sequential(*list(resnet18.children())[1:-4])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.extract(x)
        x = self.avgpool(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x


class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (3, 3), stride=1, padding=1)
        vgg16 = models.vgg16(pretrained=True)
        self.extract = nn.Sequential(*list(list(vgg16.children())[0].children())[1:-15])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.extract(x)
        x = self.avgpool(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = Model4()
    print(model)
    summary(model, (1, 28, 28))

















