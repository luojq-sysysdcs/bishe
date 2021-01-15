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
            nn.ReLU(inplace=True),
            nn.Conv2d(fm1, fm2, (3, 3), padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fm2, fm3, (3, 3), padding=1, stride=2),
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

if __name__ == '__main__':
    model = ConvModel()
    summary(model, (1, 28, 28))