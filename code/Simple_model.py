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


if __name__ == '__main__':
    model = SM()
    summary(model, (28, 28))