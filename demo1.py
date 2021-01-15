#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/7 19:01
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : demo1.py
# @Software: PyCharm

import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from Simple_model import *
from utils import *
from GenerateData import *

root = 'E:\\ljq\\data'
train_dataset,  train_dataloader = generate_data(root, 'MNIST', train=True)
test_dataset, test_dataloader = generate_data(root, 'MNIST', train=False)


if __name__ == "__main__":
    model = SM()
    # model = ConvModel()
    lr = 1e-2
    epoch = 10
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch = 5
    log_path = '.\\log'

    # train(model, train_dataloader, loss_func, optimizer, device)
    #
    # evaluate(model, test_dataloader, loss_func, device)

    fit(model, train_dataloader, test_dataloader, loss_func, optimizer, epoch, device)

    # save_model(model, log_path, 'conv_model')

# imagenet = torchvision.datasets.ImageNet(root=root, download=True, split='train')



























