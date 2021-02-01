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
from SimpleModel import *
from utils import *
from GenerateData import *
import sys


if __name__ == "__main__":
    root = 'E:/ljq/data'
    train_dataset,  train_dataloader = generate_data(root, 'MNIST', train=True, shuffle=True, shuffle_label=False)
    test_dataset, test_dataloader = generate_data(root, 'MNIST', train=False, shuffle_label=False)

    batch_size = 128
    root = './log/cw/model3-1e1-0'
    adversarial_dataset, adversarial_dataloader = get_adversarial_data(root, batch_size=batch_size, shuffle=True)

    model = Model3()
    lr = 1e-2
    epoch = 20
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(model)
    fit(model, adversarial_dataloader, test_dataloader, loss_func, optimizer, epoch, device,
        save_path='./log', name='model3-cw-1e1-0')

    # save_model(model, log_path, 'conv_model')

    # imagenet = torchvision.datasets.ImageNet(root=root, download=True, split='train')

    # model = ConvModel2()
    # batch_size = 64
    # lr = 1e-4
    # epoch = 50
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # loss_func = nn.CrossEntropyLoss()
    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    #
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # root = 'E:/ljq/data'
    # train_dataset = datasets.CIFAR10(root=root, train=True, download=False, transform=transform)
    # test_dataset = datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #
    # fit(model, train_dataloader, test_dataloader, loss_func, optimizer, epoch, device)
