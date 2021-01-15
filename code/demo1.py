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
from Simple_model import SM
from utils import *

root='E:\ljq\data'
# transform = transforms.Compose([transforms.ToTensor(),])
# train_dataset = datasets.MNIST(root='E:\ljq\data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='E:\ljq\data', train=False, download=True, transform=transform)
# print(len(train_dataset))
# print(len((test_dataset)))
#
# batch_size = 64
#
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# print(next(iter(train_dataloader))[0].shape)
# print(len(train_dataloader))
# print(len(test_dataloader))
#
#
# if __name__ == "__main__":
#     model = SM()
#     lr = 1e-2
#     epoch = 10
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     loss_func = nn.CrossEntropyLoss()
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     epoch = 5
#     log_path = '.\\log'
#
#     # train(model, train_dataloader, loss_func, optimizer, device)
#     #
#     # evaluate(model, test_dataloader, loss_func, device)
#
#     fit(model, train_dataloader, test_dataloader, loss_func, optimizer, epoch, device)
#
#     save_model(model, log_path, 'simple_model')

imagenet = torchvision.datasets.ImageNet(root=root, download=True, split='train')



























