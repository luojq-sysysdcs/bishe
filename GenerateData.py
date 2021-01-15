#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/14 10:39
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : GenerateData.py
# @Software: PyCharm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def generate_data(root, name, train=True, transform=None, batch_size=None, shuffle=False):
    if name == 'MNIST':
        loader = datasets.MNIST
    else:
        raise NotImplementedError
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(),])
    if batch_size is None:
        batch_size = 64
    if train:
        train_dataset = loader(root=root, train=True, download=True, transform=transform)
        print('num of data:', len(train_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        print('num of batch:', len(train_dataloader))
        return train_dataset, train_dataloader

    else:
        test_dataset = loader(root=root, train=False, download=True, transform=transform)
        print('num of data:', len((test_dataset)))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        print('num of batch', len(test_dataloader))
        return test_dataset, test_dataloader





