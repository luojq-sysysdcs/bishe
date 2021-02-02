#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/7 19:01
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : demo1.py
# @Software: PyCharm

from GenerateData import *
from models.SimpleModel import *
import torch
from torch import optim

if __name__ == "__main__":
    root = 'E:/ljq/data'
    batch_size = 2
    train_dataset,  train_dataloader = generate_data(root, 'MNIST', train=True, shuffle=True, shuffle_label=False)
    test_dataset, test_dataloader = generate_data(root, 'MNIST', train=False, shuffle_label=False)

    # batch_size = 128
    # # root = './log/cw/model3-1e1-0'
    # root = './log/pgd/model3-0.3'
    # adversarial_dataset, adversarial_dataloader = get_adversarial_data(root, batch_size=batch_size, shuffle=True)

    model = resnet_mnist()
    # load_model(model, './log', 'model4')
    lr = 1e-2
    epoch = 20
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(model)
    fit(model, train_dataloader, test_dataloader, loss_func, optimizer, epoch, device,
        save_path='./log/mnist/model', name='resnet')

    # model = Model3()
    # log_path = './log'
    # load_model(model, log_path, 'model3-cw-1e1-0')
    # model = model.eval()
    # model = model.to(device)
    # loss_log, acc_log = evaluate(model, test_dataloader, loss_func, device, logger=None)
    # print('loss:%.3f, accuracy:%.3f' % (loss_log.mean().item(), acc_log.mean().item()))
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
