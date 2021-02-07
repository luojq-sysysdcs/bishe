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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2'
    root = 'E:/ljq/data'
    dataclass = CIFAR(root)
    batch_size = 128
    train_dataset, train_dataloader = \
        dataclass.get_dataloader(train=True, batch_size=batch_size, shuffle=True, num_worker=8, pin_memory=True)
    test_dataset, test_dataloader = \
        dataclass.get_dataloader(train=False, batch_size=batch_size, shuffle=False, num_worker=8)

    model = vgg_cifar()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # load_model(model, './log/cifar/model', 'resnet')
    lr = 1e-4
    epoch = 50
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = None #optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=epoch//5)
    loss_func = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(model)

    # for i in range(10):
    #     acc = cal_acc(model, test_dataloader, device, label=i)
    #     print('accuracy: %f' % acc)

    # loss_log, acc_log = evaluate(model, adversarial_dataloader, loss_func, device, logger=None)
    # print('loss:%.3f, accuracy:%.3f' % (loss_log.mean().item(), acc_log.mean().item()))

    fit(model, train_dataloader, test_dataloader, loss_func, optimizer, epoch, device,
        scheduler=scheduler,
        save_path='./log/cifar/model',
        name='vgg-1')

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
