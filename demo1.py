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
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    root = 'E:/ljq/data'
    dataclass = CIFAR(root)
    batch_size = 128
    train_dataset, train_dataloader = \
        dataclass.get_dataloader(train=True, batch_size=batch_size, shuffle=True, num_worker=8)
    test_dataset, test_dataloader = \
        dataclass.get_dataloader(train=False, batch_size=batch_size, shuffle=False, num_worker=8)

    root = './log/cifar/pgd/resnet-0.03'
    adversarial_dataclass = CIFAR(root, load=True)
    adversarial_dataset, adversarial_dataloader = \
        adversarial_dataclass.get_dataloader(batch_size=128,
                                             shuffle=True, num_worker=8, adversarial=True, pin_memory=True)

    model = vgg_cifar()
    load_model(model, './log/cifar/model', 'vgg')
    model = add_normal_layer(model, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
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

    t0 = time.time()
    loss_log, acc_log = evaluate(model, test_dataloader, loss_func, device, logger=None)
    print('loss:%.3f, accuracy:%.3f' % (loss_log.mean().item(), acc_log.mean().item()))
    t1 = time.time()
    print(t1 - t0)

    # fit(model, adversarial_dataloader, test_dataloader, loss_func, optimizer, epoch, device,
    #     scheduler=scheduler,
    #     save_path='./log/mnist/model',
    #     name='resnet-vgg-0.3')

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
