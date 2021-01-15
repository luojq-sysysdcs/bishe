#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/7 19:47
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : utils.py
# @Software: PyCharm

import torch
import os

def train(model, dataloader, loss_func, optimizer, device):
    model = model.train().to(device)

    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = loss_func(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx+ 1) % (len(dataloader) // 10) == 0:
            _, pred = torch.max(output, 1)
            acc = torch.sum(pred == labels) / len(images)
            print("idx:%3d, loss: %.3f, accuracy:%.3f" % (idx, loss.item(), acc))

def evaluate(model, dataloader, loss_func, device, logger=None):
    model = model.eval().to(device)
    loss_log = torch.zeros(len(dataloader))
    acc_log = torch.zeros(len(dataloader))

    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        _, pred = torch.max(output, 1)
        acc_log[idx] = torch.sum(pred == labels) / len(images)
        loss = loss_func(output, labels)
        loss_log[idx] = loss

    return loss_log, acc_log


def fit(model, train_dataloader, test_dataloader, loss_func, optimizer, epochs, device, logger=None):
    for epoch in range(epochs):
        print('epoch:%3d' % epoch)
        train(model, train_dataloader, loss_func, optimizer, device)
        print()

        if (epoch + 1) % (epochs // 5) == 0:
            print('evaluating...')
            loss_log, acc_log = evaluate(model, test_dataloader, loss_func,device)
            print('loss:%.3f, accuracy:%.3f' % (loss_log.mean().item(), acc_log.mean().item()))
            print()

def save_model(model, path, name):
    if not os.path.exists(path):
        print('Path not exists! Creating')
        os.mkdir(path)
    torch.save(model.state_dict(), os.path.join(path, name))
    print('model saved successfully!')

def load_model(model, path, name):
    if not os.path.exists(path):
        raise ValueError('Path not exists!')
    state_dict = torch.load(os.path.join(path, name), map_location = torch.device('cpu'))
    model.load_state_dict(state_dict)
    print('model loaded successfully!')








































