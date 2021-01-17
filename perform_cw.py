#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/9 22:21
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : perform_cw.py
# @Software: PyCharm

import torch
import torchvision
from Simple_model import *
from utils import *
from FGSM import FGSM
from CW import CW
import numpy as np
from GenerateData import *

root = 'E:\ljq\data'
batch_size = 2
train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=batch_size, shuffle=False)

model = ConvModel()
log_path = '.\\log'
load_model(model, log_path, 'conv_model')
print(model)

# model = SM()
# log_path = '.\\log'
# load_model(model, log_path, 'simple_model')
# print(model)

adversary = CW(model)


save_root = '.\\log\\cw_image'
if not os.path.exists(save_root):
    os.mkdir(save_root)

best_L2 = torch.ones(batch_size) * 1e10
best_adv_images = torch.empty(next(iter(train_dataloader))[0].shape)

for idx, (images, labels) in enumerate(train_dataloader):
    # adv_images = adversary.forward(images)
    adv_labels = [[i for i in range(10)] for _ in range(batch_size)]
    for i, label in enumerate(labels):
        del adv_labels[i][label]
    print(labels)
    print(adv_labels)
    adv_labels = torch.tensor(adv_labels, dtype=torch.long)
    for i in range(2):
        adv_images, success, L2 = adversary.forward(images, adv_labels[:, i])
        print(adv_images.shape)

        mask = success.float() + (best_L2 > L2)
        mask = mask.view([-1] + [1]*(len(images.shape)-1))
        best_adv_images = mask * adv_images + (1 - mask) * best_adv_images

    torchvision.utils.make_grid(best_adv_images)
    break








