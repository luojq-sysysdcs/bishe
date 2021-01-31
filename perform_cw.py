#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/9 22:21
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : perform_cw.py
# @Software: PyCharm

import torch
import torchvision
from SimpleModel import *
from utils import *
from FGSM import FGSM
from CW import CW
import numpy as np
from GenerateData import *
from matplotlib import pyplot as plt


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    root = 'E:/ljq/data'
    batch_size = 2
    train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=batch_size, shuffle=False)

    model = Model3()
    log_path = './log'
    load_model(model, log_path, 'model3')
    print(model)

    # model = SM()
    # log_path = '.\\log'
    # load_model(model, log_path, 'simple_model')
    # print(model)

    adversary = CW(model, c=1e1, kappa=1e-10, steps=500, lr=1e-1)

    save_root = './log/cw_image'
    if not os.path.exists(save_root):
        os.mkdir(save_root)

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
            img = torchvision.utils.make_grid(
                torch.cat([images, adv_images], dim=0), nrow=len(labels), padding=2, pad_value=1)
            show(img)
        break








