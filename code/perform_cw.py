#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/9 22:21
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : perform_cw.py
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
from FGSM import FGSM
from CW import CW
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),])
train_dataset = datasets.MNIST(root='E:\ljq\data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='E:\ljq\data', train=False, download=True, transform=transform)
print(len(train_dataset))
print(len((test_dataset)))

batch_size = 2

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print(next(iter(train_dataloader))[0].shape)
print(len(train_dataloader))
print(len(test_dataloader))

model = SM()
log_path = '.\\log'
load_model(model, log_path, 'simple_model')
print(model)

adversary = CW(model)


save_root = '.\\log\\cw_image'
if not os.path.exists(save_root):
    os.mkdir(save_root)
name = 0
for idx, (images, labels) in enumerate(test_dataloader):
    # adv_images = adversary.forward(images)
    adv_labels = [[i for i in range(10)] for _ in range(batch_size)]
    for i, label in enumerate(labels):
        del adv_labels[i][label]
    print(labels)
    print(adv_labels)
    adv_labels = torch.tensor(adv_labels, dtype=torch.long)
    for i in range(9):
        adv_images, success = adversary.forward(images, adv_labels[:, i])
        for j, s in enumerate(success):
            if s:
                adv_image = adv_images[j].permute(1, 2, 0).numpy()
                plt.imshow(adv_image, cmap='binary')
                plt.savefig(os.path.join(save_root, str(name)))
                name+=1
    break

# idx = 0
# image, label = train_dataset[idx][0].unsqueeze(0), torch.tensor(train_dataset[idx][1]).unsqueeze(0)
# output = model(image)
# print(output)
# pred = torch.max(output, 1)[1].squeeze()
# print('true label:%d, predict label:%d' % (label, pred))
# print('-'*40)
# for i in range(10):
#     adv_label = torch.tensor([i], dtype=torch.long)
#     print(adv_label)
#     adv_image = adversary.forward(image,adv_label)
#     output = model(adv_image)
#     print(output)
#     after_label = torch.max(output, 1)[1].squeeze()
#     print('true label:%d, after attacking predict label:%d' % (label, after_label))


# adv_label = torch.tensor([0], dtype=torch.long)
# print(adv_label)
# adv_image = adversary.forward(image,adv_label)
# after_label = torch.max(model(adv_image), 1)[1].squeeze()
# print('true label:%d, after attacking predict label:%d' % (label, after_label))
#
# adv_image = adv_image.squeeze(0).permute(1, 2, 0).numpy()
# image = image.squeeze(0).permute(1, 2, 0).numpy()
# diff = (adv_image - image)
# diff = np.abs(diff)
# # diff = diff - np.min(diff)
# # diff /= np.max(diff)
# print(adv_image.shape)
# ax1 = plt.subplot(131)
# ax2 = plt.subplot(132)
# ax3 = plt.subplot(133)
# ax1.imshow(image, cmap='binary')
# ax1.set_title('label:%d' % label)
# ax2.imshow(adv_image, cmap='binary')
# ax2.set_title('label:%d' % after_label)
# ax3.imshow(diff, cmap='binary')
# plt.show()






