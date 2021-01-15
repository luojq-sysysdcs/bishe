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






