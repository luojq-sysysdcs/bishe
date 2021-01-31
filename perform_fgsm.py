#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/7 21:15
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : perform_fgsm.py
# @Software: PyCharm


import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from SimpleModel import *
from utils import *
from FGSM import FGSM
from PGD import PGD
from GenerateData import *
import os
from PIL import Image
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

root = 'E:\ljq\data'
batch_size = 1
train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=batch_size, shuffle=False)

# model = SM()
# model = ConvModel()
# log_path = '.\\log'
# load_model(model, log_path, 'conv_model')
# model = model.eval()
# print(model)

model = Model3()
log_path = './log'
load_model(model, log_path, 'model3')
model = model.eval()
print(model)

# adversary = FGSM(model, eps=0.15)
adversary = PGD(model, eps=0.3, alpha=1 / 255, steps=100, random_start=False)

root = './log/PGD-model3-0.3'
if not os.path.exists(root):
    os.makedirs(root)

labels = []
success = 0
count = 1000
for idx, (image, label) in enumerate(train_dataloader):
    if idx < 62:
        continue
    if idx == count:
        break
    labels.append(label.item())
    if not os.path.exists(os.path.join(root, str(idx))):
        os.makedirs(os.path.join(root, str(idx)))
    # if torch.max(model(image), dim=1)[1].item() != label:
    #     continue
    for target_label in range(10):
        if target_label == label:
            continue
        adv_label = torch.tensor([target_label], dtype=torch.long)
        adv_image = adversary.forward(image,adv_label)
        t = (adv_image.detach().clone()*255).to(torch.uint8) / 255
        after_label = torch.max(model(t), 1)[1].squeeze()
        if after_label.item() != target_label:
            continue
        success += 1
        t = np.uint8((t.squeeze() * 255).numpy())
        adv_image = Image.fromarray(t)
        adv_image = adv_image.convert('L')
        adv_image.save(os.path.join(root, str(idx), str(target_label)+'.jpg'))
    if idx % 10 == 0:
        print('success rate: %f (%d / %d)' % (success / ((idx + 1) * 9), success, (idx + 1) * 9))
print('-'*20)
print(labels)
print('success rate: %f (%d / %d)' % (success / (count * 9), success, count * 9))
np.savetxt(os.path.join(root, 'labels.txt'), labels, fmt="%d")

# -----------------------------------
# idx = 80
# image, label = train_dataset[idx][0].unsqueeze(0), torch.tensor(train_dataset[idx][1]).unsqueeze(0)
# output = model(image)
# pred = torch.max(output, 1)[1].squeeze()
# print('true label:%d, predict label:%d' % (label, pred))
#
# for i in range(10):
#     adv_label = torch.tensor([i], dtype=torch.long)
#     print(adv_label)
#     adv_image = adversary.forward(image,adv_label)
#     after_label = torch.max(model(adv_image), 1)[1].squeeze()
#     print('true label:%d, after attacking predict label:%d' % (label, after_label))
#     plt.imshow(adv_image.squeeze().detach().cpu().numpy())
#     plt.show()

# adv_image = adv_image.squeeze(0).permute(1, 2, 0).numpy()
# image = image.squeeze(0).permute(1, 2, 0).numpy()
# print(adv_image.shape)
# ax1 = plt.subplot(121)
# ax2 = plt.subplot(122)
# ax1.imshow(image)
# ax1.set_title('label:5')
# ax2.imshow(adv_image)
# ax2.set_title('label:3')
# plt.show()


























