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
from deepfool import DeepFool

transform = transforms.Compose([transforms.ToTensor(),])
train_dataset = datasets.MNIST(root='E:\ljq\data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='E:\ljq\data', train=False, download=True, transform=transform)
print(len(train_dataset))
print(len((test_dataset)))

batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print(next(iter(train_dataloader))[0].shape)
print(len(train_dataloader))
print(len(test_dataloader))

model = SM()
log_path = '.\\log'
load_model(model, log_path, 'simple_model')
print(model)

adversary = DeepFool(model)
# adversary = FGSM(model, eps=0.20)

idx = 30
image, label = train_dataset[idx][0].unsqueeze(0), torch.tensor(train_dataset[idx][1]).unsqueeze(0)
output = model(image)
print(output)
pred = torch.max(output, 1)[1].squeeze()
print('true label:%d, predict label:%d' % (label, pred))

adv_image = adversary.forward(image,label)
after_label = torch.max(model(adv_image), 1)[1].squeeze()
print('true label:%d, after attacking predict label:%d' % (label, after_label))

adv_image = adv_image.squeeze(0).permute(1, 2, 0).numpy()
image = image.squeeze(0).permute(1, 2, 0).numpy()
print(adv_image.shape)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.imshow(image)
ax1.set_title('label:%d' % label)
ax2.imshow(adv_image)
ax2.set_title('label:%d' % after_label)
plt.show()
