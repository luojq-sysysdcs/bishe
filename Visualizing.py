#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/14 10:56
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : Visualizing.py
# @Software: PyCharm

import torch
from GenerateData import *
from visualization.GradCam import *
from Simple_model import *
from utils import load_model
from matplotlib import pyplot as plt
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

root = 'E:\ljq\data'
train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=1, shuffle=False)

model = ConvModel()
log_path = '.\\log'
load_model(model, log_path, 'conv_model')
print(model)


activation_layer = '3'
gradcam = GradCam(model, model.feature, [activation_layer])

root = os.path.join(os.getcwd(), 'log', 'gradcam_results', 'mnist', 'activation_layer' + activation_layer)
if not os.path.exists(root):
    os.makedirs(root)
for idx, (image, label) in enumerate(train_dataloader):
    if idx == 20:
        break
    if torch.max(model(image), dim=1)[1].item() != label:
        continue
    activation_map = gradcam(image)
    plt.imshow(image.squeeze().numpy(), cmap='binary', interpolation='none')
    if not os.path.exists(os.path.join(root, str(idx))):
        os.makedirs(os.path.join(root, str(idx)))
    plt.savefig(os.path.join(root, str(idx), 'original.jpg'))
    plt.imshow(activation_map, cmap='binary')
    plt.savefig(os.path.join(root, str(idx), 'gradcam.jpg'))
    np.save(os.path.join(root, str(idx), 'gradcam.npy'), activation_map)


















