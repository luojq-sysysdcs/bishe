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
import sys
print(sys.platform)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if 'win' in sys.platform:
    root = '.\\log\\cw_image\\numpy'
else:
    root = './log/cw_image/numpy'

class FeatureExtractor():
    def __init__(self):
        self.features = []
        self.x = None
        self.target_modules = None

    def extractor(self, model):
        for name, module in model._modules.items():
            if 'linear' in name.lower():
                self.x = self.x.view(self.x.shape[0], -1)
            if name in self.target_modules:
                self.features.append(self.x)
                self.x = module(self.x)
            else:
                if len(module._modules) == 0:
                    self.x = module(self.x)
                else:
                    self.extractor(module)

    def __call__(self, model, x, target_modules):
        self.features = []
        self.model = model
        self.x = x
        self.target_modules =target_modules
        self.extractor(model)
        return self.features


# def feature_extractor(model, x, target_modules):
#     features = []
#     for name, module in model._modules.items():
#         if name.lower() in target_modules:
#             features.append(x)
#             x = module(x)

def readNumpyFile(root, size=(28, 28)):
    files = []
    for path, _, fnames in os.walk(root):
        for fname in sorted(fnames, key=lambda x:int(x.split('.')[0])):
            full_path = os.path.join(path, fname)
            file = np.load(full_path).reshape(size)
            files.append(file)
        files = np.stack(files, axis=0)
        return files

fake_images = readNumpyFile(root)
fake_images = torch.from_numpy(fake_images)
print(fake_images.shape)


# if 'win' in sys.platform:
#     root = 'E:\ljq\data'
# else:
#     root = './data'
# train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=1, shuffle=False)
#
model = ConvModel()
if 'win' in sys.platform:
    log_path = '.\\log'
else:
    log_path = './log'
load_model(model, log_path, 'conv_model')
print(model)

fx = FeatureExtractor()
features = fx(model, fake_images.unsqueeze(1), ['linear'])
print(features[0][0])
# activation_layer = '3'
# gradcam = GradCam(model, model.feature, [activation_layer])
#
# root = os.path.join(os.getcwd(), 'log', 'gradcam_results', 'mnist', 'activation_layer' + activation_layer)
# if not os.path.exists(root):
#     os.makedirs(root)
# for idx, (image, label) in enumerate(train_dataloader):
#     if idx == 20:
#         break
#     if torch.max(model(image), dim=1)[1].item() != label:
#         continue
#     activation_map = gradcam(image)
#     plt.imshow(image.squeeze().numpy(), cmap='binary', interpolation='none')
#     if not os.path.exists(os.path.join(root, str(idx))):
#         os.makedirs(os.path.join(root, str(idx)))
#     plt.savefig(os.path.join(root, str(idx), 'original.jpg'))
#     plt.imshow(activation_map, cmap='binary')
#     plt.savefig(os.path.join(root, str(idx), 'gradcam.jpg'))
#     np.save(os.path.join(root, str(idx), 'gradcam.npy'), activation_map)

# for image in fake_images:
#     image = image.unsqueeze(0).unsqueeze(0)
#     activation_map_for_fake_images = gradcam(image)
#     plt.imshow(activation_map_for_fake_images, cmap='binary')
#     plt.show()

















