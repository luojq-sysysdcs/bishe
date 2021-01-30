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
        self.target_modules = target_modules
        self.extractor(model)
        return self.features


def readNumpyFile(root, size=(28, 28)):
    files = []
    for path, _, fnames in os.walk(root):
        fnames = [f for f in fnames if 'npy' in f and 0 <= int(f.split('.')[0])]
        indexs = [int(f.split('.')[0]) for f in fnames if 'npy' in f and 0 <= int(f.split('.')[0])]
        for fname in sorted(fnames, key=lambda x:int(x.split('.')[0])):
            full_path = os.path.join(path, fname)
            file = np.load(full_path).reshape(size)
            files.append(file)
        files = np.stack(files, axis=0)
        return files, sorted(indexs)

if __name__ == '__main__':
    if 'win' in sys.platform:
        root = './log/one pixel'
    else:
        root = './log/cw_image/numpy'

    fake_images, indexs = readNumpyFile(root)
    fake_images = torch.from_numpy(fake_images)
    print(indexs)
    print(fake_images.shape)


    if 'win' in sys.platform:
        root = 'E:\ljq\data'
    else:
        root = './data'
    train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=1, shuffle=False)
    #
    model = ConvModel()
    model = model.eval()
    if 'win' in sys.platform:
        log_path = './log'
    else:
        log_path = './log'
    load_model(model, log_path, 'conv_model')
    print(model)

    fx = FeatureExtractor()
    target_module = ['linear']
    fake_features = fx(model, fake_images.unsqueeze(1), target_module)[0]
    # print(fake_features)

    original_images = [train_dataset[index][0] for index in indexs]
    original_images = torch.stack(original_images, dim=0)
    original_features = fx(model, original_images, target_module)[0]
    # print(original_features)

    print(torch.max(original_features))
    print(torch.max(fake_features))
    diff = original_features - fake_features
    # print(diff)
    print(torch.mean(diff))
    plt.hist(diff.cpu().detach().flatten().numpy(), bins=100, log=True)

    plt.show()
    print(torch.sum((torch.abs(diff) > 0.1).float()))

# activation_layer = '3'
# gradcam = GradCam(model, model.feature, [activation_layer])

# grad cam results for original input
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

# grad cam results for cw attack
# root = os.path.join(os.getcwd(), 'log', 'cw_image', 'gradcam', 'mnist', 'activation_layer' + activation_layer)
# if not os.path.exists(root):
#     os.makedirs(root)
# for idx, image in enumerate(fake_images):
#     image = image.unsqueeze(0).unsqueeze(0)
#     activation_map_for_fake_images = gradcam(image)
#     plt.imshow(activation_map_for_fake_images, cmap='binary')
#     plt.savefig(os.path.join(root, str(idx) + '.jpg'))

# grad cam results for one pixel
# root = os.path.join(os.getcwd(), 'log', 'one pixel', 'gradcam', 'mnist', 'activation_layer' + activation_layer)
# if not os.path.exists(root):
#     os.makedirs(root)
# for idx, image in zip(indexs, fake_images):
#     image = image.unsqueeze(0).unsqueeze(0)
#     activation_map_for_fake_images = gradcam(image, target_category=9)
#     plt.imshow(activation_map_for_fake_images, cmap='binary')
#     plt.savefig(os.path.join(root, str(idx) + '.jpg'))
#     plt.show()












