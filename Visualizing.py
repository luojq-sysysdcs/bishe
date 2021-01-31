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
        self.target_activations = []
        self.x = None
        self.target_layers = None
        self.index = 0

    def extractor(self, model):
        for name, child in model.named_children():
            if len(self.target_layers) > self.index and name == self.target_layers[self.index]:
                self.index += 1
                if self.index == len(self.target_layers):
                    self.x = child(self.x)
                    self.target_activations.append(self.x)
                else:
                    self.extract(child)
            else:
                if "avgpool" in name.lower() or 'maxpool' in name.lower():
                    self.x = child(self.x)
                    self.x = self.x.view(self.x.size(0), -1)
                elif 'fc' in name.lower() or 'linear' in name.lower():
                    self.x = self.x.view(self.x.size(0), -1)
                    self.x = child(self.x)
                else:
                    self.x = child(self.x)

    def __call__(self, model, x, target_layers):
        self.index = 0
        self.target_activations = []
        self.model = model
        self.x = x.detach().clone()
        self.target_layers = target_layers
        self.extractor(model)
        return self.target_activations


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
    root = 'E:\ljq\data'
    batch_size = 100
    train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=batch_size, shuffle=False)

    root = './log/PGD-model4-0.3'
    adversarial_dataset, adversarial_dataloader = get_adversarial_data(root, batch_size=batch_size, shuffle=False)

    model = Model3()
    model = model.eval()
    log_path = './log'
    load_model(model, log_path, 'model3')
    print(model)

    fx = FeatureExtractor()
    target_module = ['linear']

    activation_layer = '3'
    gradcam = GradCam(model, model.feature, [activation_layer])

    # grad cam results for original input
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












