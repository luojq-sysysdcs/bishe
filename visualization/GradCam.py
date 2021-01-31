#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 15:17
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : GradCam.py
# @Software: PyCharm

from PIL import Image
import numpy as np
import torch
import cv2
import sys
sys.path.append("..")
from Simple_model import *
from GenerateData import *
from utils import *
from matplotlib import pyplot as plt


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.index = 0
        self.target_activations = []
        self.gradients = []
        self.x = None

    def get_gradients(self):
        return self.gradients

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def extract(self, model):
        # self.target_activations = []
        for name, child in model.named_children():
            if len(self.target_layers) > self.index and name == self.target_layers[self.index]:
                self.index += 1
                if self.index == len(self.target_layers):
                    self.x = child(self.x)
                    self.x.register_hook(self.save_gradient)
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

    def __call__(self, x):
        self.index = 0
        self.target_activations = []
        self.gradients = []
        self.x = x.detach().clone()
        self.extract(self.model)
        if self.index ==0:
            raise ValueError('no activation is extracted!')
        return self.target_activations, self.x


class GradCam:
    def __init__(self, model, target_layers):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.model = model.to(self.device)

        self.extractor = ModelOutputs(self.model, target_layers)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        input_img = input_img.to(self.device)

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        one_hot = one_hot.to(self.device)

        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        # print(weights)
        # print(target)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)





if __name__ == '__main__':
    root = 'E:\ljq\data'
    batch_size = 100
    train_dataset, train_dataloader = \
        generate_data(root, 'MNIST', train=True, batch_size=batch_size, shuffle=False, shuffle_label=False)

    root = '../log/PGD-model4-0.3'
    adversarial_dataset, adversarial_dataloader = \
        get_adversarial_data(root, batch_size=batch_size, shuffle=False)

    model = Model3()
    model = model.eval()
    log_path = '../log'
    load_model(model, log_path, 'model3')
    print(model)
    target_layer = ['extract']

    # model = Model4()
    # model = model.eval()
    # log_path = '../log'
    # load_model(model, log_path, 'model4-random')
    # print(model)
    # target_layer = ['extract']

    # model = ConvModel()
    # model = model.eval()
    # log_path = '../log'
    # load_model(model, log_path, 'conv_model')
    # print(model)
    # target_layer = ['feature', '2']

    gradcam = GradCam(model, target_layer)

    count = 0
    for idx, (image, label) in enumerate(train_dataset):
        if count == 10:
            break
        if label == 5:
            print(idx)
            count += 1
            cam = gradcam(image.unsqueeze(0))
            plt.imshow(cam)
            plt.show()








































