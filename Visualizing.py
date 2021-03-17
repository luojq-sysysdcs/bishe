#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/14 10:56
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : Visualizing.py
# @Software: PyCharm

from visualization.GradCam import *
from models.SimpleModel import *
from utils import load_model
from matplotlib import pyplot as plt
import os
import numpy as np
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.target_activations = []
        self.x = None
        self.target_layers = None
        self.index = 0
        self.model = model
        self.target_layers = target_layers

    def extractor(self, model):
        for name, child in model.named_children():
            if len(self.target_layers) > self.index and name == self.target_layers[self.index]:
                self.index += 1
                if self.index == len(self.target_layers):
                    self.x = child(self.x)
                    self.target_activations.append(self.x)
                else:
                    self.extractor(child)
            else:
                    self.x = child(self.x)
            if "avgpool" in name.lower() or 'maxpool' in name.lower():
                self.x = self.x.view(self.x.size(0), -1)

    def __call__(self, x):
        self.index = 0
        self.target_activations = []
        self.x = x
        self.extractor(self.model)
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
    model = vgg_mnist()















