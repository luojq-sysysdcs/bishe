#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/27 11:22
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : TSNE.py
# @Software: PyCharm
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from Simple_model import *
from utils import *
from FGSM import FGSM
from PGD import PGD
from GenerateData import *
import os
from PIL import Image
import matplotlib as mpl
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn import datasets
from sklearn.manifold import TSNE
from Visualizing import FeatureExtractor



def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    data, label, n_samples, n_features = get_data()
    print(len(data[0]))
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    fig.show()


if __name__ == '__main__':

    root = 'E:\ljq\data'
    batch_size = 100
    train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=batch_size, shuffle=False)

    root = './log/PGD-model3-0.3'
    adversarial_dataset, adversarial_dataloader = get_adversarial_data(root, batch_size=batch_size, shuffle=False)

    # model = SM()
    # model = Model3()
    # log_path = './log'
    # load_model(model, log_path, 'model3')
    # model = model.eval()
    # print(model)

    model = Model4()
    log_path = './log'
    load_model(model, log_path, 'model4')
    model = model.eval()
    print(model)

    count = 0
    for idx, (imgs, tl, al) in enumerate(adversarial_dataloader):
        output = model(imgs)
        pre = torch.max(output, dim=1)[1]
        count += torch.sum((pre==tl).float())
    print(count / len(adversarial_dataset))

    # features = []
    # labels = []
    # images = []
    # fx = FeatureExtractor()
    # target_module = ['linear']
    # for idx, (imgs, ls) in enumerate(train_dataloader):
    #     if idx == 10:
    #         break
    #     images.append(imgs.flatten(1))
    #     labels.append(ls)
    #     feature = fx(model, imgs, target_module)[0]
    #     features.append(feature)
    # images = torch.cat(images, 0).detach().cpu().numpy()
    # labels = torch.cat(labels, 0).detach().cpu().numpy()
    # features = torch.cat(features, 0).detach().cpu().numpy()
    # print(images.shape)
    # print(labels.shape)
    # print(features.shape)
    #
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # print('Computing t-SNE embedding')
    # t0 = time()
    # result = tsne.fit_transform(images)
    # fig = plot_embedding(result, labels,
    #                      't-SNE embedding of the digits (time %.2fs)'
    #                      % (time() - t0))
    # fig.show()
# -----------------------------------------
#     features = []
#     true_labels = []
#     adversarial_labels = []
#     images = []
#     fx = FeatureExtractor()
#     target_module = ['linear']
#     for idx, (imgs, tl, al) in enumerate(adversarial_dataloader):
#         if idx == 10:
#             break
#         images.append(imgs.flatten(1))
#         true_labels.append(tl)
#         adversarial_labels.append(al)
#         feature = fx(model, imgs, target_module)[0]
#         features.append(feature)
#     images = torch.cat(images, 0).detach().cpu().numpy()
#     true_labels = torch.cat(true_labels, 0).detach().cpu().numpy()
#     adversarial_labels = torch.cat(adversarial_labels, 0).detach().cpu().numpy()
#     features = torch.cat(features, 0).detach().cpu().numpy()
#     print(images.shape)
#     print(true_labels.shape)
#     print(adversarial_labels.shape)
#     print(features.shape)
#
#     tsne = TSNE(n_components=2, init='pca', random_state=0)
#     print('Computing t-SNE embedding')
#     t0 = time()
#     result = tsne.fit_transform(features)
#     fig = plot_embedding(result, adversarial_labels,
#                          't-SNE embedding of the digits (time %.2fs)'
#                          % (time() - t0))
#     fig.show()

# -----------------------------------------

    true_labels = []
    true_images = []
    true_features = []
    adversarial_labels = []
    adversarial_images = []
    adversarial_features = []
    fx = FeatureExtractor()
    target_module = ['fc']
    choice = 0
    for idx, (imgs, tl, al) in enumerate(adversarial_dataloader):
        if len(adversarial_labels) >= 500:
            break
        wanted = al == choice
        # wanted = torch.ones(imgs.shape[0], dtype=torch.bool)
        imgs = imgs[wanted]
        al = al[wanted]
        adversarial_images.append(imgs.flatten(1))
        adversarial_labels += list(al)
        # feature = fx(model, imgs, target_module)[0]
        feature = model(imgs)
        adversarial_features.append(feature)

    for idx, (imgs, ls) in enumerate(train_dataloader):
        if len(true_labels) >= 500:
            break
        wanted = ls == choice
        # wanted = torch.ones(imgs.shape[0], dtype=torch.bool)
        imgs = imgs[wanted]
        ls = ls[wanted]
        true_images.append(imgs.flatten(1))
        true_labels += list(ls)
        # feature = fx(model, imgs, target_module)[0]
        feature = model(imgs)
        true_features.append(feature)

    adversarial_features = torch.cat(adversarial_features, 0).detach().cpu().numpy()[:500]
    adversarial_images = torch.cat(adversarial_images, 0).detach().cpu().numpy()[:500]
    adversarial_labels = np.array(adversarial_labels)[:500]
    true_features = torch.cat(true_features, 0).detach().cpu().numpy()[:500]
    true_images = torch.cat(true_images, 0).detach().cpu().numpy()[:500]
    true_labels = np.array(true_labels)[:500]
    print(adversarial_images.shape)
    print(adversarial_labels.shape)
    print(adversarial_features.shape)
    print(true_images.shape)
    print(true_labels.shape)
    print(true_features.shape)

    diff = adversarial_features - true_features
    # plt.bar(range(10), np.mean(diff, axis=0))
    width = 0.3
    plt.bar(range(10), np.mean(adversarial_features, axis=0), width=width, label='adv')
    plt.bar(np.arange(10) + width, np.mean(true_features, axis=0), width=width, label='clean')
    plt.title(str(choice))
    plt.legend()
    plt.show()



    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # print('Computing t-SNE embedding')
    # t0 = time()
    # result = tsne.fit_transform(np.concatenate((adversarial_features, true_features),axis=0))
    # fig = plot_embedding(result, np.concatenate((adversarial_labels, true_labels+5), axis=0),
    #                      't-SNE embedding of the digits (time %.2fs)'
    #                      % (time() - t0))
    # fig.show()
# ------------------------------------------------
#     root = 'E:\ljq\data'
#     batch_size = 1
#     train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=batch_size, shuffle=False)
#
#     root = './log/PGD'
#     adversarial_dataset, adversarial_dataloader = get_adversarial_data(root, batch_size=batch_size, shuffle=False)
#
#     true_labels = []
#     true_images = []
#     true_features = []
#     adversarial_labels = []
#     adversarial_images = []
#     adversarial_features = []
#     fx = FeatureExtractor()
#     target_module = ['linear']
#     choice = 1
#     choice_ = 0
#
#     for idx, (imgs, ls) in enumerate(train_dataloader):
#         if idx > 1000:
#             break
#         if ls.item() != choice:
#             continue
#         img_, _, _ = adversarial_dataset.get_sample(idx, choice_)
#         if img_ is None:
#             continue
#
#         img_ = img_.unsqueeze(0)
#         adversarial_images.append(img_.flatten(1))
#         feature = fx(model, img_, target_module)[0]
#         adversarial_features.append(feature)
#         adversarial_labels.append(choice_)
#
#         true_images.append(imgs.flatten(1))
#         true_labels += list(ls)
#         feature = fx(model, imgs, target_module)[0]
#         true_features.append(feature)
#
#     adversarial_features = torch.cat(adversarial_features, 0).detach().cpu().numpy()
#     adversarial_images = torch.cat(adversarial_images, 0).detach().cpu().numpy()
#     adversarial_labels = np.array(adversarial_labels)
#     true_features = torch.cat(true_features, 0).detach().cpu().numpy()
#     true_images = torch.cat(true_images, 0).detach().cpu().numpy()
#     true_labels = np.array(true_labels)
#     print(adversarial_images.shape)
#     print(adversarial_labels.shape)
#     print(adversarial_features.shape)
#     print(true_images.shape)
#     print(true_labels.shape)
#     print(true_features.shape)
#
#     diff_features = adversarial_features - true_features
#     print(np.mean((true_features), axis=1))
#     print(np.mean((adversarial_features), axis=1))
#     print(np.mean((diff_features), axis=1))
#     print(np.count_nonzero(np.abs(diff_features) < 0.1))
#     plt.bar(range(100), np.mean((true_features), axis=0))
#     plt.show()
    # plt.bar(range(100), np.mean(true_features, axis=0))
    # # plt.hist(diff_features.flatten(), bins=1000)
    # # plt.hist(true_features.flatten(), bins=100)
    # plt.show()


# -----------------------------------------
#     print(model.linear.weight)
#     plt.hist(model.linear.weight.detach().numpy().flatten(), bins=100)
#     plt.show()









