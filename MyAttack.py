#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/30 19:12
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : MyAttack.py
# @Software: PyCharm

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Attack import Attack
from GenerateData import *
from SimpleModel import *
from matplotlib import pyplot as plt
from PIL import Image


class MyAttack(Attack):
    def __init__(self, model, eps, dataset,
                 class_num=10, T=1000, steps=500, lr=1e-2, path='./log/perturbation'):
        super(MyAttack,self).__init__('MyAttack', model)
        self.eps = eps
        self.model = model.eval()
        self.dataset = dataset
        self.T = T
        self.steps = steps
        self.class_num = class_num
        self.lr = lr
        self.perturbation = [None for i in range(10)]
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.load_perturbation()
        self.perturbation[5] = self._make_perturbation(5)

    def tanh_space(self, x):
        return (torch.tanh(x)+1) * (1/2)

    def inverse_tanh_space(self, x):
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    def _make_perturbation(self, attack_label, num=100):
        if self.perturbation[attack_label] is not None:
            return self.perturbation[attack_label]
        # imgs = []
        # count = 0
        # perturb = None
        # for img, lb in self.dataset:
        #     if count == num:
        #         break
        #     if lb == 0:
        #         imgs.append(img.detach().clone())
        # perturb = torch.sum(torch.cat(imgs, dim=0), dim=0, keepdim=True).unsqueeze(0)
        # perturb = perturb / torch.max(perturb)
        # perturb = perturb.detach().clone()
        w = torch.rand_like(self.dataset[0][0]).unsqueeze(0)
        original = w.detach().clone()
        # w = self.dataset[0][0].detach().clone().unsqueeze(0)
        w = self.inverse_tanh_space(w).detach()
        w.requires_grad = True


        mseloss = nn.MSELoss()
        optimizer = optim.Adam([w], lr=self.lr)
        logit_target = torch.ones((1, self.class_num), dtype=torch.float) * -1
        logit_target[0, 0] = 1

        celoss = nn.CrossEntropyLoss()
        # optimizer = optim.Adam([w], lr=self.lr)
        pop_target = torch.tensor([0], dtype=torch.long)

        for step in range(self.steps):
            perturb = self.tanh_space(w)
            output = self.model(perturb) / self.T
            # L1 = mseloss(output, logit_target) / 10000
            # L2 = torch.sqrt(torch.max(perturb ** 2))
            # L2 = torch.sum(torch.abs(perturb)) / 1000
            L3 = celoss(output, pop_target)
            # L4 = mseloss(perturb, original) / 10
            # L2 = torch.mean(perturb)
            loss = L3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # perturb = perturb - torch.min(perturb).item()
            # perturb = perturb / torch.max(perturb).item()

            print(output)
            print(L3.item())

        perturb = self.tanh_space(w)
        plt.imshow(perturb.detach().clone().squeeze().numpy(), cmap='binary')
        plt.show()
        print(perturb.shape)
        self.save_perturbation(perturb, attack_label)
        return perturb.squeeze(0)

    def save_perturbation(self, perturb, attack_label):
        img = np.uint8(perturb.detach().clone().cpu().squeeze().numpy() * 255)
        img = Image.fromarray(img)
        img = img.convert('L')
        img.save(os.path.join(self.path, str(attack_label) + '.jpg'))

    def load_perturbation(self):
        for _, dirs, fnames in os.walk(self.path):
            for fname in fnames:
                label = int(fname.split('.')[0])
                p = os.path.join(self.path, fname)
                perturb = Image.open(p).convert('L')
                perturb = np.array(perturb) / 255
                perturb = torch.from_numpy(perturb).float()
                perturb = perturb.reshape(-1, *perturb.shape)
                self.perturbation[label] = perturb
            break

    def __call__(self, image, label, rate=0.8):
        # plt.imshow(image.detach().clone().squeeze().numpy(), cmap='binary')
        # plt.show()
        # print(self.model(self.perturbation[label].unsqueeze(0)))
        diff = self.perturbation[label] - image
        # print(self.model(image.unsqueeze(0)))
        image = image + (1 - rate) * diff
        image = image / torch.max(image)
        output = self.model(image.unsqueeze(0))
        pre = torch.max(output, dim=1)[1]
        print(output)
        print(pre)
        plt.imshow(image.detach().clone().squeeze().numpy(), cmap='binary')
        plt.show()
        return image


if __name__ == '__main__':

    root = 'E:\ljq\data'
    batch_size = 4
    train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=batch_size, shuffle=False)

    root = './log/PGD-model3-0.3'
    adversarial_dataset, adversarial_dataloader = get_adversarial_data(root, batch_size=batch_size, shuffle=False)

    model = Model3()
    log_path = './log'
    load_model(model, log_path, 'model3-ad-0.3')
    model = model.eval()
    print(model)

    # choice = 0
    # for idx, (img, label) in enumerate(train_dataset):
    #     if idx > 150:
    #         break
    #     if label != 5:
    #         continue
    #     ad_img, _, _ = adversarial_dataset.get_sample(idx, choice)
    #     if ad_img is None:
    #         continue
    #     diff = img - ad_img
    #     diff = (diff - torch.min(diff)) / (torch.max(diff) - torch.min(diff))
    #     plt.imshow(diff.detach().squeeze().clone().numpy(), cmap='binary')
    #     plt.show()
    #     print(model(diff.unsqueeze(0)))



    adversary = MyAttack(model, eps=0.4, dataset=train_dataset)
    count = 0
    for idx, (img, label) in enumerate(train_dataset):
        if count == 2:
            break
        if label == 5:
            count += 1
            adversary(train_dataset[idx][0], 5, rate=0.8)










