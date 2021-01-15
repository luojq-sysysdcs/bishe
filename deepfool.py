#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/10 21:36
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : deepfool.py
# @Software: PyCharm


import torch
from torch import nn
from Attack import Attach

class DeepFool(Attach):
    def __init__(self, model, steps=1000):
        super(DeepFool, self).__init__('deepfool', model)
        self.steps = steps

    def forward(self, images, labels):

        adv_images = images.clone().detach().to(self.device)
        # adv_images = torch.empty_like(images).to(self.device)
        for b in range(images.shape[0]):
            image = images[b:b+1].clone().detach().to(self.device)
            label = labels[b:b+1].clone().detach().to(self.device)



            for i in range(self.steps):
                image.requires_grad = True
                output = self.model(image)[0]
                _, pred = torch.max(output, dim=0)

                wrong_classes = list(range(len(output)))
                del wrong_classes[label.item()]

                # stop if prediction is wrong
                if pred.item() != label:
                    image = torch.clamp(image, min=0, max=1).detach()
                    break

                ws = self.construct_jacobian(output, image)

                f_0 = output[label]
                w_0 = ws[label]

                f_k = output[wrong_classes]
                w_k = ws[wrong_classes]

                f_prime = f_k - f_0
                w_prime = w_k - w_0

                value = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
                _, hat_L = torch.min(value, 0)

                r = (torch.abs(f_prime[hat_L])) / torch.norm(w_prime[hat_L], p=2)**2 * w_prime[hat_L]

                image = torch.clamp(image + r, min=0, max=1).detach()



            adv_images[b:b+1] = image
        return adv_images

    def construct_jacobian(self, y, x, retain_graph=False):
        x_grads = []
        for idx, y_element in enumerate(y.flatten()):
            if x.grad is not None:
                x.grad.zero_()

            y_element.backward(retain_graph=retain_graph or idx < y.numel() - 1)
            x_grads.append(x.grad.clone().detach())

        return torch.stack(x_grads).reshape(*y.shape, *x.shape)








