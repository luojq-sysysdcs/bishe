#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/7 21:29
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : FGSM.py
# @Software: PyCharm

import torch
import torch.nn as nn
from Attack import *


class FGSM():
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT: 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        # >>> attack = torchattacks.FGSM(model, eps=0.007)
        # >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.007,random_start=False):
        self.eps = eps
        self.model = model
        self.device = next(model.parameters()).device
        self.random_start = random_start

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        adv_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        adv_images.requires_grad = True
        outputs = self.model(adv_images)
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = adv_images - self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        _, pred = torch.max(self.model(adv_images), 1)

        return adv_images, pred == labels
