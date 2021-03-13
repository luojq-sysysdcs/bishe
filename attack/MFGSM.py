#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/3/12 15:14
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : MFGSM.py
# @Software: PyCharm

from FGSM import FGSM
from Attack import *
from torch import nn
import torch

class MFGSM(Attack):
    def __init__(self, model, eps=0.3, c=3):
        super(MFGSM, self).__init__('MFGSM', model)
        self.eps = eps
        self.model = model
        self.device = next(model.parameters()).device
        self.c = c
        self.fgsm = FGSM(model, eps)

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images, success = self.fgsm.forward(images, labels)

        if not torch.any(success):
            return None, None

        images = images[success]
        labels = labels[success]
        adv_images = adv_images[success]
        pert = (adv_images - images).detach()

        success = torch.zeros(len(labels), dtype=torch.bool).to(self.device)
        for i in range(1, self.c):
            wanted = torch.logical_not(success)
            x = images[wanted] + pert[wanted] * (i / self.c)
            _, pred = torch.max(self.model(x), 1)
            now_success = pred == labels[wanted]
            update = wanted.detach().clone()
            update[wanted] = now_success
            adv_images[update] = x[now_success].detach().clone()

            success[wanted] = now_success

            if torch.all(success):
                break

        return adv_images, success












