#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/10 21:36
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : deepfool.py
# @Software: PyCharm


import torch
from torch import nn
from Attack import *
import time


class DeepFool(Attack):
    def __init__(self, model, num_classes, steps=100):
        super(DeepFool, self).__init__('deepfool', model)
        self.steps = steps
        self.num_classes = num_classes
        self.mu = 0.05

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.shape[0]
        images.requires_grad = False

        success = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        success_count = 0

        for step in range(self.steps):
            t0 = time.time()

            # output = self.model(images)
            # _, pred = torch.max(output, dim=1)

            # success = torch.logical_and(success, torch.ne(pred, labels))
            # if torch.all(success):
            #     print('attack succeed! stop early! step: %d' % step)
            #     break

            fail = torch.logical_not(success)
            fail_images = images[fail]
            fail_labels = labels[fail]
            count = fail_images.shape[0]
            wrong_classes = [list(range(self.num_classes)) for _ in range(count)]
            for i in range(count):
                del wrong_classes[i][fail_labels[i].item()]
            wrong_classes = torch.tensor(wrong_classes, dtype=torch.long, device=self.device)

            ws = [torch.autograd.functional.jacobian(self.model, fail_images[i:i + 1]) for i in range(count)]
            ws = torch.cat(ws, dim=0).squeeze(2)
            output = self.model(fail_images)

            f_0 = output[torch.arange(0, count), fail_labels].unsqueeze(1)
            w_0 = ws[torch.arange(0, count), fail_labels].unsqueeze(1)

            f_k = output[torch.arange(0, count, dtype=torch.long).unsqueeze(1), wrong_classes]
            w_k = ws[torch.arange(0, count, dtype=torch.long).unsqueeze(1), wrong_classes]

            f_prime = f_k - f_0
            w_prime = w_k - w_0

            value = torch.abs(f_prime) / torch.norm(w_prime.flatten(start_dim=2), p=1, dim=2)
            _, hat_L = torch.min(value, 1)

            f_hat = torch.abs(f_prime[torch.arange(0, count), hat_L])
            w_hat = w_prime[torch.arange(0, count), hat_L]
            w_hat_norm = torch.norm(w_hat.flatten(start_dim=1), p=2, dim=1)

            r = (f_hat / w_hat_norm ** 2).view([count, 1, 1, 1]) * \
                w_prime[torch.arange(0, count), hat_L] * (1 + self.mu)

            fail_images = torch.clamp(fail_images + r, min=0, max=1).detach()

            # a unresolved bug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
            # when i use the next code to ungrate the 'success'
            # the num of success is not equal to the one calculated in the end
            # output = self.model(fail_images)
            # _, pred = torch.max(output, dim=1)
            #
            # success[fail] = torch.ne(pred, fail_labels)
            images[fail] = fail_images.clone().detach()

            output = self.model(images)
            _, pred = torch.max(output, dim=1)
            success = torch.ne(pred, labels)

            if torch.all(success):
                print('attack succeed! stop early! step: %d' % step)
                break

            t1 = time.time()
            print(step, t1 - t0)
            # print(success)
            # print(torch.sum(success))
            # success_count = torch.sum(success)

        # for b in range(images.shape[0]):
        #     image = images[b:b + 1]
        #     label = labels[b:b + 1]
        #
        #     for i in range(self.steps):
        #         image.requires_grad = True
        #         output = self.model(image)[0]
        #         _, pred = torch.max(output, dim=0)
        #
        #         wrong_classes = list(range(len(output)))
        #         del wrong_classes[label.item()]
        #
        #         # stop if prediction is wrong
        #         if pred.item() != label:
        #             image = torch.clamp(image, min=0, max=1).detach()
        #             success[b] = True
        #             break
        #
        #         # ws = self.construct_jacobian(output, image)
        #         ws = torch.autograd.functional.jacobian(self.model, images[b:b + 1])[0]
        #         f_0 = output[label]
        #         w_0 = ws[label]
        #
        #         f_k = output[wrong_classes]
        #         w_k = ws[wrong_classes]
        #
        #         f_prime = f_k - f_0
        #         w_prime = w_k - w_0
        #
        #         value = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        #         _, hat_L = torch.min(value, 0)
        #
        #         r = (torch.abs(f_prime[hat_L])) / torch.norm(w_prime[hat_L], p=2) ** 2 * w_prime[hat_L]
        #
        #         image = torch.clamp(image + r, min=0, max=1).detach()
        #
        #     adv_images[b:b + 1] = image

        # output = self.model(images)
        # _, pred = torch.max(output, dim=1)
        # success = torch.ne(pred, labels)
        # if torch.sum(success) != success_count:
        #     raise ValueError
        images[
            torch.logical_not(success)] = 0.5  # torch.randn(size=(1, *images.shape[1:])).clamp_(0, 1).to(self.device)
        return images, success

    def construct_jacobian(self, y, x, retain_graph=False):

        x_grads = []
        for idx, y_element in enumerate(y.flatten()):
            if x.grad is not None:
                x.grad.zero_()

            y_element.backward(retain_graph=retain_graph or idx < y.numel() - 1)
            x_grads.append(x.grad.clone().detach())

        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
