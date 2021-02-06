__author__ = 'uniform64'

import torch
import torch.nn as nn
import torch.optim as optim
from Attack import *


class CW(Attack):
    def __init__(self, model, c=1e1, kappa=1e-10, steps=500, lr=1e-1, binary_search_steps=2, use_cuda=False):
        super(CW, self).__init__('CW', model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.repeat = self.binary_search_steps > 10
        self.use_cuda = use_cuda
        print(self.device)

    def forward(self, images, labels):
        if self.use_cuda:
            images = images.clone().detach().cuda()
            labels = labels.clone().detach().cuda()
            best_adv_images = torch.randn_like(images).cuda()
            best_l2 = 1e10 * torch.ones((len(images))).cuda()
        else:
            images = images.clone().detach()
            labels = labels.clone().detach()
            best_adv_images = torch.randn_like(images)
            best_l2 = 1e10 * torch.ones((len(images)))
        batch_size = images.shape[0]

        dim = len(images.shape)

        MESLoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        success = torch.empty(batch_size, dtype=torch.bool).to(self.device) * torch.tensor([False]).to(self.device)

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True
        optimizer = optim.Adam([w], lr=self.lr)

        prev_cost = 1e10

        for step in range(self.steps):
            adv_images = self.tanh_space(w)

            current_l2 = MESLoss(Flatten(adv_images), Flatten(images)).sum(dim=1)

            # compute the loss
            l2_loss = current_l2.sum()
            outputs = self.model(adv_images)
            f_loss = (self.f(outputs, labels) * self.c).sum()

            # optimize the 'w'
            loss = l2_loss + f_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # judge whether the attack is successful
            _, pre = torch.max(outputs.detach(), dim=1)
            correct = (pre == labels)

            success += correct

            # update the beat_l2 and best_adv_images
            mask = correct.float() * (best_l2 > current_l2.detach())
            best_l2 = mask * current_l2.detach() + (1 - mask) * best_l2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # early stop if loss does not converge
            if step % (self.steps // 10) == 0:
                if loss.item() > prev_cost and torch.all(success):
                    print('early stop: %d !' % (step + 1))
                    break
                else:
                    prev_cost = loss.item()
        print(success)
        return best_adv_images, success

    def f(self, outputs, labels):
        onehot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        target = torch.masked_select(outputs, onehot_labels.bool())
        others = torch.max((1 - onehot_labels) * outputs, dim=1)[0]
        return torch.clamp(self._targeted * (target - others), min=-self.kappa)  # notice the minus

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        return self.atanh(x * 2 - 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))
