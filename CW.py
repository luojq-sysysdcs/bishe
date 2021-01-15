__author__ = 'uniform64'

import os
import  numpy
import torch
import torch.nn as nn
import torch.optim as optim
from Attack import Attach


class CW(Attach):
    def __init__(self, model, c=1e-1, kappa=1e-10, steps=10000, lr=1e-1, binary_search_steps=2):
        super(CW, self).__init__('CW', model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.repeat = self.binary_search_steps > 10

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.shape[0]
        # labels = self._transform_label(images, labels)

        best_adv_images = torch.randn_like(images).to(self.device)
        best_l2 = 1e10*torch.ones((len(images))).to(self.device)

        dim = len(images.shape)

        MESLoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        lower_bound = torch.zeros(batch_size, dtype=torch.float).to(self.device)
        const = (torch.ones(batch_size, dtype=torch.float) * self.c).to(self.device)
        upper_bound = (torch.ones(batch_size, dtype=torch.float) * 1e9).to(self.device)

        outer_success = torch.empty(batch_size, dtype=torch.bool).to(self.device) * torch.tensor([False]).to(self.device)
        for outer_step in range(self.binary_search_steps):
            print('-'*20)
            print('const', const)
            w = self.inverse_tanh_space(images).detach()
            w.requires_grad = True
            optimizer = optim.Adam([w], lr=self.lr)

            if self.repeat and outer_step == self.binary_search_steps - 1:
                const = upper_bound

            prev_cost = 1e10
            success = torch.empty(batch_size, dtype=torch.bool).to(self.device) * torch.tensor([False]).to(self.device)
            for step in range(self.steps):
                adv_images = self.tanh_space(w)

                current_l2 = MESLoss(Flatten(adv_images), Flatten(images)).sum(dim=1)

                # compute the loss
                l2_loss = current_l2.sum()
                outputs = self.model(adv_images)
                f_loss = (self.f(outputs, labels) * const).sum()

                # optimize the 'w'
                loss =  l2_loss + f_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # judge whether the attack is successful
                _, pre = torch.max(outputs.detach(), dim=1)
                correct = (pre==labels)
                success += correct
                outer_success += correct

                # update the beat_l2 and best_adv_images
                mask = correct.float() * (best_l2 > current_l2.detach())
                best_l2 = mask * current_l2.detach() + (1 - mask) * best_l2

                mask = mask.view([-1] + [1]*(dim-1))
                best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

                # early stop if loss does not converge
                if step % (self.steps // 10) == 0:
                    if loss.item() > prev_cost:
                        break
                    else:
                        prev_cost = loss.item()
            print(success)
            for i in range(batch_size):
                if success[i]:
                    upper_bound[i] = min(upper_bound[i], const[i])
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    lower_bound[i] = max(lower_bound[i], const[i])
                    if upper_bound[i] < 1e8:
                        const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        const[i] *= 10
        print('best L2 loss:', best_l2)
        return best_adv_images, outer_success, best_l2




    def f(self, outputs, labels):
        onehot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        target =torch.masked_select(outputs, onehot_labels.bool())
        others =torch.max((1 - onehot_labels)*outputs, dim=1)[0]
        # others = torch.sum((1 - onehot_labels)*outputs, dim=1)
        # print(self._targeted*(target-others))
        return torch.clamp(self._targeted*(target-others), min=-self.kappa) # notice the minus



    def tanh_space(self, x):
        return 1/2*(torch.tanh(x)+1)

    def inverse_tanh_space(self, x):
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))





