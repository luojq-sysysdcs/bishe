#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/30 19:12
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : MyAttack.py
# @Software: PyCharm

import torch.optim as optim
from attack.Attack import Attack
from GenerateData import *
from models.SimpleModel import *
from matplotlib import pyplot as plt
from PIL import Image
import torchvision


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()


class MyAttack(Attack):
    def __init__(self, model, eps, dataset,
                 class_num=10, T=100, steps=10000, lr=1e-3, path='./log/perturbation'):
        super(MyAttack,self).__init__('MyAttack', model)
        self.eps = eps
        self.model = model.eval()
        self.dataset = dataset
        self.T = T
        self.steps = steps
        self.class_num = class_num
        self.lr = lr
        self.perturbation = None
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.perturbation = self.load_perturbation()
        if self.perturbation is None:
            self.perturbation = self._make_perturbation()
            self.save_perturbation()
        self.evaluate()

    def tanh_space(self, x):
        return (torch.tanh(x)+1) * (1/2)

    def inverse_tanh_space(self, x):
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    def _make_perturbation(self):

        w = torch.rand_like(self.dataset[0][0]).to(self.device)
        w = w.repeat([10] + [1 for _ in range(len(w.shape))])
        # original = w.detach().clone()
        # w = self.dataset[0][0].detach().clone().unsqueeze(0)
        w = self.inverse_tanh_space(w).detach()
        w.requires_grad = True


        # mseloss = nn.MSELoss()
        optimizer = optim.Adam([w], lr=self.lr)
        # logit_target = torch.ones((1, self.class_num), dtype=torch.float).to(self.device) * -1
        # logit_target[0, 0] = 1

        celoss = nn.CrossEntropyLoss()
        # optimizer = optim.Adam([w], lr=self.lr)
        pop_target = torch.tensor([i for i in range(10)], dtype=torch.long).to(self.device)

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

            if (step + 1) % (self.steps // 10) == 0:
                print(output)
                print(L3.item())

                perturb = self.tanh_space(w).detach().cpu()
                img = torchvision.utils.make_grid(perturb, nrow=5, padding=2, pad_value=1)
                show(img)
        # plt.imshow(perturb.cpu().detach().clone().squeeze().numpy(), cmap='binary')
        # plt.show()
        perturb = self.tanh_space(w).detach().cpu()
        print(perturb.shape)
        return perturb

    def save_perturbation(self):
        for idx, img in enumerate(self.perturbation):
            img = np.uint8(img.detach().clone().cpu().squeeze().numpy() * 255)
            img = Image.fromarray(img)
            img = img.convert('L')
            img.save(os.path.join(self.path, str(idx) + '.jpg'))

    def load_perturbation(self):
        self.perturbation = []
        for _, dirs, fnames in os.walk(self.path):
            fnames = sorted(fnames, key=lambda x: int(x.split('.')[0]))
            for fname in fnames:
                label = int(fname.split('.')[0])
                p = os.path.join(self.path, fname)
                perturb = Image.open(p).convert('L')
                perturb = np.array(perturb) / 255
                perturb = torch.from_numpy(perturb).float()
                perturb = perturb.reshape(-1, *perturb.shape)
                self.perturbation.append(perturb)
                # self.perturbation[label] = perturb
            break
        return torch.stack(self.perturbation, dim=0)

    def evaluate(self):
        output = self.perturbation.to(self.device)
        output = self.model(output)
        pre = torch.max(output, dim=1)[1]
        print(output)
        print(pre)

    def __call__(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root = 'E:/ljq/data'
    batch_size = 4
    train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=batch_size, shuffle=False)

    # root = './log/cw/model3-1e1-0'
    # batch_size = 128
    # adversarial_dataset, adversarial_dataloader = get_adversarial_data(root, batch_size=batch_size, shuffle=False)

    model = Model3()
    log_path = './log'
    load_model(model, log_path, 'model3')
    model = model.eval()
    model = model.to(device)
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
    # count = 0
    # for idx, (img, label) in enumerate(train_dataset):
    #     if count == 10:
    #         break
    #     if label != -1:
    #         count += 1
    #         adversary(train_dataset[idx][0], 5, rate=0.8)










