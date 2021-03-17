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
from sklearn.metrics.pairwise import cosine_similarity
from Visualizing import FeatureExtractor

def show(img, str=None, path=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    if str is not None:
        plt.title(str)
    if path is not None:
        plt.savefig(path)
    plt.show()


class MyAttack(Attack):
    def __init__(self, model, eps, dataset, fx, feature_vector=None,
                 class_num=10, T=100, steps=2000, lr=1e-1, path='./log/perturbation', device='cpu'):
        super(MyAttack,self).__init__('MyAttack', model)
        self.device = device
        self.eps = eps
        self.model = model.eval()
        self.dataset = dataset
        self.T = T
        self.steps = steps
        self.class_num = class_num
        self.lr = lr
        self.perturbation = None
        self.path = path
        self.feature_vector = feature_vector
        self.fx = fx
        # if not os.path.exists(self.path):
        #     os.makedirs(self.path)
        #
        # self.perturbation = self.load_perturbation()
        # if self.perturbation is None:
        #     self.perturbation = self._make_perturbation()
        #     self.save_perturbation()
        # self.evaluate()

    def tanh_space(self, x):
        return (torch.tanh(x)+1) * (1/2)

    def inverse_tanh_space(self, x):
        return torch.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    def forward(self, images, labels, target_labels, batch_size=10, target=0):
        batch_size = images.shape[0]

        # w = torch.empty((batch_size, *self.dataset[0][0].shape), dtype=torch.float).uniform_(0.0, 1)
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        optimizer = optim.Adam([w], lr=self.lr)

        COSloss = nn.CosineEmbeddingLoss(reduction='none')
        MESLoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()
        # ce_loss = nn.CrossEntropyLoss()
        y = torch.tensor([1] * w.shape[0], dtype=torch.long).to(self.device)
        target_features = self.feature_vector[target_labels]
        # true_features = self.feature_vector[labels]

        for step in range(self.steps):
            adv_images = self.tanh_space(w)

            output = self.fx(adv_images)[0].flatten(start_dim=1)
            # l2_loss = MESLoss(Flatten(adv_images), Flatten(images)).sum()
            # loss1 = MESLoss(target_features, output).sum()
            # loss2 = MESLoss(true_features, output).sum()
            cos_loss = COSloss(target_features, output, target=y).sum()
            # cos_loss2 = COSloss(true_features, output, target=-y).sum()

            loss = cos_loss

            print(cos_loss.item() / batch_size, cos_loss.item() / batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % (self.steps // 2) == 0:
                print(loss.item())
                adv_images = self.tanh_space(w).detach().cpu()
                img = torch.cat([images, adv_images], dim=1).reshape((images.shape[0] * 2, -1, *images.shape[2:]))
                img = torchvision.utils.make_grid(img, nrow=6, padding=2, pad_value=1)
                show(img, 'model:vgg')

        adv_images = self.tanh_space(w).detach().cpu()
        print(adv_images.shape)
        return adv_images

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

    def __call__(self, batch_size, target_labels):

        w = torch.empty((batch_size, *self.dataset[0][0].shape), dtype=torch.float).uniform_(0.0, 1)
        w.requires_grad = True

        optimizer = optim.Adam([w], lr=self.lr)

        COSloss = nn.CosineEmbeddingLoss(reduction='none')
        CEloss = nn.CrossEntropyLoss(reduction='none')
        y = torch.tensor([1] * batch_size, dtype=torch.long).to(self.device)
        target_features = self.feature_vector[target_labels]

        for step in range(self.steps):
            adv_images = self.tanh_space(w)

            features = self.fx(adv_images)[0].flatten(start_dim=1)
            outputs = self.fx.x
            cos_loss = COSloss(target_features, features, target=y).sum()
            ce_loss = CEloss(outputs, target_labels).sum()

            loss = cos_loss * 0 + ce_loss
            if (step + 1) % (self.steps // 50) == 0:
                print(cos_loss, ce_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % (self.steps // 2) == 0:
                print(loss.item())
                adv_images = self.tanh_space(w).detach().cpu()
                img = torchvision.utils.make_grid(adv_images, nrow=6, padding=2, pad_value=1)
                show(img, 'model:vgg, loss:ce loss', path='log/mnist/reconstruction/' + str(target_labels[0]) + '.jpg')

        adv_images = self.tanh_space(w).detach().cpu()
        print(adv_images.shape)
        return adv_images


if __name__ == '__main__':

    torch.random.manual_seed(0)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root = 'E:/ljq/data'
    dataclass = MNIST(root, transform=False)
    batch_size = 6
    train_dataset, train_dataloader = \
        dataclass.get_dataloader(train=True, batch_size=batch_size, shuffle=False, num_worker=0)
    test_dataset, test_dataloader = \
        dataclass.get_dataloader(train=False, batch_size=batch_size, shuffle=False, num_worker=0)

    model = resnet_mnist()
    log_path = os.path.join('./log', dataclass.name, 'model')
    load_model(model, log_path, model.name.split('-')[0])
    # model = add_normal_layer(model, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    model = model.eval().to(device)
    print(model)

    # evaluate(model, test_dataloader)

    target_module = ['avgpool']
    fx = FeatureExtractor(model, target_module)

    # 计算训练样本的feature
    labels = torch.zeros((10, ), dtype=torch.int)
    features = [[] for i in range(10)]

    for idx, (imgs, ls) in enumerate(train_dataloader):
        imgs = imgs.to(device)
        ls = ls.to(device)

        if torch.all(torch.ge(labels, 10)):
            break

        if len(imgs) == 0:
            continue

        _, pred = torch.max(model(imgs), dim=1)
        imgs = imgs[ls == pred]
        ls = ls[ls == pred]

        ft = fx(imgs)[0].flatten(start_dim=1)
        for i in range(10):
            features[i].append(ft[ls == i])
            labels[i] += torch.sum((ls == i).int())

    for i in range(10):
        features[i] = torch.mean(torch.cat(features[i], dim=0), dim=0)
        print(features[i].shape)
    features = torch.stack(features, dim=0).detach().requires_grad_(False)

    myattack = MyAttack(model, eps=0.3, dataset=train_dataset, fx=fx, feature_vector=features)

    # for idx, (imgs, labels) in enumerate(train_dataloader):
    #     # imgs = imgs[-2:-1]
    #     # labels = labels[-2:-1]
    #     print(labels)
    #     target_labels = torch.tensor([0] * imgs.shape[0], dtype=torch.long).to(device)
    #     print(model(imgs))
    #     adv_imgs = myattack.forward(imgs, labels, target_labels)
    #     print(model(adv_imgs))
    #     plt.hist((adv_imgs - imgs).abs().flatten().detach().clone().numpy(), log=True)
    #     plt.show()
    #     break
    batch_size = 2
    for i in range(10):
        target_labels = torch.tensor([i] * batch_size, dtype=torch.long).to(device)
        myattack(batch_size=batch_size, target_labels=target_labels)











