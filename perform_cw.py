#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/9 22:21
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : perform_cw.py
# @Software: PyCharm

import torchvision
from utils import *
from CW import CW
from GenerateData import *
from matplotlib import pyplot as plt
import os
from PIL import Image
import time


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root = 'E:/ljq/data'
    batch_size = 50
    train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=batch_size, shuffle=False)

    model = Model3()
    model = model.eval()
    log_path = './log'
    load_model(model, log_path, 'model3')
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(model)

    # model = SM()
    # log_path = '.\\log'
    # load_model(model, log_path, 'simple_model')
    # print(model)

    adversary = CW(model, c=1e1, kappa=0, steps=500, lr=1e-1, use_cuda=torch.cuda.is_available())

    index = 0
    true_labels = []
    root = './log/cw/model4-1e1-0'
    if not os.path.exists(root):
        os.makedirs(root)

    count = 0
    for idx, (images, labels) in enumerate(train_dataloader):
        if idx * batch_size >= 1000:
            break
        # adv_images = adversary.forward(images)
        true_labels += list(labels)
        adv_labels = [[i for i in range(10)] for _ in range(batch_size)]
        for i, label in enumerate(labels):
            del adv_labels[i][label]
        print(labels)
        print(adv_labels)
        adv_labels = torch.tensor(adv_labels, dtype=torch.long)

        for i in range(len(labels)):
            p = os.path.join(root, str(index + i))
            if not os.path.exists(p):
                os.makedirs(p)

        for i in range(9):
            t0 = time.time()
            adv_images, success, L2 = adversary.forward(images, adv_labels[:, i])
            adv_images = adv_images.cpu()
            for j in range(len(labels)):
                if success[j]:
                    count += 1
                    img = np.uint8(adv_images[j].cpu().clone().detach().numpy().squeeze() * 255)
                    img = Image.fromarray(img)
                    img = img.convert('L')
                    p = os.path.join(root, str(index + j), str(adv_labels[:, i][j].item()) + '.jpg')
                    # img.save(p)
            if idx < 2:
                img = torch.cat([images, adv_images], dim=1)
                img = img.reshape((-1, 1, *img.shape[-2:]))
                img = torchvision.utils.make_grid(img, nrow=10, padding=2, pad_value=1)
                show(img)
            t1 = time.time()
            print('time used: %f' % (t1 - t0))
        index += len(labels)


        if idx % 1 == 0:
            print('success rate : %f' % (count / (9 * (idx + 1) * batch_size)))
    # np.savetxt(os.path.join(root, 'labels.txt'), true_labels, fmt="%d")






