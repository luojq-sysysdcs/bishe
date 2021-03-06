#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/7 21:15
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : perform_attack.py
# @Software: PyCharm


import torchvision
from utils import *
from GenerateData import *
import os
from PIL import Image
import numpy as np
import time
from models.SimpleModel import *
from attack import *
import torch
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root = 'E:/ljq/data'
    dataclass = CIFAR(root, transform=False)
    batch_size = 6
    train_dataset, train_dataloader = \
        dataclass.get_dataloader(train=True, batch_size=batch_size, shuffle=False, num_worker=0)
    test_dataset, test_dataloader = \
        dataclass.get_dataloader(train=False, batch_size=batch_size, shuffle=False, num_worker=0)

    model = vgg_cifar()
    log_path = os.path.join('./log', dataclass.name, 'model')
    load_model(model, log_path, model.name.split('-')[0])
    model = add_normal_layer(model, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    model = model.eval().to(device)
    print(model)
    
    # for i in range(10):
    #     acc = cal_acc(model, test_dataloader, device, label=i)
    #     print('accuracy: %f' % acc)
    # exit(0)

    # adversary = FGSM(model, eps=0.03, random_start=False)
    adversary = PGD(model, eps=0.031, alpha=2 / 255, steps=10, random_start=False)
    # adversary = MFGSM(model, eps=0.031, c=10)
    # adversary = CW(model, c=1e1, kappa=0, steps=50, lr=1e-1, use_cuda=torch.cuda.is_available())
    num_classes = 10
    # adversary = DeepFool(model, num_classes=num_classes, steps=50)

    root = os.path.join('./log', dataclass.name, 'pgd', 'vgg-0.03')
    # if os.path.exists(root):
    #     shutil.rmtree(root)
    # if not os.path.exists(root):
    #     os.makedirs(root)

    count = 0
    index = 0
    target_attack = True
    true_labels = []
    for idx, (images, labels) in enumerate(train_dataloader):
        if idx * batch_size >= 1000:
            break
        # if idx * batch_size >= 9000:
        #     break
        true_labels += list(labels)
        if target_attack:
            adv_labels = [[i for i in range(num_classes)] for _ in range(batch_size)]
            for i, label in enumerate(labels):
                del adv_labels[i][label]
        else:
            adv_labels = labels.unsqueeze(1).numpy()
        adv_labels = torch.tensor(adv_labels, dtype=torch.long)
        print(list(labels), list(adv_labels))

        for i in range(batch_size):
            p = os.path.join(root, str(index + i))
            if not os.path.exists(p):
                os.makedirs(p)

        for i in range(adv_labels.shape[1]):
            t0 = time.time()
            adv_images, success = adversary.forward(images, adv_labels[:, i])
            adv_images = adv_images.cpu()
            print(success)

            for j in range(batch_size):
                if success[j]:
                    count += 1
                    p = os.path.join(root, str(index + j), str(adv_labels[:, i][j].item()) + '.jpg')
                    # dataclass.save(adv_images[j], p)
            if idx < 5:
                img = torch.cat([images, adv_images], dim=1)
                img = img.reshape((batch_size * 2, -1, *img.shape[-2:]))
                img = torchvision.utils.make_grid(img, nrow=6, padding=2, pad_value=1)
                show(img)  #, os.path.join(root, str(idx) + '-' + str(i))
            t1 = time.time()
            print('time used: %f' % (t1 - t0))

            break

        index += len(labels)

        if idx % 1 == 0:
            if target_attack:
                print('success rate : %d / %d ( %f ) ' % (count, ((num_classes - 1) * (idx + 1) * batch_size),
                                                          count / ((num_classes - 1) * (idx + 1) * batch_size)))
            else:
                print('success rate : %d / %d ( %f ) ' % (count, ( (idx + 1) * batch_size),
                                                          count / ((idx + 1) * batch_size)))
        break
    # np.savetxt(os.path.join(root, 'labels.txt'), true_labels, fmt="%d")

    # 生成随机攻击样本
    # torch.random.manual_seed(0)
    # adversary = PGD(model, eps=0.3, alpha=5 / 255, steps=20, random_start=False)
    #
    # root = os.path.join('./log', 'random', 'pgd', 'vgg-0.03')
    # if os.path.exists(root):
    #     shutil.rmtree(root)
    # if not os.path.exists(root):
    #     os.makedirs(root)
    #
    # batch_size = 50
    # fake_labels = torch.arange(0, 10, step=1, dtype=torch.long).unsqueeze(1).repeat((1, batch_size))
    #
    # for i in range(100):
    #     random_data = torch.empty((batch_size, 3, 32, 32), dtype=torch.float).random_(0, 2)
    #     for j in range(10):
    #         adv_data, success = adversary.forward(random_data, fake_labels[j])
    #         print(success)
    #         for k in range(batch_size):
    #             if success[k]:
    #                 p = os.path.join(root, str(i * batch_size + k) + '-' + str(j) + '.jpg')
    #                 save_image(adv_data[k], p)
    #         if i == 0:
    #             img = torch.cat([random_data, adv_data], dim=1)
    #             img = img.reshape((batch_size * 2, -1, *img.shape[-2:]))
    #             img = torchvision.utils.make_grid(img, nrow=10, padding=2, pad_value=1)
    #             show(img)  #, os.path.join(root, str(idx) + '-' + str(i))


# -----------------------------
# labels = []
# success = 0
# count = 1000
# for idx, (image, label) in enumerate(train_dataloader):
#     if idx * batch_size >= 1000:
#         continue
#     labels.append(label.item())
#     if not os.path.exists(os.path.join(root, str(idx))):
#         os.makedirs(os.path.join(root, str(idx)))
#     # if torch.max(model(image), dim=1)[1].item() != label:
#     #     continue
#     for target_label in range(10):
#         if target_label == label:
#             continue
#         adv_label = torch.tensor([target_label], dtype=torch.long)
#         adv_image = adversary.forward(image,adv_label)
#         t = (adv_image.detach().clone()*255).to(torch.uint8) / 255
#         after_label = torch.max(model(t), 1)[1].squeeze()
#         if after_label.item() != target_label:
#             continue
#         success += 1
#         t = np.uint8((t.squeeze() * 255).numpy())
#         adv_image = Image.fromarray(t)
#         adv_image = adv_image.convert('L')
#         adv_image.save(os.path.join(root, str(idx), str(target_label)+'.jpg'))
#     if idx % 10 == 0:
#         print('success rate: %f (%d / %d)' % (success / ((idx + 1) * 9), success, (idx + 1) * 9))
# print('-'*20)
# print(labels)
# print('success rate: %f (%d / %d)' % (success / (count * 9), success, count * 9))
# np.savetxt(os.path.join(root, 'labels.txt'), labels, fmt="%d")
# -----------------------------
# -----------------------------------
# idx = 80
# image, label = train_dataset[idx][0].unsqueeze(0), torch.tensor(train_dataset[idx][1]).unsqueeze(0)
# output = model(image)
# pred = torch.max(output, 1)[1].squeeze()
# print('true label:%d, predict label:%d' % (label, pred))
#
# for i in range(10):
#     adv_label = torch.tensor([i], dtype=torch.long)
#     print(adv_label)
#     adv_image = adversary.forward(image,adv_label)
#     after_label = torch.max(model(adv_image), 1)[1].squeeze()
#     print('true label:%d, after attacking predict label:%d' % (label, after_label))
#     plt.imshow(adv_image.squeeze().detach().cpu().numpy())
#     plt.show()

# adv_image = adv_image.squeeze(0).permute(1, 2, 0).numpy()
# image = image.squeeze(0).permute(1, 2, 0).numpy()
# print(adv_image.shape)
# ax1 = plt.subplot(121)
# ax2 = plt.subplot(122)
# ax1.imshow(image)
# ax1.set_title('label:5')
# ax2.imshow(adv_image)
# ax2.set_title('label:3')
# plt.show()
