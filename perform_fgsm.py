#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/7 21:15
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : perform_fgsm.py
# @Software: PyCharm


import torchvision
from utils import *
from PGD import PGD
from GenerateData import *
import os
from PIL import Image
import numpy as np
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == '__main__':
    root = 'E:/ljq/data'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 100
    train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=batch_size, shuffle=False)

    # model = SM()
    # model = ConvModel()
    # log_path = '.\\log'
    # load_model(model, log_path, 'conv_model')
    # model = model.eval()
    # print(model)

    model = Model4()
    log_path = './log'
    load_model(model, log_path, 'model4')
    model = model.eval().to(device)
    print(model)

    # adversary = FGSM(model, eps=0.15)
    adversary = PGD(model, eps=0.3, alpha=1 / 255, steps=200, random_start=False)

    root = './log/pgd/model4-0.3'
    if not os.path.exists(root):
        os.makedirs(root)

    count = 0
    index = 0
    true_labels = []
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
            adv_images, success = adversary.forward(images, adv_labels[:, i])
            print(success)
            adv_images = adv_images.cpu()
            for j in range(len(labels)):
                if success[j]:
                    count += 1
                    img = np.uint8(adv_images[j].cpu().clone().detach().numpy().squeeze() * 255)
                    img = Image.fromarray(img)
                    img = img.convert('L')
                    p = os.path.join(root, str(index + j), str(adv_labels[:, i][j].item()) + '.jpg')
                    img.save(p)
            if idx < 2:
                img = torch.cat([images, adv_images], dim=1)
                img = img.reshape((-1, 1, *img.shape[-2:]))
                img = torchvision.utils.make_grid(img, nrow=10, padding=2, pad_value=1)
                show(img)
            t1 = time.time()
            print('time used: %f' % (t1 - t0))
        index += len(labels)

        if idx % 5 == 0:
            print('success rate : %f' % (count / (9 * (idx + 1) * batch_size)))
    np.savetxt(os.path.join(root, 'labels.txt'), true_labels, fmt="%d")

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


























