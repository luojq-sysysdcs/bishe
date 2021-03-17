#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/27 11:22
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : TSNE.py
# @Software: PyCharm
import time
from utils import *
from GenerateData import *
import os
from models.SimpleModel import *
from sklearn import datasets
from sklearn.manifold import TSNE
from Visualizing import FeatureExtractor
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(dpi=300)
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i] if label[i] < 10 else label[i] - 10),
                 color=plt.cm.tab20b(label[i] / 20.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    data, label, n_samples, n_features = get_data()
    print(len(data[0]))
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time.time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time.time() - t0))
    fig.show()


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root = 'E:/ljq/data'
    dataclass = CIFAR(root, transform=False)
    batch_size = 32
    train_dataset, train_dataloader = \
        dataclass.get_dataloader(train=True, batch_size=batch_size, shuffle=False, num_worker=0)
    test_dataset, test_dataloader = \
        dataclass.get_dataloader(train=False, batch_size=batch_size, shuffle=False, num_worker=0)

    # root = './log/PGD-model3-0.2'
    root = './log/cifar/pgd/resnet-0.03'
    adversarial_dataclass = CIFAR(root, load=True, transform=False)
    adversarial_dataset, adversarial_dataloader = adversarial_dataclass.get_dataloader(adversarial=True)

    model = resnet_cifar()
    log_path = './log/cifar/model'
    load_model(model, log_path, 'resnet')
    model = add_normal_layer(model, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    model = model.eval().to(device)
    print(model)

    # evaluate(model, test_dataloader)
    # evaluate(model, adversarial_dataloader)

    target_module = ['1', 'model', 'avgpool']
    fx = FeatureExtractor(model, target_module)


    # 计算训练样本的feature
    labels = torch.zeros((10, ), dtype=torch.int)
    features = [[] for i in range(10)]

    for idx, (imgs, ls) in enumerate(train_dataloader):
        print(idx)
        imgs = imgs.to(device)
        ls = ls.to(device)

        if torch.all(torch.ge(labels, 100)):
            break

        if len(imgs) == 0:
            continue

        _, pred = torch.max(model(imgs), dim=1)
        imgs = imgs[ls == pred]
        ls = ls[ls == pred]

        ft = fx(imgs)[0].flatten(start_dim=1).detach()
        for i in range(10):
            features[i].append(ft[ls == i])
            labels[i] += torch.sum((ls == i).int())

    for i in range(10):
        features[i] = torch.cat(features[i], dim=0).detach().numpy()
        print(features[i].shape)
        # num = len(features[i])
        # examples[i], features[i] = features[i][:num//2], features[i][num//2:]

    # 计算对抗样本的feature
    adv_features = [[] for i in range(10)]
    adv_labels = [0 for i in range(10)]

    for idx, (imgs, _, ls) in enumerate(adversarial_dataloader):
        if len(imgs) == 0:
            continue
        ft = fx(imgs)[0].flatten(start_dim=1).detach()
        for i in range(10):
            adv_features[i].append(ft[ls == i])
            adv_labels[i] += torch.sum((ls == i).int())

    for i in range(10):
        adv_features[i] = torch.cat(adv_features[i], dim=0).detach().numpy()
        print(adv_features[i].shape)

    # for i in range(5):
    #     cos_sim = np.array(cosine_similarity(examples[i], adv_features[i]))
    #     plt.hist(cos_sim.reshape(-1), range=(0, 1), bins=100, log=True)
    #     plt.show()
    #
    #     cos_sim = np.array(cosine_similarity(examples[i], features[i]))
    #     plt.hist(cos_sim.reshape(-1), range=(0, 1), bins=100, log=True)
    #     plt.show()

    # 计算测试集的feature
    # test_features = [[] for i in range(10)]
    #
    # for idx, (imgs, ls) in enumerate(test_dataloader):
    #     ft = fx(imgs)[0].flatten(start_dim=1).detach()
    #     for i in range(10):
    #         test_features[i].append(ft[ls == i])
    #
    # for i in range(10):
    #     test_features[i] = torch.cat(test_features[i], dim=0).detach().numpy()
    #     print(test_features[i].shape)

    # 计算对抗样本到feature之间的相似度
    cos_sim = np.zeros((10, 10))
    pred = np.zeros((10, 10), dtype=np.int32)
    adv_count = 0

    for j in range(10):
        pop = np.zeros((10, len(adv_features[j])), dtype=np.float32)
        for i in range(10):
            sim = cosine_similarity(features[i], adv_features[j])
            cos_sim[j][i] = np.nanmean(sim)
            pop[i, :] = np.nanmean(sim, axis=0)

        index = np.nanargmax(pop, axis=0)
        for i in range(10):
            pred[j][i] = np.nansum(index == i)

    print(pred)
    print(cos_sim)
    plt.imshow(pred)
    plt.show()
    plt.imshow(cos_sim)
    plt.show()
    #
    # # 计算测试样本到feature之间的相似度
    # cos_sim = np.zeros((10, 10))
    # pred = np.zeros((10, 10), dtype=np.int32)
    # adv_count = 0
    #
    # for j in range(10):
    #     pop = np.zeros((10, len(test_features[j])), dtype=np.float32)
    #     for i in range(10):
    #         sim = cosine_similarity(features[i], test_features[j])
    #         cos_sim[j][i] = np.nanmean(sim)
    #         pop[i, :] = np.nanmean(sim, axis=0)
    #
    #     index = np.nanargmax(pop, axis=0)
    #     for i in range(10):
    #         pred[j][i] = np.nansum(index == i)
    #
    # print(pred)
    # print(cos_sim)
    # plt.imshow(pred)
    # plt.show()
    # plt.imshow(cos_sim)
    # plt.show()

    # 用余弦相似度分类
    # distribution = np.empty((10, 100), dtype=np.float)
    # for i in range(10):
    #     cos_sim = cosine_similarity(features[i][:100], test_features[i])
    #     distribution[i] = np.nanmean(cos_sim, axis=1)

    # acc = 0
    # pred = np.zeros((10, 10), dtype=np.int32)
    # for j in range(10):
    #     pop = np.zeros((10, len(test_features[j])), dtype=np.float32)
    #     for i in range(10):
    #         cos_sim = cosine_similarity(features[i][:100], test_features[j])
    #         kl = entropy(distribution[i].reshape(-1, 1).repeat(cos_sim.shape[1], axis=1), cos_sim, axis=0)
    #         pop[i, :] = kl
    #     index = np.nanargmin(pop, axis=0)
    #     for i in range(10):
    #         pred[j][i] = np.nansum(index == i)
    # plt.imshow(pred)
    # plt.show()
    # print(pred)

    # pred = np.zeros((10, 10), dtype=np.int32)
    # for j in range(10):
    #     pop = np.zeros((10, len(adv_features[j])), dtype=np.float32)
    #     for i in range(10):
    #         cos_sim = cosine_similarity(features[i][:100], adv_features[j])
    #         kl = entropy(distribution[i].reshape(-1, 1).repeat(cos_sim.shape[1], axis=1), cos_sim, axis=0)
    #         pop[i, :] = kl
    #     index = np.nanargmin(pop, axis=0)
    #     for i in range(10):
    #         pred[j][i] = np.nansum(index == i)
    # plt.imshow(pred)
    # plt.show()
    # print(pred)




    # count = 0
    # for idx, (imgs, tl, al) in enumerate(adversarial_dataloader):
    #     output = model(imgs)
    #     pre = torch.max(output, dim=1)[1]
    #     count += torch.sum((pre==tl).float())
    # print(count / len(adversarial_dataset))

    # features = []
    # labels = []
    # images = []
    # choice = 10
    # fx = FeatureExtractor()
    # target_module = ['avgpool']
    # for idx, (imgs, ls) in enumerate(train_dataloader):
    #     if len(labels) >= 500:
    #         break
    #     wanted = ls < choice
    #     # wanted = torch.ones(imgs.shape[0], dtype=torch.bool)
    #     imgs = imgs[wanted]
    #     ls = ls[wanted]
    #     images.append(imgs.flatten(1))
    #     labels += list(ls)
    #     feature = fx(model, imgs, target_module)[0].flatten(start_dim=1)
    #     # feature = model(imgs)
    #     features.append(feature)
    # features = torch.cat(features, 0).detach().cpu().numpy()
    # images = torch.cat(images, 0).detach().cpu().numpy()
    # labels = np.array(labels)
    # print(images.shape)
    # print(labels.shape)
    # print(features.shape)
    #
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # print('Computing t-SNE embedding')
    # result = tsne.fit_transform(features)
    # fig = plot_embedding(result, labels,
    #                      't-SNE embedding of the digits')
    # fig.show()
    # exit(0)
# -----------------------------------------
#     features = []
#     true_labels = []
#     adversarial_labels = []
#     images = []
#     fx = FeatureExtractor()
#     target_module = ['avgpool']
#     for idx, (imgs, tl, al) in enumerate(adversarial_dataloader):
#         if idx == 10:
#             break
#         images.append(imgs.flatten(1))
#         true_labels.append(tl)
#         adversarial_labels.append(al)
#         feature = fx(model, imgs, target_module)[0].flatten(start_dim=1)
#         features.append(feature)
#     images = torch.cat(images, 0).detach().cpu().numpy()
#     true_labels = torch.cat(true_labels, 0).detach().cpu().numpy()
#     adversarial_labels = torch.cat(adversarial_labels, 0).detach().cpu().numpy()
#     features = torch.cat(features, 0).detach().cpu().numpy()
#     print(images.shape)
#     print(true_labels.shape)
#     print(adversarial_labels.shape)
#     print(features.shape)
#
#     tsne = TSNE(n_components=2, init='pca', random_state=0)
#     print('Computing t-SNE embedding')
#     t0 = time()
#     result = tsne.fit_transform(features)
#     fig = plot_embedding(result, true_labels,
#                          't-SNE embedding of the digits (time %.2fs)'
#                          % (time() - t0))
#     fig.show()

# -----------------------------------------

# true_labels = []
# true_images = []
# true_features = []
# adversarial_labels = []
# adversarial_images = []
# adversarial_features = []
# fx = FeatureExtractor()
# target_module = ['avgpool']
# choice = 10
# for idx, (imgs, tl, al) in enumerate(adversarial_dataloader):
#     if len(adversarial_labels) >= 500:
#         break
#     wanted = al < choice
#     # wanted = torch.ones(imgs.shape[0], dtype=torch.bool)
#     imgs = imgs[wanted]
#     al = al[wanted]
#     adversarial_images.append(imgs.flatten(1))
#     adversarial_labels += list(al)
#     feature = fx(model, imgs, target_module)[0].flatten(start_dim=1)
#     # feature = model(imgs)
#     adversarial_features.append(feature)
#
# for idx, (imgs, ls) in enumerate(train_dataloader):
#     if len(true_labels) >= 500:
#         break
#     wanted = ls < choice
#     # wanted = torch.ones(imgs.shape[0], dtype=torch.bool)
#     imgs = imgs[wanted]
#     ls = ls[wanted]
#     true_images.append(imgs.flatten(1))
#     true_labels += list(ls)
#     feature = fx(model, imgs, target_module)[0].flatten(start_dim=1)
#     # feature = model(imgs)
#     true_features.append(feature)
#
# count = min(len(adversarial_labels), len(true_labels))
# adversarial_features = torch.cat(adversarial_features, 0).detach().cpu().numpy()[:count].reshape(count, -1)
# adversarial_images = torch.cat(adversarial_images, 0).detach().cpu().numpy()[:count].reshape(count, -1)
# adversarial_labels = np.array(adversarial_labels)[:count]
# true_features = torch.cat(true_features, 0).detach().cpu().numpy()[:count].reshape(count, -1)
# true_images = torch.cat(true_images, 0).detach().cpu().numpy()[:count].reshape(count, -1)
# true_labels = np.array(true_labels)[:count]
# print(adversarial_images.shape)
# print(adversarial_labels.shape)
# print(adversarial_features.shape)
# print(true_images.shape)
# print(true_labels.shape)
# print(true_features.shape)

# diff = adversarial_features - true_features
# # plt.bar(range(10), np.mean(diff, axis=0))
# width = 0.3
# plt.bar(range(10), np.mean(adversarial_features, axis=0), width=width, label='adv')
# plt.bar(np.arange(10) + width, np.mean(true_features, axis=0), width=width, label='clean')
# plt.title(str(choice))
# plt.legend()
# plt.show()

# tsne = TSNE(n_components=2, init='pca', random_state=0)
# print('Computing t-SNE embedding')
# t0 = time.time()
# result = tsne.fit_transform(np.concatenate((adversarial_features, true_features), axis=0))
# t1 = time.time()
# print('used time: %d' % (t1 - t0))
# file_name = '-'.join((root + '-' + model.name).split('/')[-2:])
# print(file_name)
# fig = plot_embedding(result, np.concatenate((adversarial_labels + 10, true_labels), axis=0), file_name)
# tsne_path = './log/mnist/tsne'
# if not os.path.exists(tsne_path):
#     os.makedirs(tsne_path)
# plt.savefig(os.path.join(tsne_path, file_name + '.jpg'))
# plt.show()
# ------------------------------------------------
#     root = 'E:\ljq\data'
#     batch_size = 1
#     train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True, batch_size=batch_size, shuffle=False)
#
#     root = './log/PGD'
#     adversarial_dataset, adversarial_dataloader = get_adversarial_data(root, batch_size=batch_size, shuffle=False)
#
#     true_labels = []
#     true_images = []
#     true_features = []
#     adversarial_labels = []
#     adversarial_images = []
#     adversarial_features = []
#     fx = FeatureExtractor()
#     target_module = ['linear']
#     choice = 1
#     choice_ = 0
#
#     for idx, (imgs, ls) in enumerate(train_dataloader):
#         if idx > 1000:
#             break
#         if ls.item() != choice:
#             continue
#         img_, _, _ = adversarial_dataset.get_sample(idx, choice_)
#         if img_ is None:
#             continue
#
#         img_ = img_.unsqueeze(0)
#         adversarial_images.append(img_.flatten(1))
#         feature = fx(model, img_, target_module)[0]
#         adversarial_features.append(feature)
#         adversarial_labels.append(choice_)
#
#         true_images.append(imgs.flatten(1))
#         true_labels += list(ls)
#         feature = fx(model, imgs, target_module)[0]
#         true_features.append(feature)
#
#     adversarial_features = torch.cat(adversarial_features, 0).detach().cpu().numpy()
#     adversarial_images = torch.cat(adversarial_images, 0).detach().cpu().numpy()
#     adversarial_labels = np.array(adversarial_labels)
#     true_features = torch.cat(true_features, 0).detach().cpu().numpy()
#     true_images = torch.cat(true_images, 0).detach().cpu().numpy()
#     true_labels = np.array(true_labels)
#     print(adversarial_images.shape)
#     print(adversarial_labels.shape)
#     print(adversarial_features.shape)
#     print(true_images.shape)
#     print(true_labels.shape)
#     print(true_features.shape)
#
#     diff_features = adversarial_features - true_features
#     print(np.mean((true_features), axis=1))
#     print(np.mean((adversarial_features), axis=1))
#     print(np.mean((diff_features), axis=1))
#     print(np.count_nonzero(np.abs(diff_features) < 0.1))
#     plt.bar(range(100), np.mean((true_features), axis=0))
#     plt.show()
# plt.bar(range(100), np.mean(true_features, axis=0))
# # plt.hist(diff_features.flatten(), bins=1000)
# # plt.hist(true_features.flatten(), bins=100)
# plt.show()


# -----------------------------------------
#     print(model.linear.weight)
#     plt.hist(model.linear.weight.detach().numpy().flatten(), bins=100)
#     plt.show()
