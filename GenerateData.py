#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/1/14 10:39
# @Author  : luojq
# @Email   : luojq_sysusdcs@163.com
# @File    : GenerateData.py
# @Software: PyCharm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import random

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)


def make_dataset(path):
    if os.path.exists(os.path.join(path, 'labels.txt')):
        labels = np.loadtxt(os.path.join(path, 'labels.txt'), dtype=np.long)
    else:
        labels = None
    instances = []
    for root, dirs, files in sorted(os.walk(path)):
        dirs = sorted(dirs, key=lambda x: int(x))
        for idx, dir in enumerate(dirs):
            if labels is None:
                label = -1
            else:
                label = labels[idx]
            for r, _, fnames in os.walk(os.path.join(path, dir)):
                random.shuffle(fnames)  # pay attention
                for fname in fnames:
                    if not fname.split('.')[0].isdigit():
                        continue
                    p = os.path.join(r, fname)
                    adv_label = int(fname.split('.')[0])
                    if is_image_file(p):
                        if label == adv_label:
                            print(idx, label, adv_label)
                            raise ValueError
                        item = p, label, adv_label
                        instances.append(item)
                    # break # only load one adversarial image for one clean image
                break
        break
    return instances, labels


def pil_loader(path: str, mode: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path, mode='RGB')


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path, mode='RGB')


class AdversarialDataset():
    def __init__(self, root, transform=None, target_transform=None, mode='RGB', load=False):
        self.samples, self.true_labels = make_dataset(root)
        self.loader = pil_loader
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.mode = mode
        self.load = load

        if load:
            for i in range(len(self.samples)):
                path, tl, al = self.samples[i]
                self.samples[i] = (self.loader(path, self.mode), tl, al)

    def __getitem__(self, index):
        path, tl, al = self.samples[index]
        if not self.load:
            sample = self.loader(path, self.mode)
        else:
            sample = path
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return sample, tl, al

    def get_sample(self, idx, choice):
        path = os.path.join(self.root, str(idx), str(choice) + '.jpg')
        if not os.path.exists(path):
            return None, None, None
        else:
            sample = self.loader(path, self.mode)
            if self.transform is not None:
                sample = self.transform(sample)
            # if self.target_transform is not None:
            #     target = self.target_transform(target)
            if self.true_labels is not None:
                return sample, self.true_labels[idx], choice
            else:
                return sample, None, choice

    def __len__(self):
        return min(len(self.samples), 100000)


# def get_adversarial_data(root, batch_size=64, shuffle=False):
#     print('getting adversarial dataset...')
#     transform = transforms.Compose([transforms.ToTensor(), ])
#     adversarial_dataset = MyDataSet2(root, transform=transform)
#     adversarial_dataloader = DataLoader(adversarial_dataset,
#                                         batch_size=batch_size, shuffle=shuffle, num_workers=0)
#     print('num of data:', len(adversarial_dataset))
#     print('num of batch', len(adversarial_dataloader))
#     return adversarial_dataset, adversarial_dataloader


class CIFAR():
    def __init__(self, root, transform=False, load=False):
        self.name = 'cifar'
        self.root = root
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        self.load = load
        if transform:
            self.train_transform = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
                # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def get_dataset(self, train=True, adversarial=False):
        if adversarial:
            dataset = AdversarialDataset(self.root, transform=self.test_transform, mode='RGB', load=self.load)
        elif train:
            dataset = datasets.CIFAR10(self.root, train=train, transform=self.train_transform, download=False)
        else:
            dataset = datasets.CIFAR10(self.root, train=train, transform=self.test_transform, download=False)
        return dataset

    def get_dataloader(self, train=True, batch_size=64, shuffle=False, num_worker=4, adversarial=False,
                       pin_memory=False):
        dataset = self.get_dataset(train, adversarial=adversarial)
        print('loading...')
        print('num of data: %d' % len(dataset))
        dataloader = DataLoader(dataset, shuffle=shuffle,
                                batch_size=batch_size,
                                num_workers=num_worker,
                                pin_memory=False)
        print('num of batch: %d' % len(dataloader))
        return dataset, dataloader

    def unnormalize(self, tensor, inplace=False):
        if self.mean is None and self.std is None:
            return tensor
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))
        if tensor.ndim < 3:
            raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                             '{}.'.format(tensor.size()))
        if not inplace:
            tensor = tensor.clone().detach()
        dtype = tensor.dtype
        mean = torch.tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean).clamp_(min=0, max=1)
        return tensor

    def save(self, tensor, path):
        """
        将tensor保存为pillow
        :param input_tensor: 要保存的tensor
        :param filename: 保存的文件名
        """
        assert (len(tensor.shape) == 3)
        tensor = tensor.clone().detach().to(torch.device('cpu'))
        tensor = tensor.mul_(255).add_(0.5).clamp(0, 255).permute(1, 2, 0).squeeze().type(torch.uint8).numpy()
        im = Image.fromarray(tensor)
        im.save(path)


class MNIST():
    def __init__(self, root, transform=False, load=False):
        self.name = 'mnist'
        self.root = root
        self.mean = None
        self.std = None
        self.load = load
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def get_dataset(self, train=True, adversarial=False):
        if adversarial:
            dataset = AdversarialDataset(self.root, transform=self.test_transform, mode='L', load=self.load)
        elif train:
            dataset = datasets.MNIST(self.root, train=train, transform=self.train_transform, download=False)
        else:
            dataset = datasets.MNIST(self.root, train=train, transform=self.test_transform, download=False)
        return dataset

    def get_dataloader(self, train=True, batch_size=64, shuffle=False, num_worker=4, adversarial=False,
                       pin_memory=False):
        dataset = self.get_dataset(train, adversarial=adversarial)
        print('loading...')
        print('num of data: %d' % len(dataset))
        dataloader = DataLoader(dataset,
                                shuffle=shuffle,
                                batch_size=batch_size,
                                num_workers=num_worker,
                                pin_memory=pin_memory)
        print('num of batch: %d' % len(dataloader))
        return dataset, dataloader

    def unnormalize(self, tensor, inplace=False):
        if self.mean is None and self.std is None:
            return tensor
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))
        if tensor.ndim < 3:
            raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                             '{}.'.format(tensor.size()))
        if not inplace:
            tensor = tensor.clone().detach()
        dtype = tensor.dtype
        mean = torch.tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean).clamp_(min=0, max=1)
        return tensor

    def save(self, tensor, path):
        """
        将tensor保存为pillow
        :param input_tensor: 要保存的tensor
        :param filename: 保存的文件名
        """
        assert (len(tensor.shape) == 3)
        tensor = tensor.clone().detach().to(torch.device('cpu'))
        tensor = tensor.mul_(255).add_(0.5).clamp(0, 255).permute(1, 2, 0).squeeze().type(torch.uint8).numpy()
        im = Image.fromarray(tensor)
        im.save(path)


# def generate_data(root, name, train=True, transform=None, batch_size=None, shuffle=True, shuffle_label=False):
#     if name == 'MNIST':
#         loader = datasets.MNIST
#     elif name == 'CIFAR':
#         loader = datasets.CIFAR10
#     else:
#         raise NotImplementedError
#     if transform is None:
#         transform_train = transforms.Compose([transforms.ToTensor(),])
#         transform_test = transforms.Compose([transforms.ToTensor(),])
#     else:
#         transform_train = transform
#         transform_test = transform
#     if name == 'CIFAR':
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
#             transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
#         ])
#
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])
#     if batch_size is None:
#         batch_size = 64
#     if train:
#         train_dataset = loader(root=root, train=True, download=True, transform=transform_train)
#         if shuffle_label:
#             print('train label shuffled!')
#             torch.random.manual_seed(0)
#             train_dataset.targets = torch.randint(0, 10, size=(len(train_dataset),))
#         print('num of data:', len(train_dataset))
#         train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
#         print('num of batch:', len(train_dataloader))
#         return train_dataset, train_dataloader
#
#     else:
#         test_dataset = loader(root=root, train=False, download=True, transform=transform_test)
#         print('num of data:', len((test_dataset)))
#         if shuffle_label:
#             print('test label shuffled!')
#             torch.random.manual_seed(0)
#             test_dataset.targets = torch.randint(0, 10, size=(len(test_dataset),))
#         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#         print('num of batch', len(test_dataloader))
#         return test_dataset, test_dataloader


if __name__ == '__main__':
    root = 'E:/ljq/data'
    dataclass = CIFAR(root)
    batch_size = 64
    train_dataset, train_dataloader = dataclass.get_dataloader(train=True, batch_size=batch_size, shuffle=False,
                                                               num_worker=4)
    test_dataset, test_dataloader = dataclass.get_dataloader(train=False, batch_size=batch_size, shuffle=False,
                                                             num_worker=4)
    print(torch.min(dataclass.unnormalize(train_dataset[0][0])))
