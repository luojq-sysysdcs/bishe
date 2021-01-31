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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
        dirs = sorted(dirs, key=lambda x:int(x))
        for idx, dir in enumerate(dirs):
            if labels is None:
                label = -1
            else:
                label = labels[idx]
            for r, _, fnames in os.walk(os.path.join(path, dir)):
                random.shuffle(fnames) # pay attention
                for fname in fnames:
                    p = os.path.join(r, fname)
                    if is_image_file(p):
                        if label == int(fname.split('.')[0]):
                            raise ValueError
                        item = p, label, int(fname.split('.')[0])
                        instances.append(item)
                    # break # only load one adversarial image for one clean image
                break
        break
    return instances, labels


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MyDataSet2():
    def __init__(self, root, transform=None, target_transform=None):
        self.samples, self.true_labels = make_dataset(root)
        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform
        self.root = root

    def __getitem__(self, index):
        path, tl, al = self.samples[index]
        sample = self.loader(path)
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
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            # if self.target_transform is not None:
            #     target = self.target_transform(target)
            if self.true_labels is not None:
                return sample, self.true_labels[idx], choice
            else:
                return sample, None, choice

    def __len__(self):
        return len(self.samples)

def get_adversarial_data(root, batch_size=64, shuffle=False):
    transform = transforms.Compose([transforms.ToTensor(), ])
    adversarial_dataset = MyDataSet2(root, transform=transform)
    adversarial_dataloader = DataLoader(adversarial_dataset,
                                        batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return adversarial_dataset, adversarial_dataloader

class MyDataSet():
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True,
                 shuffle_label=False, label_root=None):
        self.train = train  # training set or test set
        self.processed_folder = os.path.join(root, 'MNIST', 'processed')
        self.transform = transform
        self.target_transform = target_transform
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        if shuffle_label:
            self.targets = np.loadtxt(label_root)

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def generate_data(root, name, train=True, transform=None, batch_size=None, shuffle=True, shuffle_label=False):
    if name == 'MNIST':
        loader = datasets.MNIST
    elif name == 'CIFAR':
        loader = datasets.CIFAR10
    else:
        raise NotImplementedError
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(),])
    if batch_size is None:
        batch_size = 64
    if train:
        train_dataset = loader(root=root, train=True, download=True, transform=transform)
        if shuffle_label:
            print('train label shuffled!')
            torch.random.manual_seed(0)
            train_dataset.targets = torch.randint(0, 10, size=(len(train_dataset),))
        print('num of data:', len(train_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        print('num of batch:', len(train_dataloader))
        return train_dataset, train_dataloader

    else:
        test_dataset = loader(root=root, train=False, download=True, transform=transform)
        print('num of data:', len((test_dataset)))
        if shuffle_label:
            print('test label shuffled!')
            torch.random.manual_seed(0)
            test_dataset.targets = torch.randint(0, 10, size=(len(test_dataset),))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        print('num of batch', len(test_dataloader))
        return test_dataset, test_dataloader

# class NumpyDataset(datasets):
#     def __init__(self):

if __name__ == '__main__':
    root = 'E:\ljq\data'
    batch_size = 64
    train_dataset, train_dataloader = generate_data(root, 'MNIST', train=True,
                                                    batch_size=batch_size, shuffle=False, shuffle_label=True)


    # root = './log/PGD'
    # adversarial_dataset, train_dataloader = get_adversarial_data(root)
    # for imgs, true_labels, fake_labels in train_dataloader:
    #     print(imgs.shape)
    #     print(true_labels)
    #     break
    # print(adversarial_dataset.get_sample(0, 2))
    # root = 'E:/ljq/data'
    # train_dataset = datasets.CIFAR10(root, train=True, download=True)
    # print(train_dataset[0])









