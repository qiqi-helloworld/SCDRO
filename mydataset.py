__author__ = 'Qi'
# Created by on 12/3/21.

import os
import math
import torch
import pickle
import numpy as np
from collections import  Counter
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image



__DATASETS_DEFAULT_PATH = './data/'



class MyCIFAR10(datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root = root, train=train, transform=transform, target_transform=target_transform,
                 download=download)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target


class MyCIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root = root, train=train, transform=transform, target_transform=target_transform,
                                  download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH):
    train = (split == 'train')
    root = os.path.join(datasets_path, name)
    if name == 'cifar10':
        return MyCIFAR10(root=root,
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)

    elif name == 'cifar100':
        return MyCIFAR100(root=root,
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'mnist':
        return datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'stl10':
        return datasets.STL10(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'imagenet':
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)
    elif name == 'svhn':
        return datasets.SVHN(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name =='chexpert':
        return ChexpertSmall(root=root, mode=split, transform=transform)


def get_balanced_dataset(name, transform):
    train_data = get_dataset(name, 'train', transform['train'])
    if name == 'svhn':
        val_data = get_dataset(name, 'test', transform['eval'])
    elif name == 'chexpert':
        val_data = get_dataset(name, 'valid', transform['eval'])
    else:
        val_data = get_dataset(name, 'val', transform['eval'])
    return train_data, val_data

def get_imbalanced_dataset(name, im_ratio, transform):

    im_train_data_file = './data/' + name + '/im_train_data' + "_" + str(im_ratio)
    if not os.path.isfile(im_train_data_file):
        print('Hello')
        train_data= get_dataset(name, 'train', transform['train'])
        label_list = []
        for (_, input, label) in train_data:
            label_list.append(label)
        np_label = np.array(label_list)
        label_stats = Counter(np_label)
        saved_indexes = []
        for i in range(len(np.unique(np_label))):
            if i >= len(np.unique(np_label)) // 2:
                saved_indexes_start = math.floor(label_stats[i]* (1 - im_ratio))
                saved_indexes = saved_indexes + list(np.where(np_label == i)[0][saved_indexes_start:])
            else:
                saved_indexes = saved_indexes + list(np.where(np_label == i)[0])

        imbalanced_train_data = torch.utils.data.Subset(train_data, saved_indexes)
        print(len(imbalanced_train_data))

        f = open(im_train_data_file, 'wb')
        pickle.dump(imbalanced_train_data, f)
        f.close()
    val_data_file = './data/' + name + '/val_data'
    if not os.path.isfile(val_data_file):
        if name == 'svhn' or name == 'stl10':
            val_data = get_dataset(name, 'test', transform['eval'])
        else:
            val_data = get_dataset(name, 'val', transform['eval'])
        f = open('./data/'+ name + '/val_data', 'wb')
        pickle.dump(val_data, f)
        f.close()

    f = open('./data/' + name + '/im_train_data'+ "_" + str(im_ratio), 'rb')
    train_data = pickle.load(f)
    f.close()
    f = open('./data/' + name + '/val_data', 'rb')
    val_data = pickle.load(f)
    f.close()
    return  train_data, val_data


def get_cls_num_list(args):
    if args.dataset == 'imagenet-LT':
        train_data = LT(args.data_root, './data/ImageNet_LT/ImageNet_LT_train.txt', transform = None)
        class_dict = Counter(train_data.labels)
    elif args.dataset == 'places-LT':
        train_data = LT(args.data_root, './data/Places_LT/Places_LT_train.txt', transform = None)
        class_dict = Counter(train_data.labels)
    elif args.dataset == 'covid-LT':
        train_data = LT(args.data_root, './data/Covid_LT/' + str(args.imb_factor) + '_Covid_LT_train.txt', transform=None)
        class_dict = Counter(train_data.labels)
    elif args.dataset == 'iNaturalist18':
        train_data = LT(args.dataset, './data/iNaturalist18/iNaturalist18_train.txt', transform = None)
        class_dict = Counter(train_data.labels)

    cls_num_list = [value for key, value in sorted(class_dict.items())]
    return cls_num_list


def get_num_classes(args):
    '''
    :param args: dataset
    :return: the number of total classes in dataset
    '''
    num_classes = 0
    if args.dataset == 'ina':
        num_classes = 1010
    elif args.dataset == 'imagenet-LT':
        num_classes = 1000
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200
    elif args.dataset == 'places-LT':
        num_classes = 365
    elif args.dataset == 'covid-LT':
        num_classes = 4
    elif args.dataset == 'iNaturalist18':
        num_classes = 8142
    return num_classes


class featLT(Dataset):

    def __init__(self, root, set_image):
        self.data = np.load(root + set_image + '_feat.npy')
        self.labels = np.load(root + set_image + '_label.npy')
        # print(self.labels.shape, self.data.shape)
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return index, torch.tensor(self.data[index]), torch.tensor(self.labels[index])


class indexCIFARDatasets(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        # transform
        # self.transform = transform
        # print(self.labels.shape, self.data.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        img =  torch.tensor(self.data[index])
        label = torch.tensor(self.labels[index])
        # print(">>>>:", index, img.shape)
        # img = Image.fromarray(img)
        # print("<<<<<<:", type(img))
        # if self.transform is not None:
        #     img = self.transform(img)
        return index, img, label



class LT(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

        # print("LT:", txt)
        # if 'test' in txt and 'ImageNet' in txt:
        #     with open(txt) as f:
        #         for line in f:
        #             img_name = '/'.join([line.split()[0].split('/')[0], line.split()[0].split('/')[2]])
        #             self.img_path.append(os.path.join(root, img_name))
        #             self.labels.append(int(line.split()[1]))
        # else:
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return index, sample, label  # , index
