#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from src_ly.utils.sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, cifar_iid, cifar_noniid, \
    FashionMnist_noniid
from src_ly.utils.options import args_parser
import ssl
import random
import numpy as np

def clip(data):
    clip_data = torch.tensor(data.shape)


def get_dataset(args,data_noise=False,gradient_noise=False):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset_all = datasets.CIFAR10(data_dir, train=False, download=True,
                                            transform=apply_transform)
        test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset_all, [2000, 8000])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist':
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset_all = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

        test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset_all, [2000, 8000])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'fmnist':
        data_dir = '../data/fmnist'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)

        test_dataset_all = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

        test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset_all, [2000, 8000])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = FashionMnist_noniid(train_dataset, args.num_users)

    #train_dataset.train_labels[train_dataset.train_labels == 0] = 1
    if data_noise == False and gradient_noise == False:
        train_dataset.targets = torch.tensor(train_dataset.targets)
        for i in range(0,9,2):
            train_dataset.targets[train_dataset.targets == i] = i+1
    elif data_noise:
        new_train_dataset = []
        for i in range(len(train_dataset)):
            feature,label = train_dataset[i]
            if int(i/200)%2 == 0:
                noise = torch.tensor(np.random.normal(0,0.1,feature.shape))
                noise = noise.to(torch.float32)
                new_data = feature+noise
                clip_data = torch.clamp(new_data,-1,1)
                new_train_dataset.append((clip_data,label))
            else:
                new_train_dataset.append((feature,label))
        train_dataset = new_train_dataset
    
    if gradient_noise == False:
        indices = []
        for i in range(len(valid_dataset)):
            data, label = valid_dataset[i]
            if label%2 != 0:
                indices.append(i)
        new_valid_dataset = torch.utils.data.Subset(valid_dataset, indices)
        valid_dataset = new_valid_dataset

        indices = []
        for i in range(len(test_dataset)):
            data, label = test_dataset[i]
            if label%2 != 0:
                indices.append(i)
        new_test_dataset = torch.utils.data.Subset(test_dataset, indices)
        test_dataset = new_test_dataset
    

    return train_dataset, valid_dataset, test_dataset, user_groups

    #return train_dataset, valid_dataset, test_dataset, user_groups
    # return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    最正常的平均
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def avgSV_weights(w, shapley, ori, Flag = False,idxs=None):
    """
        Shapley权值平均
        Returns the average of the weights.
    """
    w_avg = copy.deepcopy(ori)
    for key in w_avg.keys():
        for i in range(0, len(w)):
            if Flag == False:
                w_avg[key] += (w[i][key]-ori[key]) * shapley[i]
            else:
                noise = torch.tensor(np.random.normal(0,0.01,w_avg[key].shape))
                noise = noise.to(torch.float32)
                w_avg[key] += (w[i][key]-ori[key]+noise) * shapley[i]
    return w_avg


"""
    p: The probabilities of each arm been picked in the first round
    C: The number of arms that been picked in each round
"""
def arms_selection(p,C):
    selected = []
    tuples = []
    for i in range(len(p)):
        tuples.append((i,p[i]))
    remain = 1
    for _ in range(C):
        rand = random.random()
        pre = 0
        for i in range(len(tuples)):
            if tuples[i][0] not in selected:
                if rand >= pre and rand < pre+tuples[i][1]/remain:
                    selected.append(i)
                    remain -= tuples[i][1]
                    break
                else:
                    pre += tuples[i][1]/remain
    return selected

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset   : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
