#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os,sys
path = os.path.dirname("D:\Pyproject\Federated-Learning-PyTorch-master\src_ly")
sys.path.append(path)
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from src_ly.utils.options import args_parser
from src_ly.utils.update import LocalUpdate, test_inference
from src_ly.utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from util import get_dataset, average_weights, exp_details,avgSV_weights
from src_ly.utils.plot import draw
from util import avgSV_weights
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

def imshow(img):

    #反归一化，将数据重新映射到0-1之间
    img = img / 2 + 0.5

    plt.imshow(np.transpose(img.numpy(), (1,2,0)))

    plt.show()


def solver():
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    # device = 'cuda' if args.gpu else 'cpu'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset and user groups
    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(args,False,True)
    data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for i, (images, _) in enumerate(data_loader):
        print(images.numpy().shape)
        imshow(torchvision.utils.make_grid(images))
        break

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer perceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()
    original_weights = copy.copy(global_weights)
    # Training
    train_loss, train_accuracy = [], []
    allAcc_list = []
    print_every = 2

    accuracy_list = []

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        # local_weights.append(copy.deepcopy(global_weights))
        weights = np.array([1/len(local_weights) for _ in range(len(local_weights))])
        global_weights = avgSV_weights(local_weights,weights,original_weights,True,idxs_users)

        # if epoch < 75:
        #     global_weights = average_weights(local_weights)
        # elif epoch < 100:
        #     shapley = np.ones(m)
        #     shapley = F.softmax(torch.tensor(shapley), dim=0)
        #     global_weights = SVAtt_weights(local_weights, shapley, original_weights, 0.1, epoch)

        # update global weights
        global_model.load_state_dict(global_weights)
        original_weights = copy.copy(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        allAcc_list.append(test_acc)
        print(" \nglobal accuracy:{:.2f}%".format(100 * test_acc))

    #draw(args.epochs, allAcc_list, "FedAvg 10 100")
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    accuracy_list.append(test_acc)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    return test_acc, train_accuracy[-1],accuracy_list

def show_avg(list):
    ans = []
    ans.append(np.mean(list[17:22]))
    ans.append(np.mean(list[37:42]))
    ans.append(np.mean(list[57:62]))
    ans.append(np.mean(list[77:82]))
    ans.append(np.mean(list[95:]))
    print(ans)


if __name__ == '__main__':
    test_acc, train_acc = 0, 0
    accuracy_list = []
    for _ in range(1):
        print("|---- 第「{}」次 ----|".format(_ + 1))
        test, train ,accuracy_list = solver()
        test_acc += test
        train_acc += train

        show_avg(accuracy_list)
