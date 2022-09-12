#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from src_ly.utils.options import args_parser
from src_ly.utils.scaffold_update import LocalUpdate, test_inference
from src_ly.utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from src_ly.util import get_dataset, average_weights, exp_details,SVAtt_weights
from src_ly.utils.plot import draw

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
    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(args)

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

    init_variate = copy.copy(global_weights)
    for key in init_variate.keys():
        init_variate[key] = torch.zeros_like(init_variate[key])

    clients_variate = []
    for i in range(args.num_users):
        clients_variate.append(copy.copy(init_variate))
    server_variate = copy.deepcopy(init_variate)

    # Training
    train_loss, train_accuracy = [], []
    allAcc_list = []
    print_every = 2

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses ,delta_variates = [], [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss, client_variate, delta_variate = local_model.update_weights(model=copy.deepcopy(global_model),
                                                 global_round=epoch, client_variate = clients_variate[idx], server_variate = server_variate)
            clients_variate[idx] = client_variate
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            delta_variates.append(copy.deepcopy(delta_variate))

        # update global weights
        # local_weights.append(copy.deepcopy(global_weights))
        global_weights = average_weights(local_weights)

        for key in server_variate.keys():
            for i in range(len(delta_variate)):
                server_variate[key] = server_variate[key] + 1/args.num_users*delta_variates[i][key]

        # update global weights
        global_model.load_state_dict(global_weights)

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

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    return test_acc, train_accuracy[-1]


if __name__ == '__main__':
    test_acc, train_acc = 0, 0
    for _ in range(1):
        print("|---- 第「{}」次 ----|".format(_ + 1))
        test, train = solver()
        test_acc += test
        train_acc += train
    print('|---------------------------------')
    print('|---------------------------------')
    print('|---------------------------------')
    print("|---- Train Accuracy: {:.2f}%".format(100 * (train_acc / 10)))
    print("|---- Test Accuracy: {:.2f}%".format(100 * (test_acc / 10)))
