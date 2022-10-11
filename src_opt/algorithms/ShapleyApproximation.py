#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys,os
path = os.path.dirname("D:\Pyproject\Federated-Learning-PyTorch-master\src_opt")
sys.path.append(path)

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from src_opt.utils.options import args_parser
from src_opt.utils.update import LocalUpdate, test_inference
from src_opt.utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from src_opt.utils.tools import get_dataset, average_weights, exp_details
from src_opt.utils.plot import draw
from src_opt.utils.Shapley import Shapley


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
    original_weights = copy.copy(global_weights)
    # Training
    train_loss, train_accuracy = [], []
    allAcc_list = []
    print_every = 2
    init_acc = 0
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

        Fed_sv = Shapley(local_weights, args, global_model, valid_dataset, init_acc)
        #shapley_benchmark = Fed_sv.eval_mcshap(100)
        #shapley_benchmark_cc = Fed_sv.eval_ccshap(50)
        #shapley_mc = Fed_sv.eval_mcshap(5)
        #shapley_cc = Fed_sv.eval_ccshap(5)
        shapley_benchmark =Fed_sv.eval_ccshap_stratified(20)
        shapley_benchmark1 =Fed_sv.eval_ccshap_stratified(20)
        shapley_cc_str =Fed_sv.eval_ccshap_stratified(5)
        shapley_cc_opt = Fed_sv.ccshap_optimal_sampling(5,2,0.2)
        

        # print(shapley_benchmark)
        # print(shapley_cc)
        # print(shapley_cc_opt)

        # err_bench = 0
        # for i in range(len(shapley_benchmark)):
        #     err_bench += np.abs(shapley_benchmark[i]-shapley_benchmark_cc[i])

        # err_mc = 0
        # for i in range(len(shapley_benchmark)):
        #     err_mc += np.abs(shapley_benchmark[i]-shapley_mc[i])

        # err_cc = 0
        # for i in range(len(shapley_benchmark)):
        #     err_cc += np.abs(shapley_benchmark[i]-shapley_cc[i])

        err_benchmark = 0
        for i in range(len(shapley_benchmark)):
            err_benchmark += np.abs(shapley_benchmark[i]-shapley_benchmark1[i])

        err_cc_str = 0
        for i in range(len(shapley_benchmark)):
            err_cc_str += np.abs(shapley_benchmark[i]-shapley_cc_str[i])

        err_cc_opt = 0
        for i in range(len(shapley_benchmark)):
            err_cc_opt += np.abs(shapley_benchmark[i]-shapley_cc_opt[i])
        
        # print("bench err: ", err_bench)
        # print("mc err: ", err_mc)
        # print("cc err: ", err_cc)
        print("bench err: ", err_benchmark)
        print("cc_str err: ", err_cc_str)
        print("cc_opt err: ", err_cc_opt)
        
        global_weights = average_weights(local_weights)

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
        init_acc = test_acc

    #draw(args.epochs, allAcc_list, "FedAvg 10 100")
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    accuracy_list.append(test_acc)

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
    return test_acc, train_accuracy[-1],accuracy_list


if __name__ == '__main__':
    test_acc, train_acc = 0, 0
    accuracy_list = []
    for _ in range(1):
        print("|---- 第「{}」次 ----|".format(_ + 1))
        test, train ,accuracy_list = solver()
        test_acc += test
        train_acc += train

    print('|---------------------------------')
    print('|---------------------------------')
    print('|---------------------------------')
    print("|---- Train Accuracy: {:.2f}%".format(100 * (train_acc / 10)))
    print("|---- Test Accuracy: {:.2f}%".format(100 * (test_acc / 10)))
