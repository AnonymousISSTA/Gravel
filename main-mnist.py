import argparse
import os
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import numpy as np
from lib import *
from lib.train import train_one_epoch
from free.train import free_adv_train_one_epoch
from gravel import gravel_adv_train_one_epoch
from gravel_random_1 import gravel_random_adv_train_one_epoch_v1
from gravel_random_2 import gravel_random_adv_tsrain_one_epoch_v2 
# from validation import validate, validate_pgd
import torchvision.models as models


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    # parser.add_argument('data', metavar='DIR', default='',
    #                 help='path to dataset')
    parser.add_argument('--output_prefix', default='free_adv', type=str,
                    help='prefix used to define output path')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                    help='path to the config file (default: configs.yml)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    parser.add_argument('--dataset', '-d', type=str, default='cifar10')
    parser.add_argument('--mode', type=int, default=1)  # 1: normal train  2: free train  3: Gravel train
    parser.add_argument('--version', '-v', type=int, default=99)
    parser.add_argument('--ratio', '-r', type=float, default=0.1)
    return parser.parse_args()



def process():
    args = parse_args()
    print(args)
    configs = parse_config_file(args)
    x = configs.TRAIN.mean

    best_prec1 = 0
    if configs.mode != 1:
        configs.TRAIN.epochs = int(math.ceil(configs.TRAIN.epochs / configs.ADV.n_repeats))
    else:
        configs.ADV.n_repeats = 1
    configs.ADV.fgsm_step /= configs.DATA.max_color_value
    configs.ADV.clip_eps /= configs.DATA.max_color_value

    # if not os.path.isdir(os.path.join('trained_models', configs.output_name)):
    #     os.makedirs(os.path.join('trained_models', configs.output_name))

    # load the model.
    model = get_model_by_name(configs.TRAIN.arch, args.dataset)
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    model_weights_dir = os.path.join(weights_dir, configs.TRAIN.arch)
    model_weights_dir = os.path.join(model_weights_dir, args.dataset)
    model_weights_dir = os.path.join(model_weights_dir, str(args.version))

    eval_logger = initiate_logger(model_weights_dir)
    logger = Logger(model_weights_dir, 
                        filename='log_{}_repeats{}.txt'.format(args.mode, configs.ADV.n_repeats),
                        record_filename='record_{}_repeats{}.npy'.format(args.mode, configs.ADV.n_repeats)
                    )

    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)

    model_trained_path = os.path.join(model_weights_dir, 'trained_{}_repeats{}_eps{}.pth'.format(args.mode, configs.ADV.n_repeats, configs.ADV.fgsm_step))
    # model_trained_path = os.path.join(model_weights_dir, 'trained_{}_repeats{}.pth'.format(args.mode, configs.ADV.n_repeats))

    train_acc_path = os.path.join(model_weights_dir, 'mode_{}_repeats{}_eps{}_train.npy'.format(args.mode, configs.ADV.n_repeats, configs.ADV.fgsm_step))
    test_acc_path = os.path.join(model_weights_dir, 'mode_{}_repeats{}_eps{}_test.npy'.format(args.mode, configs.ADV.n_repeats, configs.ADV.fgsm_step))

    print('learning rate: {}'.format(configs.TRAIN.lr))
    optimizer = torch.optim.SGD(model.parameters(), configs.TRAIN.lr,
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), configs.TRAIN.lr)

    if configs.resume:
        if os.path.isfile(configs.resume):
            print("=> loading checkpoint '{}'".format(configs.resume))
            checkpoint = torch.load(configs.resume)
            configs.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(configs.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

    train_loader, test_loader = get_dataloader(configs.dataset)

    # If in evaluate mode: perform validation on PGD attacks as well as clean samples
    if configs.evaluate:
        model.load_state_dict(torch.load(model_trained_path))
        eval_logger.info(pad_str(' Performing PGD Attacks '))
        for pgd_param in configs.ADV.pgd_attack:
            validate_pgd(test_loader, model, criterion, pgd_param[0], pgd_param[1], configs, eval_logger, channel=1)
        validate(train_loader, model, criterion, configs, eval_logger, channel=1)
        validate(test_loader, model, criterion, configs, eval_logger, channel=1)
        return

    global_noise_data = torch.zeros([configs.DATA.batch_size, 1, configs.DATA.crop_size, configs.DATA.crop_size]).cuda()
    current_lr = None

    train_acc_list = []
    test_acc_list = []
    
    print(configs.TRAIN.start_epoch, configs.TRAIN.epochs)
    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        adjust_learning_rate(configs.TRAIN.lr, optimizer, epoch, configs.ADV.n_repeats, step=10.0)
        
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        print('current learning rate: {}'.format(current_lr))

        if configs.mode == 1:  # normal train
            train_acc = train_one_epoch(model, train_loader=train_loader, 
                            criterion=criterion, 
                            optimizer=optimizer, 
                            epoch=epoch, 
                            configs=configs,
                            logger=logger,
                            channel=1)
            train_acc = validate(train_loader, model, criterion, configs, eval_logger, channel=1)
            test_acc = validate(test_loader, model, criterion, configs, eval_logger, channel=1)

        elif configs.mode == 2:  # free train
            global_noise_data, train_acc = free_adv_train_one_epoch(model, 
                            train_loader=train_loader, 
                            criterion=criterion, 
                            optimizer=optimizer, 
                            epoch=epoch, 
                            configs=configs, 
                            global_noise_data=global_noise_data,
                            logger=logger,
                            channel=1)
            train_acc = validate(train_loader, model, criterion, configs, eval_logger, channel=1)
            test_acc = validate(test_loader, model, criterion, configs, eval_logger, channel=1)

        elif configs.mode == 3:  # gravel train
            global_noise_data, train_acc = gravel_adv_train_one_epoch(model, 
                            train_loader=train_loader, 
                            criterion=criterion, 
                            optimizer=optimizer, 
                            epoch=epoch, 
                            configs=configs, 
                            global_noise_data=global_noise_data,
                            logger=logger,
                            channel=1)
            train_acc = validate(train_loader, model, criterion, configs, eval_logger, channel=1)
            test_acc = validate(test_loader, model, criterion, configs, eval_logger, channel=1)

        elif configs.mode == 4:  # Random-1
            global_noise_data, train_acc = gravel_random_adv_train_one_epoch_v1(model, 
                            train_loader=train_loader, 
                            criterion=criterion, 
                            optimizer=optimizer, 
                            epoch=epoch, 
                            configs=configs, 
                            global_noise_data=global_noise_data,
                            logger=logger,
                            channel=1)
            train_acc = validate(train_loader, model, criterion, configs, eval_logger, channel=1)
            test_acc = validate(test_loader, model, criterion, configs, eval_logger, channel=1)

        elif configs.mode == 5:  # Random-2
            global_noise_data, train_acc = gravel_random_adv_tsrain_one_epoch_v2(model, 
                            train_loader=train_loader, 
                            criterion=criterion, 
                            optimizer=optimizer, 
                            epoch=epoch, 
                            configs=configs, 
                            global_noise_data=global_noise_data,
                            logger=logger,
                            channel=1)
            train_acc = validate(train_loader, model, criterion, configs, eval_logger, channel=1)
            test_acc = validate(test_loader, model, criterion, configs, eval_logger, channel=1)

        if isinstance(train_acc, torch.Tensor):
            train_acc = float(train_acc)
        if isinstance(test_acc, torch.Tensor):
            test_acc = float(test_acc)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        torch.save(model.state_dict(), model_trained_path)
        # logger.save_npy(np.array(train_acc_list))
        np.save(train_acc_path, np.array(train_acc_list))
        np.save(test_acc_path, np.array(test_acc_list))


if __name__ == '__main__':
    print(validate, validate_pgd)
    process()