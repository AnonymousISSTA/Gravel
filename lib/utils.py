import logging
import os
import datetime
import torchvision.models as models
import math
import torch
import yaml
from easydict import EasyDict
import shutil
from models import *
from datetime import datetime
import numpy as np


class Logger():
    def __init__(self, log_dir, filename='log.txt', record_filename='train.npy'):
        self.log_path = os.path.join(log_dir, filename)
        self.record_path = os.path.join(log_dir, record_filename)

    def write_info(self, msg):
        print(msg)
        now = datetime.now()
        str_now = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        f = open(self.log_path, 'a+')
        w_msg = "[{}] {}\n".format(str_now, msg)
        f.write(w_msg)
        f.close()
    
    def save_npy(self, npy_list):
        np.save(self.record_path, npy_list)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats, step=60.0):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // int(math.ceil(step/n_repeats))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def initiate_logger(output_path):
    if not os.path.isdir(os.path.join('output', output_path)):
        os.makedirs(os.path.join('output', output_path))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join('output', output_path, 'log.txt'),'w'))
    logger.info(pad_str(' LOGISTICS '))
    logger.info('Experiment Date: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M')))
    logger.info('Output Name: {}'.format(output_path))
    logger.info('User: {}'.format(os.getenv('USER')))
    return logger

def get_model_names():
	return sorted(name for name in models.__dict__
    		if name.islower() and not name.startswith("__")
    		and callable(models.__dict__[name]))

def pad_str(msg, total_len=70):
    rem_len = total_len - len(msg)
    return '*'*int(rem_len/2) + msg + '*'*int(rem_len/2)\

def parse_config_file(args):
    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.Loader))
        
    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v
        
    # Add the output path
    config.output_name = '{:s}_step{:d}_eps{:d}_repeat{:d}'.format(args.output_prefix,
                         int(config.ADV.fgsm_step), int(config.ADV.clip_eps), 
                         config.ADV.n_repeats)
    return config


def save_checkpoint(state, is_best, filepath):
    filename = os.path.join(filepath, 'checkpoint.pth.tar')
    # Save model
    torch.save(state, filename)
    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))


def get_model_by_name(model_name, dataset_name):
    if dataset_name in ['mnist', 'cifar10', 'svhn']:
        num_classes = 10
    elif dataset_name in ['cifar100']:
        num_classes = 100
    
    if model_name == 'lenet5':
        model = get_lenet5(num_classes=num_classes)
    elif model_name == 'cnn':
        model = get_cnn(num_classes=num_classes)
    elif model_name == 'mobilenetv2':
        model = mobilenet_v2(num_classes=num_classes)
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes)
    # elif model_name == 'shufflenetv2':
    #     model = shufflenet_v2(num_classes=num_classes)

    return model