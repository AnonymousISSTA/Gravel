"""
The original version of free adv training.
"""
from lib import *
import torch.nn as nn
from collections import OrderedDict
import numpy as np


def generate_grad(grad, adv_grad, layer_mask=None, lr=0.01):
    ret_grad = OrderedDict()
    for k,v in grad.items():
        g1 = grad[k]
        g2 = adv_grad[k]
        if k in layer_mask:
            # randomly perturb the gradients.
            delta = lr * torch.randn(g2.size())
            ret_grad[k] = g2 
    
        else:
            ret_grad[k] = g2

    return ret_grad

     
        
def get_grad(model, y, y_true, optimizer, criterion):
    loss = criterion(y, y_true)
    optimizer.zero_grad()
    loss.backward()
    grad = OrderedDict()
    for t1,t2 in zip(model.parameters(), model.state_dict().items()):
        grad[t2[0]] = t1.grad.data.clone()
    return grad


def set_grad(model, grad):
    for t1,t2 in zip(model.parameters(), model.state_dict().items()):
        if 'weight' in t2[0] and t2[0] in grad:
            t1.grad = grad[t2[0]]
        else:
            pass
            # print(t2[0])
            # print('something wrong.')

def get_grad_diff_layer_mask(grad, adv_grad, ratio=0.1):
    layer_mask = OrderedDict()
    avg_list = []

    def cal_mean_diff(g1, g2):
        diff = g1 - g2
        normalized_diff = (diff - torch.min(diff)) / (torch.max(diff) - torch.min(diff))
        return torch.mean(normalized_diff)

    for k,v in grad.items():
        if 'weight' not in k:
            continue
        layer_mask[k] = 0
        g1 = grad[k]
        g2 = adv_grad[k]
        avg_g = cal_mean_diff(g1, g2)
        avg_list.append(avg_g)
    
    # torch.kthvalue from smallest to largest.
    avg_list = torch.tensor(avg_list)
    threshold = torch.kthvalue(avg_list, int(avg_list.size(0) * (1 - ratio))).values
    for k,v in layer_mask.items():
        if v >= threshold:
            layer_mask[k] = 1

    return layer_mask


def gravel_random_adv_train_one_epoch_v1(model, train_loader, criterion, optimizer, epoch, configs, global_noise_data, logger, channel=3, ratio=0.1):  # TODO modify the code
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(channel,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(channel, configs.DATA.crop_size, configs.DATA.crop_size).cuda()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()

    model.train()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        in_origin = input.clone() - mean
        in_origin.div_(std)
        origin_output = model(input)
        grad = get_grad(model, origin_output, target, optimizer, criterion)


        for j in range(configs.ADV.n_repeats):
            # Ascend on the global noise
            noise_batch = global_noise_data[0:input.size(0)]
            noise_batch = Variable(noise_batch, requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)

            adv_output = model(in1)
            adv_grad = get_grad(model, adv_output, target, optimizer, criterion)
            
            layer_mask = get_grad_diff_layer_mask(grad, adv_grad, ratio=ratio)

            ret_grad = generate_grad(grad, adv_grad, layer_mask=layer_mask)
            set_grad(model, ret_grad)
            # set_grad(model, adv_grad)

            adv_prec1, adv_prec5 = accuracy(adv_output, target, topk=(1, 5))
            # losses.update(loss.item(), input.size(0))
            
            adv_top1.update(adv_prec1[0], input.size(0))
            adv_top5.update(adv_prec5[0], input.size(0))

            pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-configs.ADV.clip_eps, configs.ADV.clip_eps)


            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs.TRAIN.print_freq == 0:
                print('Train Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    #   'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, top1=adv_top1, top5=adv_top5))  # ,cls_loss=losses
                sys.stdout.flush()
    return global_noise_data, top1.avg
    