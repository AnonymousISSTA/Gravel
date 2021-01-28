# the original training process
from .utils import *
import time
import sys


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, configs, logger, channel=3):
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(channel,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(channel, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.cuda(non_blocking=True)
        input = input - mean
        input.div_(std)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        output = model(input)
        loss = criterion(output, target)
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % configs.TRAIN.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, top1=top1, top5=top5,cls_loss=losses))
            sys.stdout.flush()

    return top1.avg


