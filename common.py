import os
import random
import numpy as np
import torch
import argparse
import time
from icecream import ic
from datetime import datetime
import time

def time_format():
    return f'{datetime.now():%Y-%m-%d %H:%M:%S}'

# ic.configureOutput(prefix=time_format, includeContext=True)
ic.configureOutput(includeContext=True)
ic.disable()


def loadArgments():
    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='dataset name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='~/dataset/cifar10', type=str, help='path to Cifar10 data', required=False)
    # parser.add_argument('--data_path', default='~/dataset/imagenet', type=str, help='path to Cifar10 data', required=False)

    # quantization parameters
    parser.add_argument('--n_bits_w', default=2, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', default=True, type=bool, help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', default=True, help='apply activation quantization', type=bool)
    parser.add_argument('--disable_8bit_head_stem', default=False, type=bool)
    parser.add_argument('--test_before_calibration', default=True, type=bool)

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', default=True, type=bool, help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')

    # activation calibration parameters
    parser.add_argument('--iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')
    
    #choigj
    parser.add_argument('--make_checkpoint', default=False, type=bool, help='generate checkpoint')
    # parser.add_argument('--make_checkpoint', default=True, type=bool, help='generate checkpoint')
    parser.add_argument('--skip_test', default=False, type=bool, help='skip default test')
    parser.add_argument('--run_device', default='cuda:0', type=str, help='gpu usage')
    parser.add_argument('--msg_bot_enable', default=True, type=bool, help='use messaging bot for monitoring')
    parser.add_argument('--make_init_data', default=False, type=bool, help='Make Initiallize weight data')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
    # parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
    parser.add_argument('--bypassChannelShift', default=False, type=bool, help='do not run channel shift function')
    
    #Shift Quantization
    parser.add_argument('--mse_level',          default=1,   type=int, help='1, 2, 4, ...')
    parser.add_argument('--mse_threshold',      default=1.0, type=float, help='round 범위 얼마나 넓게 할지')
    parser.add_argument('--shift_quant_mode',   default='max', type=str, help='mse or max')
    parser.add_argument('--w_scale_method', default='mse', type=str, help='mse or max')
    parser.add_argument('--a_scale_method', default='mse', type=str, help='mse or max')
    
    parser.add_argument('--test', default=False, type=bool, help='test')
    
    return parser.parse_args()
    
def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]

@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100, print_result=False, simple=False, bit=-1):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    correct = 0
    total = 0
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    
    
    output_list = []
    
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        if bit != -1:
            output_list.append(output)
        
        #Predicted
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # measure accuracy and record loss
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and print_result:
            progress.display(i)
            
        if simple and i > 5:
            break
    
    
    if bit != -1:
        # Stack the output tensors
        stacked_output = torch.cat(output_list, dim=0)
        # print('shape : ', stacked_output.shape)
        # Save the stacked output to a file
        ref_out = torch.load(f'./output_loss/result_{bit}bit.pt')
        mse = torch.mean(torch.square(stacked_output - ref_out))
        # print(f'MSE[{bit}] : {mse:.2e}')
        # torch.save(stacked_output, f'./output_loss/result_{bit}bit.pt')
    
    acc = 100 * correct / total

    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    if print_result:
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


@torch.no_grad()
def validate_with_loss(val_loader, model, device=None, print_freq=100, print_result=False, simple=False, bit=-1):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    correct = 0
    total = 0
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    
    
    output_list = []
    
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        if bit != -1:
            output_list.append(output)
        
        #Predicted
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # measure accuracy and record loss
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and print_result:
            progress.display(i)
            
        if simple and i > 5:
            break
    
    mse = 0
    if bit != -1:
        # Stack the output tensors
        stacked_output = torch.cat(output_list, dim=0)
        print('shape : ', stacked_output.shape)
        # Save the stacked output to a file
        ref_out = torch.load(f'./output_loss/result_{bit}bit.pt')
        mse = torch.mean(torch.square(stacked_output - ref_out))
        print(f'MSE[{bit}] : {mse:.2e}')
        # torch.save(stacked_output, f'./output_loss/result_{bit}bit.pt')
    
    acc = 100 * correct / total

    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    if print_result:
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, mse

def print_model_hierarchy(model, depth=0):
    for name, child in model.named_children():
        print("--"*depth, name)
        print_model_hierarchy(child, depth + 1)
        