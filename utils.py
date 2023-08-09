import os
import time
import math
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import datasets, transforms
from networks import *

def train_eval_roop(
    epoch,
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    args,
):
    ### training mode
    total_loss = AverageMeter()
    total_cls_loss = AverageMeter()
    total_dis_loss = AverageMeter()
    model.train()
    for batch_index, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        inputs, targets = inputs.cuda(), targets.cuda()
        batch = inputs.size(0)
        eye = torch.eye(args.num_classes).cuda()
        beye = torch.eye(2).cuda()
        lam, cls_out, targets_ys, dis_out, binary_ys = model(inputs, eye[targets], is_train=True)
        loss_cls = criterion(cls_out.softmax(dim=1), targets_ys).mean()
        loss_dis = criterion(dis_out.softmax(dim=1), beye[binary_ys.long()]).mean()
        loss = (loss_cls + loss_dis)/2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        elapsed_time = end - start
        lr = optimizer.param_groups[0]['lr']

        reduced_loss = reduce_tensor(loss.data, args.world_size)
        reduced_loss_cls = reduce_tensor(loss_cls.data, args.world_size)
        reduced_loss_dis = reduce_tensor(loss_dis.data, args.world_size)
        
        total_loss.update(to_python_float(reduced_loss), batch)
        total_cls_loss.update(to_python_float(reduced_loss_cls), batch)
        total_dis_loss.update(to_python_float(reduced_loss_dis), batch)

        #wandb.log({
        #        'train_total_loss': to_python_float(reduced_loss),
        #        'train_cls_loss': to_python_float(reduced_loss_cls),
        #        'train_dis_loss': to_python_float(reduced_loss_dis),
        #        'lambda': reduce_tensor(lam.mean().data, args.world_size).item()
        #    })
        
        if args.rank == 0 and batch_index % 100 == 0:
            print('%d epochs [%d/%d]\tloss: %.4f (avg: %.4f)\tLR: %.5f\telapsed_time: %.4f' % (
                epoch, (batch_index + 1), len(train_loader), loss.item(), total_loss/(batch_index+1), lr, elapsed_time))
    train_cls_loss_avg = total_cls_loss / len(train_loader)
    train_dis_loss_avg = total_dis_loss / len(train_loader)
    train_total_loss_avg = total_loss / len(train_loader)
    
    ### evaluation mode
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            eye = torch.eye(args.num_classes).cuda()
            logits = model(inputs, is_train=False)
            loss = criterion(logits.softmax(dim=1), eye[targets]).mean().item()

            prec1, prec5 = accuracy(logits.data, targets.data, topk=(1,5))
            
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            prec1 = reduce_tensor(prec1, args.world_size)
            prec5 = reduce_tensor(prec5, args.world_size)

            losses.update(to_python_float(reduced_loss), batch)
            top1.update(to_python_float(prec1), batch)
            top5.update(to_python_float(prec5), batch)

    return total_cls_loss.avg, train_dis_loss.avg, total_loss.avg, losses.avg, top1.avg, top5.avg




def get_dataset(args):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    if 'svhn' in args.dataset:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()])
        train_dataset = datasets.__dict__[args.dataset.upper()](args.dataroot,
                                                                split='train',
                                                                transform=train_transforms, 
                                                                download=True)
        test_dataset = datasets.__dict__[args.dataset.upper()](args.dataroot,
                                                               split='test', 
                                                               transform=transforms.ToTensor(), 
                                                               download=True)
    elif args.dataset == 'imagenet':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        train_dataset = datasets.ImageFolder(
            os.path.join(args.dataroot, 'train'), 
            transform=train_transforms
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(args.dataroot, 'val'), 
            transform=test_transforms
        )
    elif 'tiny' in args.dataset:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))])
        train_dataset = datasets.ImageFolder(
            os.path.join(args.dataroot, 'train'), 
            transform=train_transforms
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(args.dataroot, 'val'), 
            transform=test_transforms
        )
    elif 'cifar' in args.dataset:
        train_dataset = datasets.__dict__[args.dataset.upper()](args.dataroot, 
                                                                train=True, 
                                                                transform=train_transforms, 
                                                                download=True
                                                               )
        test_dataset = datasets.__dict__[args.dataset.upper()](args.dataroot, 
                                                               train=False, 
                                                               transform=test_transforms, 
                                                               download=True
                                                              )
    elif 'mnist' in args.dataset:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor()])
        train_dataset = datasets.__dict__[args.dataset.upper()](args.dataroot, 
                                                                train=True, 
                                                                transform=train_transforms, 
                                                                download=True
                                                               )
        test_dataset = datasets.__dict__[args.dataset.upper()](args.dataroot, 
                                                               train=False, 
                                                               transform=transforms.ToTensor(), 
                                                               download=True
                                                              )
    elif 'stl10' in args.dataset:
        train_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))])
        
        test_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))])
        train_dataset = datasets.__dict__[args.dataset.upper()](
            os.path.join(args.dataroot, args.dataset), 
            split='train', 
            transform=train_transforms, 
            download=True)
        test_dataset = datasets.__dict__[args.dataset.upper()](
            os.path.join(args.dataroot, args.dataset), 
            split='test', 
            transform=test_transforms, 
            download=True)
    
    return train_dataset, test_dataset

def get_network(args):
        
    if 'preactres18' == args.arch:
        model = PreActResNet18(num_classes=args.num_classes, stride=args.stride)
    elif 'preactres34' == args.arch:
        model = PreActResNet34(num_classes=args.num_classes, stride=args.stride)
    elif 'preactres50' == args.arch:
        model = PreActResNet50(num_classes=args.num_classes, stride=args.stride)
    elif 'preactres101' == args.arch:
        model = PreActResNet101(num_classes=args.num_classes, stride=args.stride)
    elif 'preactres152' == args.arch:
        model = PreActResNet152(num_classes=args.num_classes, stride=args.stride)
    elif 'lenet' == args.arch:
        model = Lenet5(num_classes=args.num_classes)
    elif 'wrn34-10' == args.arch:
        model = wrn34_10(num_classes=args.num_classes, stride=args.stride)
    elif 'wrn28-10' == args.arch:
        model = wrn28_10(num_classes=args.num_classes, stride=args.stride)
    elif 'resnet18' == args.arch:
        model = ResNet18(num_classes=args.num_classes, stride=args.stride)
    elif 'resnet34' == args.arch:
        model = ResNet34(num_classes=args.num_classes, stride=args.stride)
    elif 'resnet50' == args.arch:
        model = ResNet50(num_classes=args.num_classes, stride=args.stride)
    elif 'resnet101' == args.arch:
        model = ResNet101(num_classes=args.num_classes, stride=args.stride)
    elif 'resnet152' == args.arch:
        model = ResNet152(num_classes=args.num_classes, stride=args.stride)
        
    return model

##############################
def scheduler(epoch, points, optimizer):
    lr = optimizer.param_groups[0]['lr']
    if epoch in points:
        lr = optimizer.param_groups[0]['lr'] * 0.1
    optimizer.param_groups[0]['lr'] = lr

def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
    
def xent4mixup(out, t):
    return torch.sum(-t * torch.log(out), dim=1, keepdims=True)

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def to_python_float(t: torch.Tensor):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def reduce_tensor(tensor, world_size):
    """Reduce tensor across all nodes."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt