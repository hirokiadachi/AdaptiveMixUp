import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from networks import wideresnet, wideresnet_avgpool, resnet, preactresnet

import os
import math
import numpy as np
import torch
from torchvision import datasets, transforms

def get_dataset(args):
    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    else:
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
        
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    if 'svhn' in args.dataset:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)])
        train_dataset = datasets.__dict__[args.dataset.upper()](args.dataroot+'/%s'%args.dataset, split='train', transform=train_transforms, download=True)
        test_dataset = datasets.__dict__[args.dataset.upper()](args.dataroot+'/%s'%args.dataset, split='test', transform=test_transforms, download=True)
    elif 'tiny' in args.dataset:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))])
        train_dataset = datasets.ImageFolder(os.path.join(args.tinypath, 'train'), transform=train_transforms)
        test_dataset = datasets.ImageFolder(os.path.join(args.tinypath, 'val'), transform=test_transforms)
    elif 'cifar' in args.dataset:
        train_dataset = datasets.__dict__[args.dataset.upper()](args.dataroot+'/%s'%args.dataset, train=True, transform=train_transforms, download=True)
        test_dataset = datasets.__dict__[args.dataset.upper()](args.dataroot+'/%s'%args.dataset, train=False, transform=test_transforms, download=True)
    elif 'mnist' in args.dataset:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor()])
        train_dataset = datasets.__dict__[args.dataset.upper()](args.dataroot+'/%s'%args.dataset, train=True, transform=train_transforms, download=True)
        test_dataset = datasets.__dict__[args.dataset.upper()](args.dataroot+'/%s'%args.dataset, train=False, transform=test_transforms, download=True)
    
    return train_dataset, test_dataset

def xent4mixup(xent, out, t):
    return torch.sum(-t * torch.log_softmax(out, dim=1), dim=1)
            

def scheduler(epoch, points, optimizer):
    lr = optimizer.param_groups[0]['lr']
    if epoch in points:
        lr = optimizer.param_groups[0]['lr'] * 0.1
    optimizer.param_groups[0]['lr'] = lr
        