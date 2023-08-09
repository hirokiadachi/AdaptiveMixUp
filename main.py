import os
import time
import shutil
import random
import argparse
import numpy as np
import wandb
wandb.login(key="type your api key")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS

from utils import *
from configures import configures
from distributed import init_distributed_device, world_info_from_env

def random_seed(seed=1729, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

if __name__ == '__main__':
    args = configures()
    
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    random_seed(args.seed)

    if args.rank == 0:
        savepoint = os.path.join(args.checkpoint, args.arch, args.dataset, 'trial-%d'%args.num_trials)
        os.makedirs(savepoint, exist_ok=True)
    
        wandb.init(project="AdaptiveMixUp", 
                   name='{}_{}_{}-trials'.format(
                       args.arch,
                       args.dataset,
                       args.num_trials
                   ),
                   config=vars(args))
    
    model = get_network(args).cuda()
    ddp_model = DDP(model, device_ids=[device_id])
    optimizer = optim.SGD(
        ddp_model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay, 
        nesterov=False                   
    )
    
    trainset, testset = get_dataset(args)
    trainloader = DataLoader(
        trainset, 
        batch_size=args.train_batch_size*3, 
        shuffle=False, 
        drop_last=False, 
        num_workers=os.cpu_count(), 
        pin_memory=True,
        sampler=DS(
            trainset,
            num_replicas=args.world_size, 
            rank=args.rank, 
            shuffle=True
        )
    )
    testloader = DataLoader(
        testset, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=os.cpu_count(), 
        pin_memory=True,
        sampler=DS(
            testset, 
            num_replicas=args.world_size, 
            rank=args.rank, 
            shuffle=False
        )
    )

    best_acc = 0
    criterion = xent4mixup
    
    for epoch in range(1, args.epoch+1):
        avg_loss, avg_cls_loss, avg_dis_loss, val_loss, val_top1_acc, val_top5_acc = train_eval_roop(
            epoch, model, trainloader, testloader, optimizer, criterion, args
        )

        if args.rank == 0:
            wandb.log({
                'classification loss': avg_cls_loss,
                'intrusion loss': avg_cls_loss,
                'total loss': avg_loss
            })
    
            wandb.log({
                'test accuracy': val_accuracy,
                'test loss': val_loss,
            })
        
            print('Accuracy @ %d epochs: %.4f' % (epoch, val_accuracy))
        
            is_best = val_accuracy > best_acc
            states = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'accuracy': val_accuracy}
            torch.save(states, os.path.join(savepoint, 'lastcheckpoint.pth.tar'))
            if epoch > (args.epoch-10):
                torch.save(states, os.path.join(savepoint, 'checkpoint-%depochs.pth.tar' % epoch))
            if is_best:
                best_acc = val_accuracy
                torch.save(states, os.path.join(savepoint, 'bestcheckpoint.pth.tar'))
    
        scheduler(epoch, args.lr_decay, optimizer)