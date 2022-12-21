import os
import time
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from networks import preactresnet, wideresnet, lenet, resnet
from utils import get_dataset, scheduler

def training(epoch, model, dataloader, optimizer, xent, tb, args):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        start = time.time()
        inputs, targets = inputs.cuda(), targets.cuda()
        x1, x2, x3 = torch.chunk(inputs, 3, dim=0)
        y1, y2, y3 = torch.chunk(targets, 3, dim=0)
        
        m1 = torch.randn(x1.size()).cuda()
        m2 = torch.randn(x2.size()).cuda()
        x = (m1*x1+m2*x2) / 2
        alpha1, delta, alpha2 = model(x, mode='policy')
        eps = torch.FloatTensor(alpha1.size()).uniform_(0,1).cuda()
        weight = alpha1 + eps*delta
        
        mixed_x = x1 * weight.view(-1,1,1,1) + x2 * (1 - weight.view(-1,1,1,1))
        inputs_cls = torch.cat([mixed_x, x1], dim=0)
        
        pos = torch.ones(x3.size(0)*2).cuda()
        neg = torch.zeros(mixed_x.size(0)).cuda()
        binary_label = torch.cat([pos, neg], dim=0)
        inputs_dis = torch.cat([x3, x1, mixed_x], dim=0)
        random_index = torch.randperm(inputs_dis.size(0)).cuda()
        binary_label = binary_label[random_index]
        inputs_dis = inputs_dis[random_index]
        
        logit1 = model(inputs_cls, mode='classifier')
        logit2 = model(inputs_dis, mode='intrusion')
        
        onehot = torch.eye(logit1.size(1)).cuda()
        binary = torch.eye(logit2.size(1)).cuda()
        mixed_logit, clean_logit = torch.chunk(logit1, 2, dim=0)
        loss_clean = -torch.sum(onehot[y1]*torch.log_softmax(clean_logit, dim=1)) / clean_logit.size(0)
        mixed_label =  onehot[y1]*weight.view(y1.size(0), -1) + onehot[y2]*(1 - weight.view(y1.size(0), -1))
        loss_mix = - torch.sum(mixed_label*torch.log_softmax(mixed_logit, dim=1)) / mixed_logit.size(0)
        loss_classification = (loss_mix + loss_clean)
        loss_intrusion = -torch.sum(binary[binary_label.long()]*torch.log_softmax(logit2, dim=1)) / logit2.size(0)
        loss = loss_classification + loss_intrusion
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        end = time.time()
        elapsed_time = end - start
        lr = optimizer.param_groups[0]['lr']
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print('%d epochs [%d/%d]\tloss: %.4f (avg: %.4f)\tLR: %.5f\telapsed_time: %.4f' % (
                epoch, (batch_idx + 1), len(dataloader), loss.item(), total_loss/(batch_idx+1), lr, elapsed_time))
    return total_loss/len(dataloader)

def evaluation(model, dataloader):
    total_accuracies = 0
    total_losses = 0
    model.eval()
    xent = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs, mode='classifier')
            total_losses += xent(outputs, targets).item()
            total_accuracies += outputs.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    return total_losses/len(dataloader), (100 * (total_accuracies/len(dataloader.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100,
                        help='the number of epochs')
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU ids')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint',
                        help='directry name to save checkpoint')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='the number of batchs')
    parser.add_argument('--arch', type=str, default='preactres18')
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--loss', type=str, default='bce')
    
    ## Optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='amount of the momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--pretrained', type=str, default='',
                        help='pretrained model path for evalation.')
    
    ## mixup
    parser.add_argument('--lr_decay', nargs="*", type=int, default=[400, 800])
    parser.add_argument('--seed', type=int, default=np.random.randint(4294967295),
                        help='random seed')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    savepoint = os.path.join(args.checkpoint, args.arch, args.dataset, 'trial-%d'%args.num_trials)
    os.makedirs(savepoint, exist_ok=True)
    
    if args.train:
        ftb = os.path.join(savepoint, 'runs')
        if os.path.exists(ftb):
            shutil.rmtree(ftb)
        tb = SummaryWriter(log_dir=ftb)
    
    if 'tiny' in args.dataset:
        stride = 2
    else:
        stride = 1
        
    if 'preactres18' == args.arch:
        model = nn.DataParallel(preactresnet.PreActResNet18(num_classes=args.num_classes, stride=stride).cuda())
    elif 'preactres34' == args.arch:
        model = nn.DataParallel(preactresnet.PreActResNet34(num_classes=args.num_classes, stride=stride).cuda())
    elif 'lenet' == args.arch:
        model = nn.DataParallel(lenet.Lenet5(num_classes=args.num_classes).cuda())
    elif 'wrn34-10' == args.arch:
        model = nn.DataParallel(wideresnet.wrn34_10(num_classes=args.num_classes, stride=stride).cuda())
    elif 'wrn28-10' == args.arch:
        model = nn.DataParallel(wideresnet.wrn28_10(num_classes=args.num_classes, stride=stride).cuda())
    elif 'resnet18' == args.arch:
        model = nn.DataParallel(resnet.ResNet18(num_classes=args.num_classes, stride=stride).cuda())
    elif 'resnet34' == args.arch:
        model = nn.DataParallel(resnet.ResNet34(num_classes=args.num_classes, stride=stride).cuda())
    elif 'resnet50' == args.arch:
        model = nn.DataParallel(resnet.ResNet50(num_classes=args.num_classes, stride=stride).cuda())
    elif 'resnet101' == args.arch:
        model = nn.DataParallel(resnet.ResNet101(num_classes=args.num_classes, stride=stride).cuda())
    elif 'resnet152' == args.arch:
        model = nn.DataParallel(resnet.ResNet152(num_classes=args.num_classes, stride=stride).cuda())
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    
    trainset, testset = get_dataset(args)
    trainloader = DataLoader(trainset, batch_size=args.batch_size*3, shuffle=True, drop_last=True, num_workers=20, pin_memory=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, drop_last=False, num_workers=20, pin_memory=True)
    
    if args.loss == 'xent':
        criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    elif args.loss == 'bce':
        criterion = nn.BCELoss().cuda()
    best_acc = 0
    
    if args.train:
    
        for epoch in range(1, args.epoch+1):
            avg_loss = training(epoch, model, trainloader, optimizer, criterion, tb, args)
            tb.add_scalar('train_avg_loss', avg_loss, epoch)
        
            val_loss, val_accuracy = evaluation(model, testloader)
            print('Accuracy @ %d epochs: %.4f' % (epoch, val_accuracy))
            tb.add_scalar('val_accuracy', val_accuracy, epoch)
            tb.add_scalar('val_loss', val_loss, epoch)
        
            is_best = val_accuracy > best_acc
            states = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'accuracy': val_accuracy,
                      'seed': args.seed}
            torch.save(states, os.path.join(savepoint, 'lastcheckpoint.pth.tar'))
            if epoch > (args.epoch-10):
                torch.save(states, os.path.join(savepoint, 'checkpoint-%depochs.pth.tar' % epoch))
            if is_best:
                best_acc = val_accuracy
                torch.save(states, os.path.join(savepoint, 'bestcheckpoint.pth.tar'))
        
            scheduler(epoch, args.lr_decay, optimizer)
    else:
        state_dict = torch.load(args.pretrained)['state_dict']
        model.load_state_dict(state_dict)
        test_loss, test_accuracy = evaluation(model, testloader)
        print('accuracy: %.4f\nloss: %4f' % (test_accuracy, test_loss))