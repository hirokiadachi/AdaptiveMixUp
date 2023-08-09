'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, stride=1):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(4)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out).flatten(start_dim=1)
        out = self.linear(out)
        return out
    
class AdaptiveMixUp(PreActResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        block = kwargs['block']
        num_classes = kwargs['num_classes']
        self.gate1 = nn.Linear(512*block.expansion, 100)
        self.gate2 = nn.Linear(100, 3)

        ## definition classifier and intrusion discriminator
        self.shared_conv1 = self.conv1
        self.shared_layer1 = self.layer1
        self.shared_layer2 = self.layer2
        self.shared_layer3 = self.layer3
        self.shared_layer4 = self.layer4
        self.shared_avgpool = nn.AvgPool2d(4)
        self.shared_bn = nn.BatchNorm2d(512 * block.expansion)

        self.classifier = nn.Linear(512*block.expansion, num_classes)
        self.extra1 = nn.Linear(512*block.expansion, num_classes)
        self.extra2 = nn.Linear(num_classes, 2)
        
    def policy_generator(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out).flatten(start_dim=1)
        
        out = self.gate1(out)
        out = self.gate2(out)
        policies = torch.softmax(out, dim=1)
        alpha1, delta, alpha2 = torch.chunk(policies, 3, dim=1)
        return alpha1, delta, alpha2

    def shared_block(self, x):
        out = self.shared_conv1(x)
        out = self.shared_layer1(out)
        out = self.shared_layer2(out)
        out = self.shared_layer3(out)
        out = self.shared_layer4(out)

        out = self.shared_bn(out)
        out = self.relu(out)
        out = self.shared_avgpool(out).flatten(start_dim=1)
        return out
        
    
    def forward(self, x, y=None, is_train=True):
        if is_train:
            x1, x2, x3 = torch.chunk(x, 3, dim=0)
            y1, y2, y3 = torch.chunk(y, 3, dim=0)
    
            ## output mixup rate
            m1 = torch.randn(x1.size()).cuda()
            m2 = torch.randn(x2.size()).cuda()
            xs = (m1*x1 + m2*x2) / 2
            #xs = torch.cat([x1, x2], dim=0)
            ys = torch.cat([y1, y2], dim=0)
            alpha1, delta, alpha2 = self.policy_generator(xs)
            eps = torch.FloatTensor(alpha1.size()).uniform_(0,1).cuda()
            lam = alpha1 + eps * delta
    
            ## mixup
            mixed_xs = x1 * lam.view(-1,1,1,1) + x2 * (1 - lam.view(-1,1,1,1))
            mixed_ys = y1 * lam.view(-1,1) + y2 * (1 - lam.view(-1,1))
            unseen_xs = x3
            unseen_ys = y3
    
            ## input classifier
            inputs_x = torch.cat([mixed_xs, x1], dim=0)
            labels_y = torch.cat([mixed_ys, y1], dim=0)
            features_cls = self.shared_block(inputs_x)
            logits = self.classifier(features_cls)
    
            ## manifold intrusion discriminator
            ##### image
            binary_xs_pos = torch.cat([unseen_xs, x1], dim=0)
            binary_xs_neg = mixed_xs
            binary_xs = torch.cat([binary_xs_pos, binary_xs_neg], dim=0)
            ##### label
            binary_ys_pos = torch.ones(binary_xs_pos.size(0)).cuda()
            binary_ys_neg = torch.zeros(binary_xs_neg.size(0)).cuda()
            binary_ys = torch.cat([binary_ys_pos, binary_ys_neg], dim=0)
    
            features_dis = self.shared_block(binary_xs)
            dis_out1 = self.extra1(features_dis)
            dis_out2 = self.extra2(dis_out1)
    
            return lam, logits, labels_y, dis_out2, binary_ys
        else:
            features_cls = self.shared_block(x)
            logits = self.classifier(features_cls)
            return logits


def PreActResNet18(num_classes=10, stride=1):
    return AdaptiveMixUp(block=PreActBlock, num_blocks=[2,2,2,2], num_classes=num_classes, stride=stride)

def PreActResNet34(num_classes=10, stride=1):
    return AdaptiveMixUp(block=PreActBlock, num_blocks=[3,4,6,3], num_classes=num_classes, stride=stride)

def PreActResNet50(num_classes=10, stride=1):
    return AdaptiveMixUp(block=PreActBottleneck, num_blocks=[3,4,6,3], num_classes=num_classes, stride=stride)

def PreActResNet101(num_classes=10, stride=1):
    return AdaptiveMixUp(block=PreActBottleneck, num_blocks=[3,4,23,3], num_classes=num_classes, stride=stride)

def PreActResNet152(num_classes=10, stride=1):
    return AdaptiveMixUp(block=PreActBottleneck, num_blocks=[3,8,36,3], num_classes=num_classes, stride=stride)