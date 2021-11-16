'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from net import resnet20
from utils import progress_bar
import numpy as np
import random


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=0.3, type=float, help='alpha')
parser.add_argument('--lambdaa', default=0.1, type=float, help='lambda')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--loss', type=str, default='gm', help='the type of loss')
parser.add_argument('--eval', '-e', action='store_true', help='eval')
parser.add_argument('--finetune', '-f', action='store_true', help='finetune')
args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  
np.random.seed(args.seed)
random.seed(args.seed) 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.makedirs('./checkpoint/', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.loss == 'gm':
    net = resnet20(gm=True, num_classes=10)
else:
    net = resnet20(gm=False, num_classes=10)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


def loss_func(logits, labels, alpha):
    dists = logits * (-2)
    loss_lkd = torch.gather(dists, dim=1, index=labels.view(-1, 1))
    margin = 1 + torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), alpha)
    loss_ce = F.cross_entropy(logits * margin, labels, reduce=False)
    return loss_ce, loss_lkd


criterion = loss_func
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                     momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_loss_ce = 0
    train_loss_lkd = 0
    correct = 0
    total = 0

    if epoch <= 30:
        lambdaa = args.lambdaa * 0.01 + (epoch/30) *  args.lambdaa * 0.99
        alpha = args.alpha * 0.1 + (epoch/30) *  args.alpha * 0.9
    else:
        lambdaa = args.lambdaa
        alpha = args.alpha


    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, outputs = net(inputs)
        if args.loss == 'gm':
            loss_ce, loss_lkd = criterion(outputs, targets, alpha=alpha)
            loss = loss_ce + lambdaa * loss_lkd
            loss = loss.mean()
        else:
            loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if args.loss == 'gm':
            train_loss_ce += loss_ce.mean().item()
            train_loss_lkd += loss_lkd.mean().item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if args.loss == 'gm':
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f, CE: %.3f, LKD: %.3f| Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), train_loss_ce/(batch_idx+1), train_loss_lkd/(batch_idx+1), 100.*correct/total, correct, total))
        else:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'
                         % (100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    
    return acc

if not args.eval:
    best_acc = 0
    best_epoch = 0
    for epoch in range(start_epoch, start_epoch+300):
        train(epoch)
        acc = test()
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'./checkpoint/ckpt_cifar10_{args.loss}_a{args.alpha}_l{args.lambdaa}.pth')
            best_acc = acc
            best_epoch = epoch
        scheduler.step()
    print('Best Accuracy: {:.6f} - Epoch {:d}'.format(best_acc, best_epoch))
else:
    net.eval()
    if args.loss == 'gm':
        checkpoint = torch.load(f'./checkpoint/ckpt_cifar10_{args.loss}_a{args.alpha}_l{args.lambdaa}.pth')
    acc = checkpoint['acc']
    net.load_state_dict(checkpoint['net'])
    acc = test()
    print('Test Accuracy: {:.6f}'.format(acc))
    