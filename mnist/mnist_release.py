from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import random


class LGM_Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LGM_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

    def forward(self, input):
        xy = torch.matmul(input, self.mu.permute(1,0))
        xx = torch.sum(input * input, dim=1, keepdim=True)
        yy = torch.sum(self.mu * self.mu, dim=1, keepdim=True).permute(1,0)
        out = -0.5 * (xx - 2.0 * xy + yy)
        return out


class Net(nn.Module):
    def __init__(self, edim=2,loss='gm'):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.ip1 = nn.Linear(128 * 3 * 3, edim)

        gain = torch.nn.init.calculate_gain('relu')
        if loss == 'gm':
            logits = LGM_Linear(edim, 10)
            torch.nn.init.kaiming_normal_(logits.mu, gain)
            self.ip2 = logits
        elif loss == 'softmax':
            logits = nn.Linear(edim, 10)
            torch.nn.init.kaiming_normal_(logits.weight, gain)
            torch.nn.init.constant_(logits.bias, 0)
            self.ip2 = logits
        else:
            raise Exception('Wrong type of loss!')

    def forward(self, x, target=None):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
        ip1 = self.ip1(x)
        if target is not None:
            ip2 = self.ip2(ip1, target)
        else:
            ip2 = self.ip2(ip1)
        return ip1, ip2



def loss_func(logits, labels, alpha):
    dists = logits * (-2)
    loss_lkd = torch.mean(dists, dim=1)
    margin = 1 + torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), alpha)
    loss_ce = F.cross_entropy(logits * margin, labels, reduce=False)
    return loss_ce, loss_lkd



def train(args, model, device, train_loader, optimizer, epoch, loss_type, optimizer_center=None, center_loss=None):
    model.train()
    if epoch <= 5:
        lambdaa = args.lambdaa * 0.01 + (epoch/5) *  args.lambdaa * 0.99
        alpha = args.alpha * 0.1 + (epoch/5) *  args.alpha * 0.9
    else:
        lambdaa = args.lambdaa
        alpha = args.alpha
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        ip1, output = model(data)
        if loss_type == 'gm':
            loss_ce, loss_lkd = loss_func(output, target, alpha=alpha)
            loss = loss_ce + lambdaa * loss_lkd
            loss = loss.mean()
        elif loss_type == 'softmax':
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        else:
            raise Exception('Wrong loss type!')
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            if loss_type == 'gm':
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, CE: {:.6f}, LKD: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), loss_ce.mean().item(), loss_lkd.mean().item()))
            else:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def show(model, device, data_loader, loss='gm', mode='train'):
    model.eval()
    p = []
    l = []
    colors = ['b','g','r','c','m','y','k','#7f00ff','#ff7f00','#00ff7f']
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            fea, output = model(data)
            p.append(fea.cpu().numpy())
            l.append(target.cpu().numpy())
    p = np.concatenate(p, 0)
    l = np.concatenate(l, 0)
    for i in range(10):
        idx = np.argwhere(l==i).reshape(-1, )
        p_show = p[idx]
        plt.plot(p_show[:,0], p_show[:,1], '.', color = colors[i], ms=1)
    plt.savefig(f'mnist_{loss}_{mode}.png')
    plt.close()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--edim', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='M',
                        help='alpha (default: 1.)')
    parser.add_argument('--lambdaa', type=float, default=0.1, metavar='M',
                        help='lambdaa (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the trained model')                 
    parser.add_argument('--loss', type=str, default='gm', help='the type of loss')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  
    np.random.seed(args.seed)
    random.seed(args.seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net(edim=args.edim, loss=args.loss).to(device)

    if not args.eval:
        acc_best = -100.0
        best_epoch = 0
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.001, momentum=0.9, nesterov=True)
        scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
        for epoch in range(args.epochs):
            train(args, model, device, train_loader, optimizer, epoch, args.loss)
            acc = test(model, device, test_loader)
            scheduler.step()
            if acc > acc_best:
                acc_best = acc
                best_epoch = epoch
                torch.save(model.state_dict(),f'./checkpoint/mnist_Net{args.edim}_loss_{args.loss}_alpha_{args.alpha}_lambda_{args.lambdaa}_lr_{args.lr}.pt')
        print('Best Accuracy: {:.6f} - Epoch {:d}'.format(acc_best, best_epoch))
        ckt = torch.load(f'./checkpoint/mnist_Net{args.edim}_loss_{args.loss}_alpha_{args.alpha}_lambda_{args.lambdaa}_lr_{args.lr}.pt')
        model.load_state_dict(ckt, strict=True)
        if args.edim == 2:
            show(model, device, test_loader, loss=args.loss, mode='test')
    else:
        ckt = torch.load(f'./checkpoint/mnist_Net{args.edim}_loss_{args.loss}_alpha_{args.alpha}_lambda_{args.lambdaa}_lr_{args.lr}.pt')
        model.load_state_dict(ckt, strict=True)
        acc = test(model, device, test_loader)
        print('Test Accuracy: {:.6f}'.format(acc))



if __name__ == '__main__':
    main()
