import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from roc_util import clamp, lower_limit, upper_limit, std
import numpy as np
from sklearn.neighbors import KernelDensity
from roc_util import score_samples, compute_roc
import torchvision.datasets as datasets
from net import resnet50
import argparse
import os

parser = argparse.ArgumentParser(description='Compute the AUC score for ResNet50 on Imagenet against FGSM attack')
parser.add_argument('--data', default='./ImageNet2012/', type=str, help='the root path of the Imagenet Dataset')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
valdir = os.path.join(args.data, 'val')
traindir = os.path.join(args.data, 'train')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=64, shuffle=False,
        num_workers=3, pin_memory=True)

train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=40, shuffle=True,
        num_workers=3, pin_memory=True)

# Model
print('==> Building model..')
net = resnet50(pretrained=True)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
net = net.to(device)

def fgsm(test_loader, model, epsilon):
    out_feas_attack = []
    out_feas_clean = []
    out_labels = []
    preds_test_adv = []
    preds_test = []
    total = 0
    correct = 0
    for _, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        out_labels.append(y.detach().cpu().numpy())
        delta = torch.zeros_like(X).cuda()
        delta.requires_grad = True
        out_fea, output = model(X + delta[:X.size(0)])
        _, pred_test = output.max(1)
        preds_test.append(pred_test.detach().cpu().numpy())
        out_feas_clean.append(out_fea.detach().cpu().numpy())
        #y_target = (y + torch.randint_like(y, low=1, high=10)) % 10
        y_target = y
        loss = F.cross_entropy(output, y_target)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = delta + epsilon * torch.sign(grad)
        delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
        delta = delta.detach()
        out_fea, output = model(X + delta[:X.size(0)])
        _, predicted = output.max(1)
        preds_test_adv.append(predicted.detach().cpu().numpy())
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        out_feas_attack.append(out_fea.detach().cpu().numpy())
    
    out_feas_clean = np.concatenate(out_feas_clean, 0)
    out_feas_attack = np.concatenate(out_feas_attack, 0)
    out_labels = np.concatenate(out_labels, 0)
    preds_test = np.concatenate(preds_test, 0)
    preds_test_adv = np.concatenate(preds_test_adv, 0)
    acc = correct / total
    return acc, out_feas_attack, out_feas_clean, out_labels, preds_test, preds_test_adv

net.eval()
epsilon = 0.1 / std
acc_attack, X_test_adv, X_test, Y_test, preds_test, preds_test_adv = fgsm(val_loader, net, epsilon)
print('Robust Acc:{:.3f}'.format(acc_attack))
inds_correct = np.where(preds_test == Y_test)[0]
inds_adv_success = 1 - np.where(preds_test_adv == Y_test)[0]
train_feas = []
train_labels = []
preds_train = []
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        out_fea, outputs = net(inputs)
        preds_train.append(outputs.max(1)[1].detach().cpu().numpy())
        train_feas.append(out_fea.detach().cpu().numpy())
        train_labels.append(targets.detach().cpu().numpy())
        if batch_idx % 1000 == 0 and batch_idx > 0:
            print(f'Inference {batch_idx * 40} training items!')
        if batch_idx >= 10000:
            break
X_train = np.concatenate(train_feas, 0)
Y_train = np.concatenate(train_labels, 0)
preds_train = np.concatenate(preds_train, 0)
inds_correct_train = np.where(preds_train == Y_train)[0]

X_test = X_test[inds_correct]
X_test_adv = X_test_adv[inds_correct]
preds_test = preds_test[inds_correct]
preds_test_adv = preds_test_adv[inds_correct]

print('Training KDEs...')
class_inds = {}
for i in range(1000):
    class_inds[i] = np.where(Y_train == i)[0]
kdes = {}
for i in range(1000):
    kdes[i] = KernelDensity(kernel='gaussian',
                            bandwidth=1.5) \
        .fit(X_train[class_inds[i]])
# Get model predictions
print('Computing model predictions...')
densities_normal = score_samples(
    kdes,
    X_test,
    preds_test
)    
densities_adv = score_samples(
    kdes,
    X_test_adv,
    preds_test_adv
)
_, _, auc_score = compute_roc(
        probs_neg=densities_adv, 
        probs_pos=densities_normal) 
print('AUC score: {:.3f}'.format(auc_score))
