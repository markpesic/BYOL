import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tr
import torchvision.datasets as datasets
import numpy as np


from BYOL.byol import BYOL

from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

epochs = 30
batch_size = 128
offset_bs = 256
base_lr = 0.03
tempBase = 0.996

transformT = tr.Compose([
    tr.RandomResizedCrop(size=32, scale=(0.08,1), ratio=(3 / 4, 4 / 3)),
    tr.RandomApply(nn.ModuleList([tr.RandomRotation((-90, 90))]), p=0.5),
    tr.RandomApply(nn.ModuleList([tr.ColorJitter()]), p=0.8),
    tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
    #tr.RandomGrayscale(p=0.2),
    tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

transformT1 = tr.Compose([
    tr.RandomResizedCrop(size=32, scale=(0.08,1), ratio=(3 / 4, 4 / 3)),
    tr.RandomApply(nn.ModuleList([tr.RandomRotation((-90, 90))]), p=0.5),
    tr.RandomApply(nn.ModuleList([tr.ColorJitter()]), p=0.8),
    #tr.RandomGrayscale(p=0.2),
    tr.RandomApply(nn.ModuleList([tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0))]), p=0.1),
    tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

transformEvalT = tr.Compose([])

#traindt = datasets.ImageNet(root='./data', split = 'train')
#trainloader = torch.utils.data.DataLoader(traindt, batch_size=batch_size, shuffle=True)

#testdt = datasets.ImageNet(root='./data', split = 'val')
#testloader = torch.utils.data.DataLoader(traindt, batch_size=128, shuffle=True)

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=tr.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=tr.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

lr = base_lr*batch_size/offset_bs

byol = BYOL(input_size=512, closedFormPredicator=True)

byol.to(device)

#read papers:
#https://arxiv.org/pdf/1708.03888v1.pdf (sgd)
#https://arxiv.org/pdf/1608.03983.pdf (cosine decay ) in our case wihout restart

params = byol.parameters()
optimizer = optim.SGD( params, lr=lr, momentum= 0.9, weight_decay=1.5e-4)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=)

def regression_loss(x, y):
    x = F.normalize(x, dim=1, p=2)
    y = F.normalize(y, dim=1, p=2)
    return 2 -2 * (y@x).sum(dim=-1)

def criterion(xOn, yTg, yOn, xTg):
    return (regression_loss(xOn, yTg) + regression_loss(yOn, xTg)).mean()

def train_loop(model, optimizer, trainloader, transform, transform1, criterion, device):
    tk0 = tqdm(trainloader)
    train_loss = []

    for batch, _ in tk0:
        batch = batch.to(device)
        
        x = transform(batch)
        x1 = transform1(batch)

        onlinex, onlinex1, targetx, targetx1 = model(x, x1)
        loss = criterion(onlinex, targetx1, onlinex1, targetx)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        byol.updateTargetNetwork()

        del batch, x, x1, onlinex, onlinex1, targetx, targetx1
    return train_loss


for epoch in range(epochs):
    train_loss = train_loop(byol, optimizer, trainloader, transformT, transformT1, criterion, torch.device('cuda'))
    print(np.mean(train_loss))







