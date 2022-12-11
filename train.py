import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tr
import torchvision.datasets as datasets
import numpy as np


from BYOL.byol import BYOL
from BYOL.utils import criterion, get_byol_transforms, MultiViewDataInjector
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

epochs = 30
batch_size = 128
offset_bs = 256
base_lr = 0.03
tempBase = 0.996

transformT, transformT1, transformEvalT = get_byol_transforms(32, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

#traindt = datasets.ImageNet(root='./data', split = 'train', download=True)
#trainloader = torch.utils.data.DataLoader(traindt, batch_size=batch_size, shuffle=True)

#testdt = datasets.ImageNet(root='./data', split = 'test', download=True)
#testloader = torch.utils.data.DataLoader(traindt, batch_size=128, shuffle=True)

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=MultiViewDataInjector([transformT, transformT1]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=MultiViewDataInjector([transformT, transformT1]))
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

lr = base_lr*batch_size/offset_bs

byol = BYOL(input_size=512, closedFormPredicator=True, backend='resnet34')

byol.to(device)

#read papers:
#https://arxiv.org/pdf/1708.03888v1.pdf (sgd)
#https://arxiv.org/pdf/1608.03983.pdf (cosine decay ) in our case wihout restart

params = byol.parameters()
optimizer = optim.SGD( params, lr=lr, momentum= 0.9, weight_decay=1.5e-4)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=)

def train_loop(model, optimizer, trainloader, transform, transform1, criterion, device):
    tk0 = tqdm(trainloader)
    train_loss = []

    for (x, x1), _ in tk0:
        
        x = x.to(device)
        x1 = x1.to(device)

        onlinex, onlinex1, targetx, targetx1 = model(x, x1)
        loss = criterion(onlinex, targetx1, onlinex1, targetx, True)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        byol.updateTargetNetwork()

    return train_loss


for epoch in range(epochs):
    train_loss = train_loop(byol, optimizer, trainloader, transformT, transformT1, criterion, torch.device('cuda'))
    print(np.mean(train_loss))







