import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr

def regression_loss(x, y):
    x = F.normalize(x, dim=1, p=2)
    y = F.normalize(y, dim=1, p=2)
    return 2 -2 * (y@x).sum(dim=-1)

def criterion(xOn, yTg, yOn, xTg):
    return (regression_loss(xOn, yTg) + regression_loss(yOn, xTg)).mean()

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

transformEvalT = tr.Compose([
    tr.CenterCrop(size=(32,32))
])

def get_byol_transforms(size, mean, std):
    transformT = tr.Compose([
    tr.RandomResizedCrop(size=size, scale=(0.08,1), ratio=(3 / 4, 4 / 3)),
    tr.RandomApply(nn.ModuleList([tr.RandomRotation((-90, 90))]), p=0.5),
    tr.RandomApply(nn.ModuleList([tr.ColorJitter()]), p=0.8),
    tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
    #tr.RandomGrayscale(p=0.2),
    tr.Normalize(mean, std)])

    transformT1 = tr.Compose([
        tr.RandomResizedCrop(size=size, scale=(0.08,1), ratio=(3 / 4, 4 / 3)),
        tr.RandomApply(nn.ModuleList([tr.RandomRotation((-90, 90))]), p=0.5),
        tr.RandomApply(nn.ModuleList([tr.ColorJitter()]), p=0.8),
        #tr.RandomGrayscale(p=0.2),
        tr.RandomApply(nn.ModuleList([tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0))]), p=0.1),
        tr.Normalize(mean, std)])

    transformEvalT = tr.Compose([
        tr.CenterCrop(size=size),
        tr.Normalize(mean, std)
    ])

    return transformT, transformT1, transformEvalT