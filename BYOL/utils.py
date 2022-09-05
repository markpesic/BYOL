import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr

def regression_loss(x, y, closedFormPredicator = False):
    x = F.normalize(x, dim=1, p=2)
    y = F.normalize(y, dim=1, p=2)
    if closedFormPredicator:
        res = 2 - 2*(x).sum(dim=-1)
    else:
        res = 2 -2 * (x*y.T).sum(dim=-1)
    return res

def criterion(xOn, yTg, yOn, xTg, closedFormPredicator = False):
    return (regression_loss(xOn, yTg, closedFormPredicator = closedFormPredicator) + regression_loss(yOn, xTg, closedFormPredicator = closedFormPredicator)).mean()


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