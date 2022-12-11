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
        transforms.ToTensor(),
        tr.RandomResizedCrop(size=size, scale=(0.08,1), ratio=(3 / 4, 4 / 3)),
        tr.RandomRotation((-90, 90)),
        tr.ColorJitter(),
        tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
        tr.RandomGrayscale(p=0.2),
        tr.Normalize(mean, std),
        ])

    transformT1 = tr.Compose([
        transforms.ToTensor(),
        tr.RandomResizedCrop(size=size, scale=(0.08,1), ratio=(3 / 4, 4 / 3)),
        tr.RandomRotation((-90, 90)),
        tr.ColorJitter(),
        tr.RandomGrayscale(p=0.2),
        tr.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
        tr.Normalize(mean, std),
        ])

    transformEvalT = tr.Compose([
        transforms.ToTensor(),
        tr.CenterCrop(size=size),
        tr.Normalize(mean, std),
        
    ])

    return transformT, transformT1, transformEvalT

from torchvision.transforms import transforms


class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        return output