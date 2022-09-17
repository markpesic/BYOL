import torch
import torch.nn as nn
import torch.nn.functional as F
from BYOL.Backend import *

availableBackends = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50, 'resnet101':resnet101,'resnet152':resnet152}

class MLP(nn.Module):
    def __init__(self,
    input_size = 2048,
    hidden_size = 4096,
    output_size = 256,
    depth = 2,
    ):  
        super().__init__()
        layers = []
        inp = input_size
        for d in range(depth):
            if d == depth - 1:
                layers.append(nn.Linear(inp, output_size))
            else:
                layers.extend([nn.Linear(inp, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True)])
                inp = hidden_size
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class OnlineNetwork(nn.Module):
    def __init__(self,
    input_size=2048,
    hidden_size=4096,
    output_size=256,
    depth_proj=2,
    depth_pred=2,
    closedFormPredicator = False,
    backend = 'resnet50',
    pretrained = False):
        super().__init__()
        self.backend = availableBackends[backend](pretrained=pretrained)
        self.projection = MLP(input_size = input_size, output_size = output_size, hidden_size = hidden_size, depth = depth_proj)
        self.predictor = None

        if not closedFormPredicator:
            self.predictor = MLP(input_size=output_size, output_size = output_size, hidden_size = hidden_size, depth = depth_pred)

    def forward(self, x):
        x = self.backend(x)
        out = self.projection(x)

        if self.predictor is not None:
            out = self.predictor(out)
        
        return out


class TargetNetwork(nn.Module):
    def __init__(self,
    input_size=2048,
    hidden_size=4096,
    output_size=256,
    depth_proj=2,
    backend = 'resnet50',
    pretrained = False):
        super().__init__()
        self.backend = availableBackends[backend](pretrained=pretrained)
        self.projection = MLP(input_size = input_size, output_size = output_size, hidden_size = hidden_size, depth = depth_proj)
        self.predictor = None


        for model in [self.backend, self.projection]:
            for p in model.parameters():
                p.requires_grad = False

    def forward(self, x):
        out = self.backend(x)
        out = self.projection(out)
        return out

class BYOL(nn.Module):
    def __init__(self,
    input_size=2048,
    hidden_size=4096,
    output_size=256,
    depth_proj=2,
    depth_pred=2,
    closedFormPredicator = False,
    EMA = True,
    t = 0.996,
    backend='resnet50',
    pretrained = False):
        super().__init__()
        self.t = t
        self.EMA = EMA
        self.closedForm = closedFormPredicator
        self.onlineNet = OnlineNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
        depth_proj= depth_proj, depth_pred=depth_pred, closedFormPredicator = closedFormPredicator, backend=backend, pretrained=pretrained)
        self.targetNet = TargetNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, depth_proj=depth_proj, backend=backend, pretrained=pretrained)

    def updateTargetNetwork(self, lamb = 10):
        with torch.no_grad():
            for pOn,pTa in zip(self.onlineNet.parameters(), self.targetNet.parameters()):
                if self.EMA and not self.closedForm :
                    pTa = self.t*pTa + (1. - self.t)*pOn
                else:
                    pTa = lamb*pOn

    def forward(self, x, y):
        xOn = self.onlineNet(x)
        yOn = self.onlineNet(y)
        xTg = self.targetNet(x)
        yTg = self.targetNet(y)

        if self.closedForm:
            xOn = (xOn.T@yTg)/(xOn.T@xOn)
            yOn = (yOn.T@xTg)/(yOn.T@yOn)

        return xOn, yOn, xTg, yTg

        
    