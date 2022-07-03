from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from BYOL.Backend import resnet18

class MLP(nn.Module):
    def __init__(self,
    input_size = 2048,
    hidden_size = 4096,
    output_size = 256,
    depth = 2,
    normalizationFinalLayer = False
    ):  
        super().__init__()
        layers = []
        inp = input_size
        for d in range(depth):
            if d == depth - 1 and normalizationFinalLayer:
                layers.append(nn.Linear(inp, output_size))
            elif d == depth - 1 and not normalizationFinalLayer:
                layers.extend([nn.Linear(inp, output_size), nn.BatchNorm1d(output_size), nn.ReLU(inplace=True)])
            else:
                layers.extend([nn.Linear(inp, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True)])
                inp = hidden_size
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        #print(x.shape)
        return self.layer(x)

class OnlineNetwork(nn.Module):
    def __init__(self,
    input_size=2048,
    hidden_size=4096,
    output_size=256,
    depth_proj=2,
    depth_pred=2,
    closedFormPredicator = False):
        super().__init__()
        self.backend = resnet18(pretrained=False)
        self.projection = MLP(input_size = input_size, output_size = output_size, hidden_size = hidden_size, depth = depth_proj,  normalizationFinalLayer=True)
        self.predictor = None

        if not closedFormPredicator:
            self.predictor = MLP(input_size=output_size, output_size = output_size, hidden_size = hidden_size, depth = depth_pred, normalizationFinalLayer=False)

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
    depth_proj=2):
        super().__init__()
        self.backend = resnet18(pretrained=False)
        self.projection = MLP(input_size = input_size, output_size = output_size, hidden_size = hidden_size, depth = depth_proj,  normalizationFinalLayer=True)
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
    EAvg = True,
    t = 0.996):
        super().__init__()
        self.t = t
        self.eAvg = EAvg
        self.onlineNet = OnlineNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
        depth_proj= depth_proj, depth_pred=depth_pred, closedFormPredicator = closedFormPredicator)
        self.targetNet = TargetNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, depth_proj=depth_proj)

    def updateTargetNetwork(self, EAvg=True, lamb = 10):
        with torch.no_grad():
            for pOn,pTa in zip(self.onlineNet.parameters(), self.targetNet.parameters()):
                if self.eAvg:
                    pTa = self.t*pTa + (1-self.t)*pOn
                else:
                    pTa = lamb*pOn

    def forward(self, x, y):
        x1 = self.onlineNet(x)
        y1 = self.onlineNet(y)
        x2 = self.targetNet(x)
        y2 = self.targetNet(y)

        return x1, y1, x2, y2

        
    