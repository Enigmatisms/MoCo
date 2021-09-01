#-*-coding=utf-8-*-
"""
    Momentum Contrast Encoder for CIFAR 10
    @author Sentinel 
    @date 2021.9.1 
"""

import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.linear import Linear

def makeConvBlock(in_chan:int, out_chan:int, k:int, use_norm = True, pool = False):
    pad = k // 2
    bloc = [nn.Conv2d(in_chan, out_chan, k, stride = 1, padding = pad)]
    if use_norm == True:
        bloc.append(nn.BatchNorm2d(out_chan))
    bloc.append(nn.ReLU(True))
    if pool == True:
        bloc.append(nn.AvgPool2d(2))
    return nn.Sequential(*bloc)

class Encoder(nn.Module):
    def __init__(self, use_bn = False):
        super().__init__()
        self.conv1 = makeConvBlock(3, 64, 5, use_bn, True)               # out 32 * 16 * 16
        self.conv2 = makeConvBlock(64, 64, 3, use_bn, True)              # out 64 * 8 * 8
        self.convs = nn.ModuleList([makeConvBlock(64, 64, 3, use_bn, False) for i in range(3)])
        self.out_conv = makeConvBlock(64, 16, 3, use_bn, True)           # out 16 * 4 * 4
        self.lin = nn.Linear(256, 128)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.normal_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        for i in range(3):
            x = self.convs[i](x)
        x = self.out_conv(x)
        x = self.lin(x.contiguous().view(-1, 256))
        x = x / x.norm(dim = -1).view(-1, 1)        # L2 normalization
        return x


    def paramUpdate(self, param, m):
        for px, py in zip(self.parameters(), param):
            px = m * px + (1.0 - m) * py

class BaseLine(Encoder):
    def __init__(self, use_bn = False):
        super().__init__(use_bn)
        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 10)
        )
    
    def forward(self, x) -> torch.Tensor:
        x = super().forward(x)
        return self.out(x)


