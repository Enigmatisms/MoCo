#-*-coding=utf-8-*-
"""
    CIFAR 10 Linear Evaluation Protocal
    if I can manage to use a cloud server, I will try to run the CIFAR 100
    @author Sentinel 
    @date 2021.9.1 
"""
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(in_chan, 64),
            nn.Dropout(0.1),
            nn.ReLU(True),
            nn.Linear(32, out_chan),
        )

    def forward(self, x):
        return self.out(x)
