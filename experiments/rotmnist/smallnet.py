import torch
import torch.nn as nn

def ConvBNrelu(in_channels,out_channels,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class smallnet(nn.Module):
    """
    Very small CNN
    """
    def __init__(self, in_channels=3, num_targets=10,k=128,dropout=True):
        super().__init__()
        self.num_targets = num_targets
        self.net = nn.Sequential(
            ConvBNrelu(in_channels,k),
            ConvBNrelu(k,k),
            ConvBNrelu(k,2*k),
            nn.MaxPool2d(2),#MaxBlurPool(2*k),
            nn.Dropout2d(.3) if dropout else nn.Sequential(),
            ConvBNrelu(2*k,2*k),
            nn.MaxPool2d(2),#MaxBlurPool(2*k),
            nn.Dropout2d(.3) if dropout else nn.Sequential(),
            ConvBNrelu(2*k,2*k),
            nn.Dropout2d(.3) if dropout else nn.Sequential(),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_targets)
        )
    def forward(self,x):
        return self.net(x)
