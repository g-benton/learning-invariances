
from .layer13 import Expression
import torch.nn as nn
import torch.nn.functional as F


try: from efficientnet_pytorch import EfficientNet
except: EfficientNet=lambda *args,**kwargs:None
def efficientNet(in_channels=3, num_targets=10,suffix='b0'):
    assert in_channels==3, "only 3 input channels supported"
    return nn.Sequential(
        EfficientNet.from_name('efficientnet-'+suffix),
        Expression(lambda x: x[:,:num_targets]),
    )
def EfficientNetB0(in_channels=3, num_targets=10):
    return efficientNet(in_channels=3, num_targets=10,suffix='b0')
def EfficientNetB1(in_channels=3, num_targets=10):
    return efficientNet(in_channels=3, num_targets=10,suffix='b1')
def EfficientNetB2(in_channels=3, num_targets=10):
    return efficientNet(in_channels=3, num_targets=10,suffix='b2')
def EfficientNetB3(in_channels=3, num_targets=10):
    return efficientNet(in_channels=3, num_targets=10,suffix='b3')
def EfficientNetB4(in_channels=3, num_targets=10):
    return efficientNet(in_channels=3, num_targets=10,suffix='b4')
def EfficientNetB5(in_channels=3, num_targets=10):
    return efficientNet(in_channels=3, num_targets=10,suffix='b5')
def EfficientNetB6(in_channels=3, num_targets=10):
    return efficientNet(in_channels=3, num_targets=10,suffix='b6')