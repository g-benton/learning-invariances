"""
Adapted from https://github.com/odegeasslbc/Differentiable-RGB-to-HSV-convertion-pytorch/blob/master/pytorch_hsv.py
Needs to be debugged!
"""

import torch
from torch import nn
from torch.nn import functional as F


# class HSVLoss(nn.Module):
#     def __init__(self, h=0, s=1, v=0.7, eps=1e-7, threshold_h=0.03, threshold_sv=0.1):
#         super(HSVLoss, self).__init__()
#         self.hsv = [h, s, v]
#         self.loss = nn.L1Loss(reduction='none')
#         self.eps = eps

#         # since Hue is a circle (where value 0 is equal to value 1 that are both "red"), 
#         # we need a threshold to prevent the gradient explod effect
#         # the smaller the threshold, the optimal hue can to more close to target hue
#         self.threshold_h = threshold_h
#         # since Hue and (Value and Satur) are conflict when generated image' hue is not the target Hue, 
#         # we have to condition V to prevent it from interfering the Hue loss
#         # the larger the threshold, the ealier to activate V loss
#         self.threshold_sv = threshold_sv

def rgb2hsv(self, im):
    img = im * 0.5 + 0.5
    hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)

    hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
    hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
    hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

    hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    hue = hue/6

    saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + self.eps )
    saturation[ img.max(1)[0]==0 ] = 0

    value = img.max(1)[0]
    return hue, saturation, value

def hsv2rgb(hsv):
    C = hsv[2] * hsv[1]
    X = C * ( 1 - abs( (hsv[0]*6)%2 - 1 ) )
    m = hsv[2] - C

    if self.hsv[0] < 1/6:
        R_hat, G_hat, B_hat = C, X, 0
    elif self.hsv[0] < 2/6:
        R_hat, G_hat, B_hat = X, C, 0
    elif self.hsv[0] < 3/6:
        R_hat, G_hat, B_hat = 0, C, X
    elif self.hsv[0] < 4/6:
        R_hat, G_hat, B_hat = 0, X, C
    elif self.hsv[0] < 5/6:
        R_hat, G_hat, B_hat = X, 0, C
    elif self.hsv[0] <= 6/6:
        R_hat, G_hat, B_hat = C, 0, X

    R, G, B = (R_hat+m), (G_hat+m), (B_hat+m)

    return R, G, B
    
    
#     def forward(self, input):
#         h, s, v = self.get_hsv(input)

#         target_h = torch.Tensor(h.shape).fill_(self.hsv[0]).to(input.device).type_as(h)
#         target_s = torch.Tensor(s.shape).fill_(self.hsv[1]).to(input.device).type_as(s)
#         target_v = torch.Tensor(v.shape).fill_(self.hsv[2]).to(input.device).type_as(v)

#         loss_h = self.loss(h, target_h)
#         loss_h[loss_h<self.threshold_h] = 0.0
#         loss_h = loss_h.mean()

#         if loss_h < self.threshold_h*3:
#             loss_h = torch.Tensor([0]).to(input.device)
        
#         loss_s = self.loss(s, target_s).mean()
#         if loss_h.item() > self.threshold_sv:   
#             loss_s = torch.Tensor([0]).to(input.device)

#         loss_v = self.loss(v, target_v).mean()
#         if loss_h.item() > self.threshold_sv:   
#             loss_v = torch.Tensor([0]).to(input.device)

#         return loss_h + 4e-1*loss_s + 4e-1*loss_v