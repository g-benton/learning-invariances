import torch
import torch.nn as nn
import torch.nn.functional as F


class BrightnessAug(nn.Module):
    """
    Differetiable brightness adjustment
    """
    def __init__(self):
        super().__init__()
        self.aug=True
        self.log_lims = nn.Parameter(torch.tensor([-2., 2.]))

    @property
    def lims(self):
        return F.sigmoid(self.log_lims) * 2 - 1

    def forward(self, x):
        bs = x.shape[0]
        brightness_change = torch.rand(bs, device=self.lims.device) * (self.lims[1] - self.lims[0]) + self.lims[0]
        brightness_change = brightness_change[:, None, None, None]#.to(x.device)
        return torch.clamp(x + brightness_change, 0, 1)
    
    
class ContrastAug(nn.Module):
    """
    Differetiable contrast adjustment
    """
    def __init__(self):
        super().__init__()
        self.log_lims = nn.Parameter(torch.tensor([-2., 2.]))
    
    @property
    def lims(self):
        return F.sigmoid(self.log_lims) * 2 - 1

    def forward(self, x):
        bs = x.shape[0]
        contrast_change = torch.rand(bs, device=self.lims.device) * (self.lims[1] - self.lims[0]) + self.lims[0]
        contrast_change = contrast_change * 255
        contrast_change = contrast_change[:, None, None, None]
        factor = (259 * (contrast_change + 255)) / (255 * (259 - contrast_change))
        x_ = torch.clamp(factor * (x * 255 - 128) + 128, 0, 255) / 255
        return x_
    
class Normalize(nn.Module):
    """
    Normalization
    """
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(std)[None, :, None, None])
        
    def forward(self, x):
        return (x - self.mean) / self.std
