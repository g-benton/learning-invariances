import torch.nn as nn
import torch.nn.functional as F
import torch
from augerino.utils import expm


class UniformAug(nn.Module):
    """docstring for MLPAug"""
    def __init__(self, gen_scale=10., trans_scale=0.1, epsilon=1e-3):
        super(UniformAug, self).__init__()

        self.trans_scale = trans_scale

        self.width = nn.Parameter(torch.zeros(6))
        self.softplus = torch.nn.Softplus()
        self.g0 = None
        self.std_batch_size = None

    def set_width(self, vals):
        self.width.data = vals

    def transform(self, x):
        bs, _, w, h = x.size()
        weights = torch.rand(bs, 6)
        weights = weights.to(x.device, x.dtype)
        width = self.softplus(self.width)
        weights = weights * width - width.div(2.)

        generators = self.generate(weights)

        # exponential map
        affine_matrices = expm(generators.cpu()).to(weights.device)

        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size=x.size(),
                                 align_corners=True)
        x_out = F.grid_sample(x, flowgrid, align_corners=True)
        return x_out

    def generate(self, weights):
        """
        return the sum of the scaled generator matrices
        """
        bs = weights.shape[0]

        if self.g0 is None or self.std_batch_size != bs:
            self.std_batch_size = bs

            # tx
            self.g0 = torch.zeros(3, 3, device=weights.device)
            self.g0[0, 2] = 1. * self.trans_scale
            self.g0 = self.g0.unsqueeze(-1).expand(3, 3, bs)

            # ty
            self.g1 = torch.zeros(3, 3, device=weights.device)
            self.g1[1, 2] = 1. * self.trans_scale
            self.g1 = self.g1.unsqueeze(-1).expand(3, 3, bs)

            self.g2 = torch.zeros(3, 3, device=weights.device)
            self.g2[0, 1] = -1.
            self.g2[1, 0] = 1.
            self.g2 = self.g2.unsqueeze(-1).expand(3, 3, bs)

            self.g3 = torch.zeros(3, 3, device=weights.device)
            self.g3[0, 0] = 1.
            self.g3[1, 1] = 1.
            self.g3 = self.g3.unsqueeze(-1).expand(3, 3, bs)

            self.g4 = torch.zeros(3, 3, device=weights.device)
            self.g4[0, 0] = 1.
            self.g4[1, 1] = -1.
            self.g4 = self.g4.unsqueeze(-1).expand(3, 3, bs)

            self.g5 = torch.zeros(3, 3, device=weights.device)
            self.g5[0, 1] = 1.
            self.g5[1, 0] = 1.
            self.g5 = self.g5.unsqueeze(-1).expand(3, 3, bs)

        out_mat = weights[:, 0] * self.g0
        out_mat += weights[:, 1] * self.g1
        out_mat += weights[:, 2] * self.g2
        out_mat += weights[:, 3] * self.g3
        out_mat += weights[:, 4] * self.g4
        out_mat += weights[:, 5] * self.g5

        # transposes just to get everything right
        return out_mat.transpose(0, 2).transpose(2, 1)

    def forward(self, x):
        return self.transform(x)
