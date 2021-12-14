import torch.nn as nn
import torch.nn.functional as F
import torch
from augerino.utils import expm


class UniformAug(nn.Module):
    """Augerino layer

    Samples uniformly from the Lie Algebra Aff(2), containing translations,
    rotations, uniform scaling, stretching and shearing. At initialization,
    the width of the uniform distribution is 0, which means this layer is
    equivalent to the Identity.

    Parameters
    ----------
    trans_scale: float, optional
        Scale used for translation (between 0 and 1).

    """
    def __init__(self, trans_scale=0.1):
        super(UniformAug, self).__init__()

        self.trans_scale = trans_scale

        self.width = nn.Parameter(torch.zeros(6))
        self.softplus = torch.nn.Softplus()
        self.g0 = None
        self.std_batch_size = None

    def set_width(self, vals):
        """Set width/bound of uniform sampling

        Parameters
        ----------
        vals : torch.Tensor
            New values of theta_tilde from the paper (eq.9).
        """
        self.width.data = vals

    def transform(self, x):
        """Augments an input by sampling an affine transformation from the
        parametrized distribution.

        Parameters
        ----------
        x : torch.Tenso
            Input batch.

        Returns
        -------
        torch.Tensor
            Augmented batch.
        """
        bs, _, w, h = x.size()
        # sample epsilon for each example in the batch and each generator
        # (eq. 9)
        weights = torch.rand(bs, 6)
        weights = weights.to(x.device, x.dtype)

        # ensure positivity of width paremeter theta (eq. 9)
        width = self.softplus(self.width)

        # compute epsilon * theta (div because epsilon is usually in [-1,1])
        weights = weights * width - width.div(2.)

        # compute the sum of the scaled generator matrices inside the exp
        # in eq. 9
        generators = self.generate(weights)

        # compute exponential map by solving a linear ode
        affine_matrices = expm(generators.cpu()).to(weights.device)

        # spatial sampling from Spatial Tranformer Networks, used to sample
        # uniformly from the Lie Algebra of Aff(2) and apply to x
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

            # rotation
            self.g2 = torch.zeros(3, 3, device=weights.device)
            self.g2[0, 1] = -1.
            self.g2[1, 0] = 1.
            self.g2 = self.g2.unsqueeze(-1).expand(3, 3, bs)

            # uniform scaling
            self.g3 = torch.zeros(3, 3, device=weights.device)
            self.g3[0, 0] = 1.
            self.g3[1, 1] = 1.
            self.g3 = self.g3.unsqueeze(-1).expand(3, 3, bs)

            # stretching
            self.g4 = torch.zeros(3, 3, device=weights.device)
            self.g4[0, 0] = 1.
            self.g4[1, 1] = -1.
            self.g4 = self.g4.unsqueeze(-1).expand(3, 3, bs)

            # shearing
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
