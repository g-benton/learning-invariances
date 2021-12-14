import torch.nn as nn
import torch.nn.functional as F
import torch


class DiffAug(nn.Module):
    """ """
    def __init__(self):
        super().__init__()
        self.Sigma = nn.Parameter(torch.eye(3))
        self.Mu = nn.Parameter(torch.zeros(3))

    def translateRotate(self, x):
        bs, _, w, h = x.size()
        z = torch.randn(
            bs, 3, device=x.device, dtype=x.dtype) @ self.Sigma + self.Mu
        # Build affine matrices for random translation of each image
        affineMatrices = torch.zeros(bs, 2, 3, device=x.device, dtype=x.dtype)
        affineMatrices[:, 0, 0] = z[:, 2].cos()
        affineMatrices[:, 0, 1] = -z[:, 2].sin()
        affineMatrices[:, 1, 0] = z[:, 2].sin()
        affineMatrices[:, 1, 1] = z[:, 2].cos()
        affineMatrices[:, :2, 2] = z[:, :2]/(.5*w+.5*h)
        affineMatrices = affineMatrices

        flowgrid = F.affine_grid(
            affineMatrices, size=x.size(), align_corners=True)
        x_out = F.grid_sample(x, flowgrid, align_corners=True)
        return x_out

    def forward(self, x):
        return self.translateRotate(x)

    def log_data(self, logger, step, name):
        print(self.Sigma @ self.Sigma.T, self.Mu)

    def __repr__(self):
        return self.__class__.__name__


class AugAveragedModel(nn.Module):
    """Augerino super-model

    It encapsulates a backbone neural network with an augmentation layer in the
    front-end and an averaging layer on the back-end.

    Parameters
    ----------
    model: torch.nn.Module
        Backbone network.
    aug: torch.nn.Moduel
        Augmentation layer, e.g. ``augerino.models.UniformAug``.
    n_copies: int, optional
        Number of batch copies made before augmentation. Only used for
        inference, not for training (n_copies=1). Defaults to 4.
    """
    def __init__(self, model, aug, ncopies=4):
        super().__init__()
        self.aug = aug
        self.model = model
        self.ncopies = ncopies

    def forward(self, x):
        if self.training:
            # In training mode, the batch is not copied:
            # We just augment and compute the backbone forward
            return self.model(self.aug(x))
        else:
            # Replicate the batch n_copies times, augment each of them and
            # concatenate
            bs = x.shape[0]
            aug_x = torch.cat(
                [self.aug(x) for _ in range(self.ncopies)], dim=0)

            # Compute the forward of the augmented super-batch, the logits,
            # and averages over n_copies
            return sum(
                torch.split(F.log_softmax(self.model(aug_x), dim=-1), bs)
            ) / self.ncopies
