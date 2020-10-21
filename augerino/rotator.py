import torch
import torch.nn.functional as F
from augerino.utils import expm

def rotator(inputs):
    '''
    just a simple helper function to randomly rotate inputs
    '''
    bs, _, w, h = inputs.size()

    ## rotation generating matrix ##
    g2 = torch.zeros(3, 3)
    g2[0, 1] = -1.
    g2[1, 0] = 1.
    g2 = g2.unsqueeze(-1).expand(3,3, bs)

    ## weight the rotations randomly ##
    upper = 10
    lower = -10
    wghts = upper * torch.rand(bs) - lower
    g2 = wghts * g2
    generators = (g2).transpose(0, 2).transpose(2, 1)

    affine_mats = expm(generators)

    flowgrid = F.affine_grid(affine_mats[:, :2, :], size=inputs.size(),
                        align_corners=True)
    transformed = F.grid_sample(inputs, flowgrid, align_corners=True)

    return transformed
