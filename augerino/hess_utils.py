import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
tf.enable_v2_behavior()
import numpy as np
import argparse
import glob

import torch
from torch.autograd import Variable
from pytorch2keras import pytorch_to_keras
from augerino import models

import sys

from augerino.spectral_density import lanczos_algorithm
from augerino.spectral_density import density as density_lib

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

def convert_dset_to_tf(loader, model=None):
    xs = np.array([]).reshape(0, 3, 32, 32)
    ys = np.array([]).reshape(0, 1)
    with torch.no_grad():
        for dt in loader:
            xx, yy = dt
            xx = xx
            ys = np.vstack([ys, yy.unsqueeze(-1)])

            if model is not None:
                xx = model.aug(xx.cuda()).cpu().detach().numpy()
            xs = np.vstack([xs, xx])
#         xs = xs.transpose(0, 2, 3, 1)
        print(xs.shape)
        print(ys.shape)
    return tf.data.Dataset.from_tensor_slices((xs, ys))

def loss_fn(model, inputs):
    x, y = inputs
    preds = model(x, training=False)
    SCCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return SCCE(y, preds)


def compute_hess_eigs(model, loader, neigs=50,
                     augerino=False):
    model = model.cuda()
    input_shape = Variable(torch.rand(1, 3, 32, 32))
    if augerino:
        tfdset = convert_dset_to_tf(loader, model)
        keras_model = pytorch_to_keras(model.model.cpu(), input_shape)
    else:
        keras_model = pytorch_to_keras(model.cpu(), input_shape)
        tfdset = convert_dset_to_tf(loader)
        
        
    V, T = lanczos_algorithm.approximate_hessian(
            keras_model,
            loss_fn,
            tfdset.batch(128),
            order=neigs,
            random_seed=1)

    eigs, wghts = density_lib.tridiag_to_eigv([T.numpy()])
    torch.cuda.empty_cache()
    
    return eigs