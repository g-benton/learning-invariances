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
sys.path.append("./hessian_utils/")
import lanczos_algorithm
import density as density_lib

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

def convert_dset_to_tf(model, loader):
    xs = np.array([]).reshape(0, 3, 32, 32)
    ys = np.array([]).reshape(0, 1)
    with torch.no_grad():
        for dt in loader:
            xx, yy = dt
            xx = xx.cuda()
            ys = np.vstack([ys, yy.unsqueeze(-1)])
            
            xx = model.aug(xx).cpu().detach().numpy()
            xs = np.vstack([xs, xx])
#         xs = xs.transpose(0, 2, 3, 1)
        print(xs.shape)
        print(ys.shape)
    return tf.data.Dataset.from_tensor_slices((xs, ys))

def main(args):    

    net = models.layer13s(in_channels=3,num_targets=10)
    augerino = models.UniformAug()
    model = models.AugAveragedModel(net, augerino,ncopies=4)
    model.load_state_dict(torch.load(args.saved_model))
    
    transform = transforms.Compose([
    # you can add other transformations in this list
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.CIFAR10("/datasets/", train=True, download=False,
                                           transform=transform)
    trainloader = DataLoader(dataset, batch_size=128)
    
    
    SCCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    def loss_fn(model, inputs):
        x, y = inputs
        preds = model(x, training=False)
        return SCCE(y, preds)
    
    
    neigs = 50
    input_shape = Variable(torch.rand(1, 3, 32, 32))
    
    model = model.cuda()

    tfdset = convert_dset_to_tf(model, trainloader)
    
    torch.cuda.empty_cache()
    if args.augerino=='T':
        keras_model = pytorch_to_keras(model.model.cpu(), input_shape)
    else:
        keras_model = pytorch_to_keras(model.cpu(), input_shape)

    V, T = lanczos_algorithm.approximate_hessian(
                keras_model,
                loss_fn,
                tfdset.batch(128),
                order=neigs,
                random_seed=1)

    eigs, wghts = density_lib.tridiag_to_eigv([T.numpy()])
    all_eigs = eigs[0, :]
    np.save(args.savename, all_eigs)

    tf.keras.backend.clear_session()
           
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="eigenvalues")

    parser.add_argument(
        "--saved_model",
        type=str,
        default='.saved-outputs/state_dict.pt',
        help="path to saved state dict",
    )
    
    parser.add_argument(
        "--savename",
        type=str,
        default='.saved-outputs/model_eigs.npy',
        help="path and name to saved file",
    )
    args = parser.parse_args()

    main(args)