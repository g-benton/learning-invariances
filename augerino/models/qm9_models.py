import torch.nn as nn
import torch.nn.functional as F
import torch
from augerino.utils import fixed_compute_expm as expm


def cross_matrix(k):
    K = torch.zeros(*k.shape[:-1],3,3,device=k.device,dtype=k.dtype)
    K[...,0,1] = -k[...,2]
    K[...,0,2] = k[...,1]
    K[...,1,0] = k[...,2]
    K[...,1,2] = -k[...,0]
    K[...,2,0] = -k[...,1]
    K[...,2,1] = k[...,0]
    return K

def shear_matrix(k):
    K = torch.zeros(*k.shape[:-1],3,3,device=k.device,dtype=k.dtype)
    K[...,0,1] = k[...,2]
    K[...,0,2] = k[...,1]
    K[...,1,0] = k[...,2]
    K[...,1,2] = k[...,0]
    K[...,2,0] = k[...,1]
    K[...,2,1] = k[...,0]
    return K

def squeeze_matrix(k):
    K = torch.zeros(*k.shape[:-1],3,3,device=k.device,dtype=k.dtype)
    K[...,0,0] = k[...,0]+k[...,2] # squeeze + scale
    K[...,1,1] = -k[...,0]+k[...,1]+k[...,2]
    K[...,2,2] = -k[...,1]+k[...,2]
    return K


class AffineUniform3d(nn.Module):
    def __init__(self, trans_scale=.5):
        super().__init__()
        self._width = nn.Parameter(.54*torch.ones(12))
        self.trans_scale=trans_scale
    def forward(self,inp):
        xyz,vals,mask = inp #  (bs,n,3), (bs,n,c), (bs,n) 
        bs = xyz.shape[0]
        z = torch.rand(bs,12).to(xyz.device,xyz.dtype)*F.softplus(self._width)
        affine_generators = torch.zeros(bs,4,4,dtype=xyz.dtype,device=xyz.device)
        affine_generators[:,:3,:3] += cross_matrix(z[:,:3])+shear_matrix(z[:,3:6])+squeeze_matrix(z[:,6:9])
        affine_generators[:,:3,3] += z[:,9:]
        affine_matrices = expm(affine_generators)
        transformed_xyz = xyz@affine_matrices[:,:3,:3] + affine_matrices[:,None,:3,3]*self.trans_scale
        return transformed_xyz,vals,mask

    # def log_data(self,logger,step,name):
    #     print("ub and lb:",self.lower,self.upper)

import torch
import torch.nn as nn
from oil.model_trainers import Trainer
from lie_conv.lieConv import PointConv, Pass, Swish, GlobalPool
from lie_conv.lieConv import norm, LieResNet, BottleBlock
from lie_conv.utils import export, Named
from lie_conv.datasets import SO3aug, SE3aug
from lie_conv.lieGroups import SE3
import numpy as np

@export 
class MolecLieResNet(LieResNet):
    def __init__(self, num_species, charge_scale, aug=False, augerino=False, group=SE3,ncopies=1, **kwargs):
        super().__init__(chin=3*num_species,num_outputs=1,group=group,ds_frac=1,**kwargs)
        self.charge_scale = charge_scale
        self.aug =aug
        self.ncopies = ncopies
        self.augmentation = AffineUniform3d() if augerino else SE3aug()

    def featurize(self, mb):
        charges = mb['charges'] / self.charge_scale
        c_vec = torch.stack([torch.ones_like(charges),charges,charges**2],dim=-1) # 
        one_hot_charges = (mb['one_hot'][:,:,:,None]*c_vec[:,:,None,:]).float().reshape(*charges.shape,-1)
        atomic_coords = mb['positions'].float()
        atom_mask = mb['charges']>0
        #print('orig_mask',atom_mask[0].sum())
        return (atomic_coords, one_hot_charges, atom_mask)
    def forward(self,mb):
        x = self.featurize(mb)
        if self.training or not self.aug:
            x = self.augmentation(x) if self.aug else x
            return super().forward(x).squeeze(-1)
        else:
            bs = x[0].shape[0]
            aug_xyz = torch.cat([self.augmentation(x)[0] for _ in range(self.ncopies)],dim=0)
            aug_vals = torch.cat([x[1] for _ in range(self.ncopies)],dim=0)
            aug_mask = torch.cat([x[2] for _ in range(self.ncopies)],dim=0)
            aug_x = (aug_xyz,aug_vals,aug_mask)
            return sum(torch.split(super().forward(aug_x).squeeze(-1),bs))/self.ncopies
            #return (sum(super().forward(self.augmentation(x)).squeeze(-1) for _ in range(self.ncopies))/self.ncopies)

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, islice, cosLr, FixedNumpySeed
from oil.tuning.args import argupdated_config
from oil.tuning.study import train_trial
from oil.utils.parallel import try_multigpu_parallelize
from lie_conv.datasets import QM9datasets
from corm_data.collate import collate_fn
from lie_conv.moleculeTrainer import MoleculeTrainer
from oil.datasetup.datasets import split_dataset
import lie_conv.moleculeTrainer as moleculeTrainer
import lie_conv.lieGroups as lieGroups
import functools
import copy
import pandas as pd

class RegMoleculeTrainer(MoleculeTrainer):
    def loss(self,minibatch):
        if isinstance(self.model.augmentation,AffineUniform3d):
            width = F.softplus(self.model.augmentation._width)
            return super().loss(minibatch)-.001*(width[width<10]**2).sum()
            #return super().loss(minibatch)-.001*((upper-lower)[(upper-lower).abs()<10]**2).sum()
        else: return super().loss(minibatch)

def makeTrainer(*, task='homo', device='cuda', lr=3e-3, bs=75, num_epochs=500,network=MolecLieResNet, 
                net_config={'k':1536,'nbhd':100,'act':'swish','group':lieGroups.T(3),'fill':1.0,
                'bn':True,'aug':True,'augerino':True,'ncopies':1,'mean':True,'num_layers':6},save=False,name='',
                subsample=False, trainer_config={'log_dir':None,'log_suffix':'augerino','log_args':{'timeFrac':1/4,'minPeriod':0}}):
    # Create Training set and model
    device = torch.device(device)
    with FixedNumpySeed(0):
        datasets, num_species, charge_scale = QM9datasets()
        if subsample: datasets.update(split_dataset(datasets['train'],{'train':subsample}))
    ds_stats = datasets['train'].stats[task]
    model = network(num_species,charge_scale,**net_config).to(device)
    # Create train and Val(Test) dataloaders and move elems to gpu
    dataloaders = {key:LoaderTo(DataLoader(dataset,batch_size=bs,num_workers=0,
                    shuffle=(key=='train'),pin_memory=False,collate_fn=collate_fn,drop_last=True),
                    device) for key,dataset in datasets.items()}
    # subsampled training dataloader for faster logging of training performance
    dataloaders['Train'] = dataloaders['train']#islice(dataloaders['train'],len(dataloaders['train'])//10)
    
    # Initialize optimizer and learning rate schedule
    opt_constr = functools.partial(Adam, lr=lr)
    cos = cosLr(num_epochs)
    lr_sched = lambda e: min(e / (.01 * num_epochs), 1) * cos(e)
    return RegMoleculeTrainer(model,dataloaders,opt_constr,lr_sched,
                            task=task,ds_stats=ds_stats,**trainer_config)
