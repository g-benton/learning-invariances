import copy, warnings
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from oil.tuning.study import train_trial
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, islice, FixedNumpySeed, cosLr
from lie_conv.datasets import SpringDynamics
from lie_conv import datasets
from lie_conv.dynamicsTrainer import IntegratedDynamicsTrainer, FC, HLieResNet,Partial

import lie_conv.lieGroups as lieGroups
from lie_conv.lieGroups import Tx
from lie_conv import dynamicsTrainer
#from lie_conv.dynamics_trial import DynamicsTrial
try:
    import lie_conv.graphnets as graphnets
except ImportError:
    import lie_conv.lieConv as graphnets
    warnings.warn('Failed to import graphnets. Please install using \
                `pip install .[GN]` for this functionality', ImportWarning)


import copy
import torch
import torch.nn as nn
from oil.utils.utils import Eval
from oil.model_trainers import Trainer
from lie_conv.hamiltonian import HamiltonianDynamics,EuclideanK
from lie_conv.lieConv import pConvBNrelu, PointConv, Pass, Swish, LieResNet
from lie_conv.moleculeTrainer import BottleBlock, GlobalPool
from lie_conv.utils import Expression, export, Named
import numpy as np
from torchdiffeq import odeint
from lie_conv.lieGroups import T
from augerino.utils import fixed_compute_expm as expm


class Partial(nn.Module):
    def __init__(self,module,*args,**kwargs):
        super().__init__()
        self.module = module
        self.args = args
        self.kwargs = kwargs
    def forward(self,*x):
        self.module.nfe +=1
        return self.module(*x,*self.args,**self.kwargs)

@export
class IntegratedDynamicsTrainer2(IntegratedDynamicsTrainer):

    def _rollout_model(self, z0, ts, sys_params):
        """ inputs [z0: (bs, z_dim), ts: (bs, T), sys_params: (bs, n, c)]
            outputs pred_zs: (bs, T, z_dim) """
        if hasattr(self.model,'integrate'): return self.model.integrate(z0,ts,sys_params,tol=self.hypers['tol'])
        dynamics = Partial(self.model, sysP=sys_params)
        zs = odeint(dynamics, z0, ts[0], rtol=self.hypers['tol'], method='rk4')
        return zs.permute(1, 0, 2)

class LinearUniform2d(nn.Module):
    def __init__(self, trans_scale=.1):
        super().__init__()
        self.lower = nn.Parameter(.1*torch.rand(4))
        self.upper = nn.Parameter(.1*torch.rand(4))
    def affine(self,xyz): #  (bs,n,3), (bs,n,c), (bs,n) 
        bs = xyz.shape[0]
        z = torch.rand(bs,4,dtype=xyz.dtype,device=xyz.device)
        affine_generators = (z*(self.upper-self.lower)+self.lower).reshape(bs,2,2)
        affine_matrices = expm(torch.cat([affine_generators,-affine_generators],dim=0))
        A,Ainv = affine_matrices[:bs],affine_matrices[bs:]
        return A,Ainv

    def log_data(self,logger,step,name):
        print("ub and lb:",self.lower,self.upper)

class AugHLieResNet(HLieResNet):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.aug = LinearUniform2d()
    def integrate(self,z0,ts,sys_params,tol):
        dynamics = Partial(self, sysP=sys_params)
        if self.training:
            A,Ainv = self.aug.affine(z0)
            bs = z0.shape[0]
            Tz0 = (z0.reshape(bs,-1,2)@A).reshape(bs,-1)
            Tzs = odeint(dynamics, Tz0, ts[0], rtol=tol, method='rk4').permute(1, 0, 2)
            zs = (Tzs.reshape(bs,-1,2)@Ainv).reshape(*Tzs.shape)
        else:
            zs = odeint(dynamics, z0, ts[0], rtol=tol, method='rk4').permute(1, 0, 2)
        return zs



def makeTrainer(*,network=AugHLieResNet,net_cfg={},lr=1e-2,n_train=3000,regen=False,dataset=SpringDynamics,
                dtype=torch.float32,device=torch.device('cuda'),bs=200,num_epochs=2,
                trainer_config={}):
    # Create Training set and model
    splits = {'train':n_train,'val':200,'test':2000}
    dataset = dataset(n_systems=10000,regen=regen)
    with FixedNumpySeed(0):
        datasets = split_dataset(dataset,splits)
    model = network(sys_dim=dataset.sys_dim,d=dataset.space_dim,**net_cfg).to(device=device,dtype=dtype)
    # Create train and Dev(Test) dataloaders and move elems to gpu
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,n_train),num_workers=0,shuffle=(k=='train')),
                                device=device,dtype=dtype) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],len(dataloaders['val']))
    # Initialize optimizer and learning rate schedule
    opt_constr = lambda params: Adam(params, lr=lr)
    lr_sched = cosLr(num_epochs)
    return IntegratedDynamicsTrainer2(model,dataloaders,opt_constr,lr_sched,
                                    log_args={'timeFrac':1/4,'minPeriod':0.0},**trainer_config)

Trial = train_trial(makeTrainer)
if __name__=='__main__':
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    defaults['save']=False
    defaults['trainer_config']['early_stop_metric']='val_MSE'
    print(Trial(argupdated_config(defaults,namespace=(dynamicsTrainer,lieGroups,datasets,graphnets))))