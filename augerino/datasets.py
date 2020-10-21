import math
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import h5py
import os
from torch.utils.data import Dataset
from .utils import Named, export, Expression, FixedNumpySeed, RandomZrotation, GaussianNoise
from oil.datasetup.datasets import EasyIMGDataset
from oil.datasetup import augLayers
from torchdiffeq import odeint_adjoint as odeint
import torchvision


#ModelNet40 code adapted from
#https://github.com/DylanWusee/pointconv_pytorch/blob/master/data_utils/ModelNetDataLoader.py

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = []
    return (data, label, seg)

def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label

def load_data(dir,classification = False):
    data_train0, label_train0,Seglabel_train0  = load_h5(dir + 'ply_data_train0.h5')
    data_train1, label_train1,Seglabel_train1 = load_h5(dir + 'ply_data_train1.h5')
    data_train2, label_train2,Seglabel_train2 = load_h5(dir + 'ply_data_train2.h5')
    data_train3, label_train3,Seglabel_train3 = load_h5(dir + 'ply_data_train3.h5')
    data_train4, label_train4,Seglabel_train4 = load_h5(dir + 'ply_data_train4.h5')
    data_test0, label_test0,Seglabel_test0 = load_h5(dir + 'ply_data_test0.h5')
    data_test1, label_test1,Seglabel_test1 = load_h5(dir + 'ply_data_test1.h5')
    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])
    train_Seglabel = np.concatenate([Seglabel_train0,Seglabel_train1,Seglabel_train2,Seglabel_train3,Seglabel_train4])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    test_Seglabel = np.concatenate([Seglabel_test0,Seglabel_test1])

    if classification:
        return train_data, train_label, test_data, test_label
    else:
        return train_data, train_Seglabel, test_data, test_Seglabel


@export
class ModelNet40(Dataset,metaclass=Named):
    ignored_index = -100
    class_weights = None
    stratify=True
    num_targets=40
    classes=['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
        'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
        'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
        'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
        'wardrobe', 'xbox']
    default_root_dir = '~/datasets/ModelNet40/'
    def __init__(self,root_dir=default_root_dir,train=True,transform=None,size=1024):
        super().__init__()
        #self.transform = torchvision.transforms.ToTensor() if transform is None else transform
        train_x,train_y,test_x,test_y = load_data(os.path.expanduser(root_dir),classification=True)
        self.coords = train_x if train else test_x
        # SWAP y and z so that z (gravity direction) is in component 3
        self.coords[...,2] += self.coords[...,1]
        self.coords[...,1] = self.coords[...,2]-self.coords[...,1]
        self.coords[...,2] -= self.coords[...,1]
        # N x m x 3
        self.labels = train_y if train else test_y
        self.coords_std = np.std(train_x,axis=(0,1))
        self.coords /= self.coords_std
        self.coords = self.coords.transpose((0,2,1)) # B x n x c -> B x c x n
        self.size=size
        #pt_coords = torch.from_numpy(self.coords)
        #self.coords = FarthestSubsample(ds_frac=size/2048)((pt_coords,pt_coords))[0].numpy()

    def __getitem__(self,index):
        return torch.from_numpy(self.coords[index]).float(), int(self.labels[index])
    def __len__(self):
        return len(self.labels)
    def default_aug_layers(self):
        subsample = Expression(lambda x: x[:,:,np.random.permutation(x.shape[-1])[:self.size]])
        return nn.Sequential(subsample,RandomZrotation(),GaussianNoise(.01))#,augLayers.PointcloudScale())#


try:
    import torch_geometric
    warnings.filterwarnings('ignore')
    @export
    class MNISTSuperpixels(torch_geometric.datasets.MNISTSuperpixels,metaclass=Named):
        ignored_index = -100
        class_weights = None
        stratify=True
        num_targets = 10
        # def __init__(self,*args,**kwargs):
        #     super().__init__(*args,**kwargs)
        # coord scale is 0-25, std of unif [0-25] is
        def __getitem__(self,index):
            datapoint = super().__getitem__(int(index))
            coords = (datapoint.pos.T-13.5)/5 # 2 x M array of coordinates
            bchannel = (datapoint.x.T-.1307)/0.3081 # 1 x M array of blackwhite info
            label = int(datapoint.y.item())
            return ((coords,bchannel),label)
        def default_aug_layers(self):
            return nn.Sequential()
except ImportError:
    warnings.warn('torch_geometric failed to import MNISTSuperpixel cannot be used.', ImportWarning)

class RandomRotateTranslate(nn.Module):
    def __init__(self,max_trans=2):
        super().__init__()
        self.max_trans = max_trans
    def forward(self,img):
        if not self.training: return img
        bs,c,h,w = img.shape
        angles = torch.rand(bs)*2*np.pi
        affineMatrices = torch.zeros(bs,2,3)
        affineMatrices[:,0,0] = angles.cos()
        affineMatrices[:,1,1] = angles.cos()
        affineMatrices[:,0,1] = angles.sin()
        affineMatrices[:,1,0] = -angles.sin()
        affineMatrices[:,0,2] = (2*torch.rand(bs)-1)*self.max_trans/w
        affineMatrices[:,1,2] = (2*torch.rand(bs)-1)*self.max_trans/h
        flowgrid = F.affine_grid(affineMatrices.to(img.device), size = img.shape)
        transformed_img = F.grid_sample(img,flowgrid)
        return transformed_img

@export
class RotMNIST(EasyIMGDataset,torchvision.datasets.MNIST):
    """ Unofficial RotMNIST dataset created on the fly by rotating MNIST"""
    means = (0.5,)
    stds = (0.25,)
    num_targets = 10
    def __init__(self,*args,dataseed=0, max_rotation=2*np.pi, **kwargs):
        super().__init__(*args,download=True,**kwargs)
        # xy = (np.mgrid[:28,:28]-13.5)/5
        # disk_cutout = xy[0]**2 +xy[1]**2 < 7
        # self.img_coords = torch.from_numpy(xy[:,disk_cutout]).float()
        # self.cutout_data = self.data[:,disk_cutout].unsqueeze(1)
        N = len(self)
        with FixedNumpySeed(dataseed):
            angles = torch.rand(N)* 2 * max_rotation - max_rotation
        with torch.no_grad():
            # R = torch.zeros(N,2,2)
            # R[:,0,0] = R[:,1,1] = angles.cos()
            # R[:,0,1] = R[:,1,0] = angles.sin()
            # R[:,1,0] *=-1
            # Build affine matrices for random translation of each image
            affineMatrices = torch.zeros(N,2,3)
            affineMatrices[:,0,0] = angles.cos()
            affineMatrices[:,1,1] = angles.cos()
            affineMatrices[:,0,1] = angles.sin()
            affineMatrices[:,1,0] = -angles.sin()
            # affineMatrices[:,0,2] = -2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/w
            # affineMatrices[:,1,2] = 2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/h
            self.data = self.data.unsqueeze(1).float()
            flowgrid = F.affine_grid(affineMatrices, size = self.data.size())
            self.data = F.grid_sample(self.data, flowgrid)
    def __getitem__(self,idx):
        return (self.data[idx]-.5)/.25, int(self.targets[idx])
    def default_aug_layers(self):
        return RandomRotateTranslate(0)# no translation

@export
class NormalRotMNIST(EasyIMGDataset,torchvision.datasets.MNIST):
    """ Unofficial RotMNIST dataset created on the fly by rotating MNIST"""
    means = (0.5,)
    stds = (0.25,)
    num_targets = 10
    def __init__(self,*args,dataseed=0, rot_sigma=1, **kwargs):
        super().__init__(*args,download=True,**kwargs)
        # xy = (np.mgrid[:28,:28]-13.5)/5
        # disk_cutout = xy[0]**2 +xy[1]**2 < 7
        # self.img_coords = torch.from_numpy(xy[:,disk_cutout]).float()
        # self.cutout_data = self.data[:,disk_cutout].unsqueeze(1)
        N = len(self)
        with FixedNumpySeed(dataseed):
            angles = torch.randn(N)* rot_sigma
        with torch.no_grad():
            # R = torch.zeros(N,2,2)
            # R[:,0,0] = R[:,1,1] = angles.cos()
            # R[:,0,1] = R[:,1,0] = angles.sin()
            # R[:,1,0] *=-1
            # Build affine matrices for random translation of each image
            affineMatrices = torch.zeros(N,2,3)
            affineMatrices[:,0,0] = angles.cos()
            affineMatrices[:,1,1] = angles.cos()
            affineMatrices[:,0,1] = angles.sin()
            affineMatrices[:,1,0] = -angles.sin()
            # affineMatrices[:,0,2] = -2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/w
            # affineMatrices[:,1,2] = 2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/h
            self.data = self.data.unsqueeze(1).float()
            flowgrid = F.affine_grid(affineMatrices, size = self.data.size())
            self.data = F.grid_sample(self.data, flowgrid)
    def __getitem__(self,idx):
        return (self.data[idx]-.5)/.25, int(self.targets[idx])
    def default_aug_layers(self):
        return RandomRotateTranslate(0)# no translation

from PIL import Image
# from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, \
#     makedir_exist_ok, verify_str_arg
from torchvision.datasets.vision import VisionDataset
# !wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
# # uncompress the zip file
# !unzip -n mnist_rotation_new.zip -d mnist_rotation_new
class MnistRotDataset(VisionDataset,metaclass=Named):
    """ Official RotMNIST dataset."""
    ignored_index = -100
    class_weights = None
    balanced = True
    stratify = True
    means = (0.130,)
    stds = (0.297,)
    num_targets=10
    num_channels=1
    resources = ["http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"]
    training_file = 'mnist_all_rotation_normalized_float_train_valid.amat'
    test_file = 'mnist_all_rotation_normalized_float_test.amat'
    def __init__(self,root, train=True, transform=None,download=True):
        if transform is None:
            normalize = transforms.Normalize(self.means, self.stds)
            transform = transforms.Compose([transforms.ToTensor(),normalize])
        super().__init__(root,transform=transform)
        self.train = train
        if download:
            self.download()
        if train:
            file=os.path.join(self.raw_folder, self.training_file)
        else:
            file=os.path.join(self.raw_folder, self.test_file)

        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')

        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    def _check_exists(self):
        return (os.path.exists(os.path.join(self.raw_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.raw_folder,
                                            self.test_file)))
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')
    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=None)
        print('Downloaded!')

    def __len__(self):
        return len(self.labels)
    def default_aug_layers(self):
        return RandomRotateTranslate(0)# no translation

@export
class STL10(EasyIMGDataset,torchvision.datasets.STL10):
    # means = (0.4467, 0.4398, 0.4066)
    # stds = (.2603,.2566,.2713)
    means = (0.4914, 0.4822, 0.4465) #using Cifar10 normalization #s
    stds = (.247,.243,.261) # see https://github.com/uoguelph-mlrg/Cutout/issues/2
    num_targets=10
    def default_aug_layers(self):
        return nn.Sequential(
        augLayers.RandomTranslate(12),
        augLayers.RandomHorizontalFlip(),
        augLayers.RandomErasing()
        )
    def __init__(self,*args,train=True,**kwargs):
        return super().__init__(*args,split='train' if train else 'test',**kwargs)


if __name__=='__main__':
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import cv2

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    i = 0
    # a = load_data(os.path.expanduser('~/datasets/ModelNet40/'))[0]
    # a[...,2] += a[...,1]
    # a[...,1] = a[...,2]-a[...,1]
    # a[...,2] -= a[...,1]
    D = ModelNet40()
    def update_plot(e):
        global i
        if e.key == "right": i+=1
        elif e.key == "left": i-=1
        else:return
        ax.cla()
        xyz,label = D[i]#.T
        x,y,z = xyz.numpy()*D.coords_std[:,None]
        # d[2] += d[1]
        # d[1] = d[2]-d[1]
        # d[2] -= d[1]
        ax.scatter(x,y,z,c=z)
        ax.text2D(0.05, 0.95, D.classes[label], transform=ax.transAxes)
        #ax.contour3D(d[0],d[2],d[1],cmap='viridis',edgecolor='none')
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        fig.canvas.draw()
    fig.canvas.mpl_connect('key_press_event',update_plot)
    plt.show()
