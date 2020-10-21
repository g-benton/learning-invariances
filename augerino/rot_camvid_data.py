import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from pathlib import Path
import torch.nn.functional as F

from .camvid_data import IMG_EXTENSIONS, classes, class_weight
from .camvid_data import mean, std, class_color
from .camvid_data import has_file_allowed_extension, is_image_file, _make_dataset


class RotCamVid(data.Dataset):

    def __init__(self, root, rotation_tensor, 
                 split='train', joint_transform=None,
                 transform=None, target_transform=None,
                 download=False,
                 loader=default_loader):
        self.root = root
        #self.root = Path(root)
        
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.class_weight = class_weight
        self.classes = classes
        self.mean = mean
        self.std = std
        self.rotation_tensor = rotation_tensor

        if download:
            self.download()
        self.imgs = _make_dataset(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        target = Image.open(path.replace(self.split, self.split + 'annot'))

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        # Applying the rotation
        rotation = self.rotation_tensor[index][None, :]
        
        img = img[None, :]
        flowgrid = F.affine_grid(rotation, size=img.size(), align_corners=True)
        img = F.grid_sample(img, flowgrid, align_corners=True)
        
        target = target[None, None, :].float() + 1.
        flowgrid = F.affine_grid(rotation, size=target.size(), align_corners=True)
        target = F.grid_sample(target, flowgrid, align_corners=True)
        target = target[:, 0].long() - 1
        target[target == -1] = 12 #padding with 'void' class

        return img[0], target[0]

    def __len__(self):
        return len(self.imgs)

    def download(self):
        # TODO: please download the dataset from
        # https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
        raise NotImplementedError


def rot_camvid_loaders(path, batch_size, num_workers, transform_train, transform_test,
                   train_rotations, test_rotations, val_rotations,
                   use_validation=False, val_size=0, shuffle_train=True, 
                   joint_transform=None, ft_joint_transform=None, ft_batch_size=1, **kwargs):

    #load training and finetuning datasets
    print(path)
    train_set = RotCamVid(root=path, split='train', rotation_tensor=train_rotations, 
                          joint_transform=joint_transform, transform=transform_train, **kwargs)
    ft_train_set = RotCamVid(root=path, split='train', rotation_tensor=train_rotations, 
                          joint_transform=ft_joint_transform, transform=transform_train, **kwargs)

    val_set = RotCamVid(root=path, split='val', rotation_tensor=val_rotations,
                        joint_transform=None, transform=transform_test, **kwargs)
    test_set = RotCamVid(root=path, split='test', rotation_tensor=test_rotations,
                         joint_transform=None, transform=transform_test, **kwargs)

    num_classes = 11 # hard coded labels here
    
    return {'train': torch.utils.data.DataLoader(
                        train_set, 
                        batch_size=batch_size, 
                        shuffle=shuffle_train, 
                        num_workers=num_workers,
                        pin_memory=True
                ),
            'fine_tune': torch.utils.data.DataLoader(
                        ft_train_set, 
                        batch_size=ft_batch_size, 
                        shuffle=shuffle_train, 
                        num_workers=num_workers,
                        pin_memory=True
                ),
            'val': torch.utils.data.DataLoader(
                        val_set, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers,
                        pin_memory=True
                ),
            'test': torch.utils.data.DataLoader(
                        test_set, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers,
                        pin_memory=True
                )}, num_classes
