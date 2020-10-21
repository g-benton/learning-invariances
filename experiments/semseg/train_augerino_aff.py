# Code adapted from github.com/wjmaddox/swa_gaussian
# which is in turn adapted from github.com/bfortuner/pytorch_tiramisu

import time
from pathlib import Path
import numpy as np
import os, sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from functools import partial

import utils.training as train_utils

from utils.training import adjust_learning_rate, schedule, save_checkpoint
from utils.training import seg_cross_entropy
import augerino.models as models
from augerino.camvid_data import camvid_loaders
from augerino.rot_camvid_data import rot_camvid_loaders
from augerino.aug_eq_model import DiffEqAug, AugEqModel, UniformEqAug

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="SGD/SWA training")

parser.add_argument("--dataset", type=str, default="CamVid")
parser.add_argument(
    "--data_path",
    type=str,
    default="/home/wesley/Documents/Code/SegNet-Tutorial/CamVid/",
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--dir",
    type=str,
    default=None,
    required=True,
    help="training directory (default: None)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=850,
    metavar="N",
    help="number of epochs to train (default: 850)",
)
parser.add_argument(
    "--save_freq",
    type=int,
    default=10,
    metavar="N",
    help="save frequency (default: 10)",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=5,
    metavar="N",
    help="evaluation frequency (default: 5)",
)

parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    metavar="MODEL",
    help="model name (default: None)",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=2,
    metavar="N",
    help="input batch size (default: 2)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=1e-4,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--lr_decay",
    type=float,
    default=0.995,
    help="amount of learning rate decay per epoch (default: 0.995)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument(
    "--ft_start",
    type=int,
    default=750,
    help="begin fine-tuning with full sized images (default: 750)",
)
parser.add_argument(
    "--ft_batch_size", type=int, default=1, help="fine-tuning batch size (default: 1)"
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to resume training from (default: None)",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--rotations_npz",
    type=str,
    default=None,
    metavar="CKPT",
    help="path to rotations npz array (default: None)",
)
parser.add_argument(
    "--padding",
    type=int,
    default=50,
    metavar="N",
    help="padding to use in augerino (default: 4)",
)
parser.add_argument(
    "--aug_reg",
    type=float,
    default=0.01,
    metavar="N",
    help="Augerino regularizer (default: 0.01)",
)

args = parser.parse_args()
if torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = False
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("Preparing directory %s" % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, "command.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")

writer = SummaryWriter(log_dir=args.dir)

print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

if args.rotations_npz is not None:
    print("Using RotCamVid data")
    arr = np.load(args.rotations_npz)
    train_rotations = torch.from_numpy(arr["train"]).float()
    test_rotations = torch.from_numpy(arr["test"]).float()
    val_rotations = torch.from_numpy(arr["val"]).float()
    loaders, num_classes = rot_camvid_loaders(
        args.data_path,
        args.batch_size,
        args.num_workers,
        ft_batch_size=args.ft_batch_size,
        transform_train=model_cfg.transform_train,
        transform_test=model_cfg.transform_test,
        joint_transform=model_cfg.joint_transform,
        ft_joint_transform=model_cfg.ft_joint_transform,
        target_transform=model_cfg.target_transform,
        train_rotations=train_rotations,
        test_rotations=test_rotations,
        val_rotations=val_rotations
    )
else:
    print("Using RotCamVid data")
    loaders, num_classes = camvid_loaders(
        args.data_path,
        args.batch_size,
        args.num_workers,
        ft_batch_size=args.ft_batch_size,
        transform_train=model_cfg.transform_train,
        transform_test=model_cfg.transform_test,
        joint_transform=model_cfg.joint_transform,
        ft_joint_transform=model_cfg.ft_joint_transform,
        target_transform=model_cfg.target_transform
    )

print("Beginning with cropped images")
train_loader = loaders["train"]

print("Preparing model")
model = model_cfg.base(
    *model_cfg.args,
    num_classes=num_classes,
    **model_cfg.kwargs,
    use_aleatoric=False
)

print("Using Augerino Model")
aug = UniformEqAug(padding=args.padding)
model = AugEqModel(model, aug, traincopies=1, testcopies=2)

model.cuda()
model.apply(train_utils.weights_init)

optimizer = torch.optim.SGD(
    model.parameters(), lr=args.lr_init, weight_decay=args.wd, momentum=0.9
)

start_epoch = 1

criterion = lambda m, i, t: seg_cross_entropy(m, i, t, reg=args.aug_reg)

if args.resume is not None:
    print("Resume training from %s" % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    del checkpoint

for epoch in range(start_epoch, args.epochs + 1):
    since = time.time()

    ### Train ###
    if epoch == args.ft_start:
        print("Now replacing data loader with fine-tuned data loader.")
        train_loader = loaders["fine_tune"]

    trn_loss, trn_err = train_utils.train(model, train_loader, optimizer, criterion, epoch=epoch, writer=writer)
    writer.add_scalar("train/loss", trn_loss, epoch)
    writer.add_scalar("train/error", trn_err, epoch)
    writer.add_scalar("params/lower0", model.aug.lower[0], epoch)
    writer.add_scalar("params/lower1", model.aug.lower[1], epoch)
    writer.add_scalar("params/lower2", model.aug.lower[2], epoch)
    writer.add_scalar("params/lower3", model.aug.lower[3], epoch)
    writer.add_scalar("params/lower4", model.aug.lower[4], epoch)
    writer.add_scalar("params/lower5", model.aug.lower[5], epoch)
    writer.add_scalar("params/upper0", model.aug.upper[0], epoch)
    writer.add_scalar("params/upper1", model.aug.upper[1], epoch)
    writer.add_scalar("params/upper2", model.aug.upper[2], epoch)
    writer.add_scalar("params/upper3", model.aug.upper[3], epoch)
    writer.add_scalar("params/upper4", model.aug.upper[4], epoch)
    writer.add_scalar("params/upper5", model.aug.upper[5], epoch)

    print(
        "Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}".format(
            epoch, trn_loss, 1 - trn_err
        )
    )
    time_elapsed = time.time() - since
    print("Train Time {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    if epoch % args.eval_freq == 0:
        pass
        ### Test ###
        #val_loss, val_err, val_iou = train_utils.test(model, loaders["val"], criterion, epoch=epoch, writer=writer)
        #print(
        #    "Val - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}".format(
        #        val_loss, 1 - val_err, val_iou
        #    )
        #)
        #writer.add_scalar("val/loss", val_loss, epoch)
        #writer.add_scalar("val/error", val_err, epoch)

    time_elapsed = time.time() - since
    print("Total Time {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))

    ### Checkpoint ###
    if epoch % args.save_freq == 0:
        print("Saving model at Epoch: ", epoch)
        save_checkpoint(
            dir=args.dir,
            epoch=epoch,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )

    lr = schedule(
        epoch, args.lr_init, args.epochs
    )
    adjust_learning_rate(optimizer, lr)
    writer.add_scalar("hypers/lr", lr, epoch)

### Test set ###

test_loss, test_err, test_iou = train_utils.test(model, loaders["test"], criterion)
print(
    "SGD Test - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}".format(
        test_loss, 1 - test_err, test_iou
    )
)
