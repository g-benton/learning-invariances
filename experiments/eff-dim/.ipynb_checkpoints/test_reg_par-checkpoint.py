import torch
import numpy as np
import math
import argparse
from augerino import datasets, models, losses
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

def main(args):
    net = models.layer13s(in_channels=3,num_targets=10)
    augerino = models.UniformAug()
    model = models.AugAveragedModel(net, augerino,ncopies=args.ncopies)

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    fname = "aug_reg" + str(args.aug_reg) + ".pt"
    
    dataset = torchvision.datasets.CIFAR10("/datasets/cifar10", train=True, download=False,
                                           transform=transform)
    trainloader = DataLoader(dataset, batch_size=args.batch_size)

    optimizer = torch.optim.Adam([{'name': 'model', 
                                   'params': model.model.parameters(), 
                                   "weight_decay": args.wd}, 
                                  {'name': 'aug', 
                                   'params': model.aug.parameters(), 
                                   "weight_decay": 0.}], 
                                 lr=args.lr)


    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        print("Using Cuda")

    criterion = losses.safe_unif_aug_loss

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        epoch_loss = 0
        batches = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels, model, reg=args.aug_reg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        print("Epoch = ", epoch)
        print("Epoch loss = ", epoch_loss/batches)
        print("\n")


    torch.save(model.state_dict(), args.dir + fname)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="cifar10 augerino")

    parser.add_argument(
        "--dir",
        type=str,
        default='./saved-outputs/',
        help="training directory (default: None)",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default='none',
        help="default transforms, options = 'none' or 'fixed'",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 50)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--aug_reg",
        type=float,
        default=0.01,
        help="augmentation regularization weight",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
        metavar="weight_decay",
        help="weight decay",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )

    parser.add_argument(
        "--ncopies",
        type=int,
        default=4,
        metavar="N",
        help="number of augmentations in network (defualt: 4)"
    )
    args = parser.parse_args()

    main(args)
