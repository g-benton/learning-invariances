import torch
import numpy as np
import math
import argparse
from augerino import datasets, models, losses
from augerino.models.e2_steerable import C8SteerableCNN, SmallE2
from torch.utils.data import DataLoader
from oil.utils.mytqdm import tqdm

def main(args):
    model = SmallE2(channel_in=1, n_classes=10, rot_n=10)

    dataset = datasets.RotMNIST("~/datasets/", train=True)
    trainloader = DataLoader(dataset, batch_size=args.batch_size)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
    #                                 weight_decay=args.wd)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,
                                 weight_decay=args.wd)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        print("Using Cuda")

    ## save init model ##
    fname = "/e2_init.pt"
    torch.save(model.state_dict(), args.dir + fname)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(args.epochs)):  # loop over the dataset multiple times

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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            batches += 1
            print(loss.item())
        print("Epoch = ", epoch)
        print("\n")

        
    fname = "/e2_epoch" + str(epoch+1) + ".pt"
    torch.save(model.state_dict(), args.dir + fname)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="olivetti augerino")

    parser.add_argument(
        "--dir",
        type=str,
        default='./saved-outputs',
        help="training directory (default: None)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        metavar="N",
        help="input batch size (default: 50)",
    )

    parser.add_argument(
        "--num_channels", type=int, default=4, help="number of channels for network"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
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
        "--save_freq",
        type=int,
        default=25,
        metavar="N",
        help="save frequency (default: 25)",
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
