from model import *
from data.get_faces_loaders import *
import torch
import numpy as np
import math
import argparse
from augerino import models, losses
from augerino.models.e2_steerable import C8SteerableCNN, SmallE2

def main(args):
    model = SmallE2(channel_in=1, n_classes=1, rot_n=8)

    trainloader = get_faces_loaders(batch_size=args.batch_size, test=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    criterion = torch.nn.MSELoss()

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            print(loss.item())
            loss.backward()
            optimizer.step()

    fname = "/e2_trained.pt"
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
        default=128,
        metavar="N",
        help="input batch size (default: 128)",
    )

    parser.add_argument(
        "--num_channels", type=int, default=4, help="number of channels for network"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )
    args = parser.parse_args()

    main(args)
