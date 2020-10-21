from model import *
from data.get_faces_loaders import *
import torch
import numpy as np
import math
import argparse
from augerino import models, losses
import pandas as pd
def main(args):
    net = models.SimpleConv(c=32, num_classes=1, in_channel=1)
    augerino = models.UniformAug()
    model = AugAveragedModel(net, augerino)
    start_widths = torch.ones(6) * -5.
    start_widths[2] = 0.
    model.aug.set_width(start_widths)
    
    trainloader = get_faces_loaders(batch_size=args.batch_size, test=False)
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

    criterion = losses.safe_unif_aug_loss
    logger = []
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
            loss = criterion(outputs.squeeze(), labels, model,
                            base_loss_fn=torch.nn.MSELoss(),
                            reg=args.aug_reg)
            print(loss.item(), model.aug.width[2].item())
            loss.backward()
            optimizer.step()
            log = model.aug.width.tolist()
            log += model.aug.width.grad.data.tolist()
            log += [loss.item()]
            logger.append(log)        
            

    fname = "/model_" + str(args.aug_reg) + ".pt"
    torch.save(model.state_dict(), args.dir + fname)
    df = pd.DataFrame(logger)
    df.to_pickle(args.dir + "/auglog_" + str(args.aug_reg) + ".pkl")



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
        "--aug_reg",
        type=float,
        default=0.01,
        help="augmentation regularization weight",
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
        default=50,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )
    args = parser.parse_args()

    main(args)
