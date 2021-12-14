import numpy as np
import torch
import torch.nn.functional as F


def generate_mario_data(
    ntrain=10000,
    ntest=5000,
    batch_size=128,
    dpath="./data/",
    dataseed=88
):

    imgs = np.load(dpath + "images.npz")
    mario = torch.FloatTensor(imgs['mario'])
    iggy = torch.FloatTensor(imgs['iggy'])

    ntrain_each = int(ntrain/2)
    ntest_each = int(ntest/2)

    train_mario = torch.cat(ntrain_each*[mario])
    train_iggy = torch.cat(ntrain_each*[iggy])

    test_mario = torch.cat(ntest_each*[mario])
    test_iggy = torch.cat(ntest_each*[iggy])

    torch.random.manual_seed(dataseed)

    # get angles and make labels ##
    # this is a bunch of stupid algebra ##
    train_mario_pos = torch.rand(int(ntrain_each/2)) * np.pi/2. - np.pi/4.
    neg_angles = torch.rand(int(ntrain_each/2)) * np.pi/2. - np.pi/4.
    train_mario_neg = neg_angles.clone()
    train_mario_neg[neg_angles < 0] = neg_angles[neg_angles < 0] + np.pi
    train_mario_neg[neg_angles > 0] = neg_angles[neg_angles > 0] - np.pi
    train_mario_angles = torch.cat((train_mario_pos, train_mario_neg))

    train_iggy_pos = torch.rand(int(ntrain_each/2)) * np.pi/2. - np.pi/4.
    neg_angles = torch.rand(int(ntrain_each/2)) * np.pi/2. - np.pi/4.
    train_iggy_neg = neg_angles.clone()
    train_iggy_neg[neg_angles < 0] = neg_angles[neg_angles < 0] + np.pi
    train_iggy_neg[neg_angles > 0] = neg_angles[neg_angles > 0] - np.pi
    train_iggy_angles = torch.cat((train_iggy_pos, train_iggy_neg))

    test_mario_pos = torch.rand(int(ntest_each/2)) * np.pi/2. - np.pi/4.
    neg_angles = torch.rand(int(ntest_each/2)) * np.pi/2. - np.pi/4.
    test_mario_neg = neg_angles.clone()
    test_mario_neg[neg_angles < 0] = neg_angles[neg_angles < 0] + np.pi
    test_mario_neg[neg_angles > 0] = neg_angles[neg_angles > 0] - np.pi
    test_mario_angles = torch.cat((test_mario_pos, test_mario_neg))

    test_iggy_pos = torch.rand(int(ntest_each/2)) * np.pi/2. - np.pi/4.
    neg_angles = torch.rand(int(ntest_each/2)) * np.pi/2. - np.pi/4.
    test_iggy_neg = neg_angles.clone()
    test_iggy_neg[neg_angles < 0] = neg_angles[neg_angles < 0] + np.pi
    test_iggy_neg[neg_angles > 0] = neg_angles[neg_angles > 0] - np.pi
    test_iggy_angles = torch.cat((test_iggy_pos, test_iggy_neg))

    train_mario_labs = torch.zeros_like(train_mario_angles)
    train_mario_labs[train_mario_angles.abs() > 1.] = 1.
    train_iggy_labs = torch.zeros_like(train_iggy_angles)
    train_iggy_labs[train_iggy_angles.abs() < 1.] = 2.
    train_iggy_labs[train_iggy_angles.abs() > 1.] = 3.

    test_mario_labs = torch.zeros_like(test_mario_angles)
    test_mario_labs[test_mario_angles.abs() > 1.] = 1.
    test_iggy_labs = torch.zeros_like(test_iggy_angles)
    test_iggy_labs[test_iggy_angles.abs() < 1.] = 2.
    test_iggy_labs[test_iggy_angles.abs() > 1.] = 3.

    # combine to just train and test ##
    train_images = torch.cat((train_mario, train_iggy))
    test_images = torch.cat((test_mario, test_iggy))

    train_angles = torch.cat((train_mario_angles, train_iggy_angles))
    test_angles = torch.cat((test_mario_angles, test_iggy_angles))

    train_labs = torch.cat(
        (train_mario_labs, train_iggy_labs)).type(torch.LongTensor)
    test_labs = torch.cat(
        (test_mario_labs, test_iggy_labs)).type(torch.LongTensor)

    # rotate ##
    # train #
    with torch.no_grad():
        # Build affine matrices for random translation of each image
        affineMatrices = torch.zeros(ntrain, 2, 3)
        affineMatrices[:, 0, 0] = train_angles.cos()
        affineMatrices[:, 1, 1] = train_angles.cos()
        affineMatrices[:, 0, 1] = train_angles.sin()
        affineMatrices[:, 1, 0] = -train_angles.sin()

        flowgrid = F.affine_grid(affineMatrices, size=train_images.size())
        train_images = F.grid_sample(train_images, flowgrid)

    # test #
    with torch.no_grad():
        # Build affine matrices for random translation of each image
        affineMatrices = torch.zeros(ntest, 2, 3)
        affineMatrices[:, 0, 0] = test_angles.cos()
        affineMatrices[:, 1, 1] = test_angles.cos()
        affineMatrices[:, 0, 1] = test_angles.sin()
        affineMatrices[:, 1, 0] = -test_angles.sin()

        flowgrid = F.affine_grid(affineMatrices, size=test_images.size())
        test_images = F.grid_sample(test_images, flowgrid)

    # shuffle ##
    trainshuffler = np.random.permutation(ntrain)
    testshuffler = np.random.permutation(ntest)

    train_images = train_images[np.ix_(trainshuffler), ::].squeeze()
    train_labs = train_labs[np.ix_(trainshuffler)]

    test_images = test_images[np.ix_(testshuffler), ::].squeeze()
    test_labs = test_labs[np.ix_(testshuffler)]

    if batch_size == ntrain:
        return train_images, train_labs, test_images, test_labs
    else:
        traindata = torch.utils.data.TensorDataset(train_images, train_labs)
        trainloader = torch.utils.data.DataLoader(
            traindata, batch_size=batch_size)
        testdata = torch.utils.data.TensorDataset(test_images, test_labs)
        testloader = torch.utils.data.DataLoader(
            testdata, batch_size=batch_size)

        return trainloader, testloader
