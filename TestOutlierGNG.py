__authors__ = 'Antonio Ritacco'
__email__ = 'ritacco.ant@gmail.com'


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import datasets


import argparse
from gng import GrowingNeuralGas
from igng import IncrementalGrowingNeuralGas
from AutoEncoder import AutoEncoder
import torch.nn.functional as F


# import torch.distributions.kl as kl
import torch
import torch.nn as nn

from torch.autograd import Variable


from sklearn import preprocessing

BETA = 0.01
NUM_EPOCHS = 100
batchSIZE = 32



if __name__ == '__main__':
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('-f', '--filename', dest='filename', type=str)
    # args = argparser.parse_args()
    # datasetOrig = pd.read_csv(args.filename, header=None)
    dim = 2
    epochs = 5
    n_samples = 1000
    dataset_type = 'circles'

    if dataset_type == 'blobs':
        data = datasets.make_blobs(n_samples=n_samples, random_state=8)
    elif dataset_type == 'moons':
        data = datasets.make_moons(n_samples=n_samples, noise=.05)
    elif dataset_type == 'circles':
        data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    # datasetOrig = np.random.normal(loc=0, scale=0.1, size=[1000, dim])
    # datasetOrig = np.concatenate((datasetOrig, np.random.normal(loc=0.25, scale=0.1, size=[1000, dim])))
    # datasetOrig = np.concatenate((datasetOrig, np.random.normal(loc=0.75, scale=0.1, size=[1000, dim])))
    # datasetOrig = np.concatenate((datasetOrig, np.random.normal(loc=1, scale=0.1, size=[1000, dim])))
    scaler = preprocessing.StandardScaler()
    data_scaled = scaler.fit_transform(data[0])
    eps = np.std(data_scaled)/3
    igng = IncrementalGrowingNeuralGas(epsilon= eps, amature = 5, alfac1 = 0.1, alfacN= 0.01)
    for ep in range(epochs):
        data_scaled = shuffle(data_scaled)
        for d in data_scaled:
            igng.forward(torch.tensor(d).view(1,-1))
        print('\repoch [{}/{}]'
              .format(ep + 1, epochs), end="")
    # gng.fit_network(e_b=0.05, e_n=0.006, a_max=8, l=100, a=0.5, d=0.995, passes=10, plot_evolution=False)
    # gng.plot_clusters(clustered_data)
    mature_neurons, mature_indexes = igng.getMatureNeurons()
    print('Finish')
    plt.scatter(x=data_scaled[:, 0], y=data_scaled[:, 1], s=10, c='b', marker="s", label='data')
    plt.scatter(x=mature_neurons[:, 0], y=mature_neurons[:, 1], s=10, c='r', marker="o", label='data')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xs =data[:, 0], ys=data[:, 1], zs=data[:, 2], s=10, c='b', marker="s", label='data')
    # ax.scatter(xs =mature_neurons[:, 0], ys=mature_neurons[:, 1], zs=mature_neurons[:, 2], s=20, c='r', marker="o", label='neurons')
    plt.show()

    print('******')
