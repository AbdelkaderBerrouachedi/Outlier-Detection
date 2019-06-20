__authors__ = 'Antonio Ritacco'
__email__ = 'ritacco.ant@gmail.com'


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helpers import *


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import datasets


import argparse
from gng_old import GrowingNeuralGas
from igng import IncrementalGrowingNeuralGas
from AutoEncoder import AutoEncoder
import torch.nn.functional as F


# import torch.distributions.kl as kl
import torch
import torch.nn as nn

from torch.autograd import Variable


from sklearn import preprocessing



if __name__ == '__main__':
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('-f', '--filename', dest='filename', type=str)
    # args = argparser.parse_args()
    # datasetOrig = pd.read_csv(args.filename, header=None)
    dim = 2
    epochs: int = 500
    n_mature = 20
    n_data = 5000
    dataset_type = 'circles'

    if dataset_type == 'blobs':
        data = datasets.make_blobs(n_samples=n_data, random_state=8)
    elif dataset_type == 'moons':
        data = datasets.make_moons(n_samples=n_data, noise=.1)
    elif dataset_type == 'circles':
        data = datasets.make_circles(n_samples=n_data, factor=.5, noise=.1)
    # datasetOrig = np.random.normal(loc=0, scale=0.1, size=[1000, dim])
    # datasetOrig = np.concatenate((datasetOrig, np.random.normal(loc=0.25, scale=0.1, size=[1000, dim])))
    # datasetOrig = np.concatenate((datasetOrig, np.random.normal(loc=0.75, scale=0.1, size=[1000, dim])))
    # datasetOrig = np.concatenate((datasetOrig, np.random.normal(loc=1, scale=0.1, size=[1000, dim])))
    scaler = preprocessing.StandardScaler()
    data_scaled = scaler.fit_transform(data[0])
    eps = np.std(data_scaled)
    igng = IncrementalGrowingNeuralGas(epsilon=eps, amature=n_mature, alfac1=0.5, alfacN=0.1)
    numData = 0
    global_train_err = []
    global_test_err = []
    num_mature_neurons = []
    data_scaled = shuffle(data_scaled)
    train_data, test_data = train_test_split(data_scaled, test_size=0.5)
    print("Train, test size: {} {} - eps: {} - matureNeurons: {}".format(train_data.shape[0], test_data.shape[0], eps, n_mature))
    start = time.time()
    try:
        mature_neurons_ratio = 0.0
        for ep in range(epochs):
            for d in train_data:
                igng.forward(torch.tensor(d).view(1, -1))
            mature_torch_neurons, mature_indexes = igng.getMatureNeurons()
            mean_train_error = igng.compute_global_error(mature_torch_neurons, torch.tensor(train_data))
            mean_test_error = igng.compute_global_error(mature_torch_neurons, torch.tensor(test_data))
            actual_mature_neurons = mature_torch_neurons.shape[0]

            print("\repoch [{}/{}] - Train euclidean error : {:.4f} - Test euclidean error : {:.4f} - #mature neurons: {} - Epsilon : {:.4f} - Time :{} - Process:{}%"
                  .format(ep + 1, epochs, mean_train_error, mean_test_error, actual_mature_neurons, igng.epsilon, time_since(start), round(ep/epochs*100), 3), end="")

            if ep > 1:
                mature_neurons_ratio = num_mature_neurons[-1] / actual_mature_neurons
                if igng.epsilon > eps/3:
                    igng.epsilon = igng.epsilon * mature_neurons_ratio
            if mature_neurons_ratio < 0.99:
                global_train_err.append(mean_train_error)
                global_test_err.append(mean_test_error)
                num_mature_neurons.append(actual_mature_neurons)
            else:
                break

    except KeyboardInterrupt:
        mature_torch_neurons, mature_indexes = igng.getMatureNeurons()
        mature_neurons = mature_torch_neurons.data.numpy()
        fig, ax = plt.subplots()
        ax.scatter(x=data_scaled[:, 0], y=data_scaled[:, 1], s=10, c='b', marker="s", label='data')
        ax.scatter(x=mature_neurons[:, 0], y=mature_neurons[:, 1], s=10, c='r', marker="o", label='data')
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(xs =data[:, 0], ys=data[:, 1], zs=data[:, 2], s=10, c='b', marker="s", label='data')
        # ax.scatter(xs =mature_neurons[:, 0], ys=mature_neurons[:, 1], zs=mature_neurons[:, 2], s=20, c='r', marker="o", label='neurons')
        plt.show()

        fig1, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('MEE', color=color)
        ax1.plot(global_train_err, label='Train MEE', color='blue')
        ax1.plot(global_test_err, label='Test MEE', color='red')
        ax1.legend()
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Num Mature Neurons', color=color)
        ax2.plot(num_mature_neurons, label='Mature neurons', linestyle='dashed', color='green')
        ax2.legend()
        fig1.tight_layout()
        plt.show()
    # gng.fit_network(e_b=0.05, e_n=0.006, a_max=8, l=100, a=0.5, d=0.995, passes=10, plot_evolution=False)
    # gng.plot_clusters(clustered_data)
    mature_torch_neurons, mature_indexes = igng.getMatureNeurons()
    mature_neurons = mature_torch_neurons.data.numpy()
    fig, ax = plt.subplots()
    ax.scatter(x=data_scaled[:, 0], y=data_scaled[:, 1], s=10, c='b', marker="s", label='data')
    ax.scatter(x=mature_neurons[:, 0], y=mature_neurons[:, 1], s=10, c='r', marker="o", label='data')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xs =data[:, 0], ys=data[:, 1], zs=data[:, 2], s=10, c='b', marker="s", label='data')
    # ax.scatter(xs =mature_neurons[:, 0], ys=mature_neurons[:, 1], zs=mature_neurons[:, 2], s=20, c='r', marker="o", label='neurons')
    plt.show()

    fig1, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MEE', color=color)
    ax1.plot(global_train_err, label='Train MEE', color='blue')
    ax1.plot(global_test_err, label='Test MEE', color='red')
    ax1.legend()
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Num Mature Neurons', color=color)
    ax2.plot(num_mature_neurons, label='Mature neurons', linestyle='dashed', color='green')
    ax2.legend()
    fig1.tight_layout()
    plt.show()

    print('******')
