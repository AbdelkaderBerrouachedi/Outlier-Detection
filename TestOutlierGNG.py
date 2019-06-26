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
import os
import networkx as nx


import argparse
from gng_old import GrowingNeuralGas
from AutoEncoder import AutoEncoder
import torch.nn.functional as F


# import torch.distributions.kl as kl
import torch
import torch.nn as nn

from torch.autograd import Variable


from sklearn import preprocessing



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--filename', dest='filename', type=str)
    argparser.add_argument('--epochs', type=int, default=20)
    argparser.add_argument('--save', action='store_true')
    args = argparser.parse_args()
    datasetOrig = pd.read_csv(args.filename, header=None)
    epochs: int = 10
    data = np.array(datasetOrig.iloc[:, :-1])
    labels = np.array(datasetOrig.iloc[:, -1])
    n_mature = 20

    # dataset_type = 'circles'
    # n_data = 5000
    # n_outliers = 50
    # dataset_type = 'circles'
    # if dataset_type == 'blobs':
    #     data = datasets.make_blobs(n_samples=n_data, random_state=8)[0]
    # elif dataset_type == 'moons':
    #     data = datasets.make_moons(n_samples=n_data, noise=.05)[0]
    # elif dataset_type == 'circles':
    #     data = datasets.make_circles(n_samples=n_data, factor=.5, noise=.1)[0]
    #
    # # data_outliers = datasets.make_circles(n_samples=n_data, factor=.5, noise=.1)[0]
    # data_outliers = datasets.make_moons(n_samples=n_data, noise=.01)[0]
    # data[:n_outliers] = shuffle(data_outliers)[:n_outliers]
    # labels = np.zeros(n_data)
    # for i in range(n_outliers):
    #     labels[i] = 1


    scaler = preprocessing.StandardScaler()
    data_scaled = scaler.fit_transform(data)
    eps = np.std(data_scaled)
    # e_b = 0.1, e_n = 0.006, a_max = 10, l = 200, a = 0.5, d = 0.995, passes = 10
    numData = 0
    global_train_err = []
    global_test_err = []
    num_mature_neurons = []

    train_data, test_data = train_test_split(shuffle(data_scaled), test_size=0.25)
    gng = GrowingNeuralGas(input_data=train_data)

    print("Train, test size: {} {} - eps: {} - matureNeurons: {}".format(train_data.shape[0], test_data.shape[0], eps, n_mature))
    start = time.time()
    gng.fit_network(e_b=0.1, e_n=0.006, a_max=10, l=200, a=0.5, d=0.995, passes=args.epochs, test_data=test_data,
                    global_train_err=global_train_err, global_test_err=global_test_err, num_mature_neurons=num_mature_neurons)

    neurons = []
    for k in gng.network._node:
        neurons.append(gng.network._node[k]['vector'])
    mature_torch_neurons = torch.tensor(neurons)
    mature_neurons = mature_torch_neurons.data.numpy()
    min_distances = torch.transpose(min_distance_from_centroids(mature_torch_neurons, torch.tensor(data_scaled)),
                                    dim0=0, dim1=-1).numpy()

    avg_distances_from_K_centroids = torch.transpose(
        avg_k_distance_from_centroids(mature_torch_neurons, torch.tensor(data_scaled)),
        dim0=0, dim1=-1).numpy()

    max_distances_from_K_centroids = torch.transpose(
        max_k_distance_from_centroids(mature_torch_neurons, torch.tensor(data_scaled)),
        dim0=0, dim1=-1).numpy()

    outlier_K_factor_in_cluster = torch.transpose(
        outlier_factor_in_cluster(mature_torch_neurons, torch.tensor(data_scaled)),
        dim0=0, dim1=-1).numpy()

    data_cluster_sparsity = cluster_sparsity(mature_torch_neurons, torch.tensor(data_scaled))

    lof_cluster = local_outlier_factor_cluster(mature_torch_neurons, torch.tensor(data_scaled), k=20, k_lof=5)

    outliers = pd.DataFrame(data=np.concatenate((min_distances.reshape(-1, 1),
                                                 avg_distances_from_K_centroids.reshape(-1, 1),
                                                 max_distances_from_K_centroids.reshape(-1, 1),
                                                 outlier_K_factor_in_cluster.reshape(-1, 1),
                                                 data_cluster_sparsity.reshape(-1, 1),
                                                 lof_cluster.reshape(-1, 1),
                                                 labels.reshape(-1, 1)),
                                                axis=1),
                            columns=['min_distances', 'avg_k_distances', 'max_k_distances', 'outlier_K_factor',
                                     'cluster_sparsity', 'lof_clusters', 'label'])



    # fig, ax = plt.subplots()
    # ax.scatter(x=data_scaled[n_outliers:, 0], y=data_scaled[n_outliers:, 1], s=10, c='b', marker="s", label='data')
    # ax.scatter(x=data_scaled[:n_outliers, 0], y=data_scaled[:n_outliers, 1], s=12, c='G', marker="s", label='outliers')
    # ax.scatter(x=mature_neurons[:, 0], y=mature_neurons[:, 1], s=10, c='r', marker="o", label='data')

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

    # node_pos = {}
    # for u in gng.network.nodes():
    #     vector = gng.network.node[u]['vector']
    #     node_pos[u] = (vector[0], vector[1])
    # nx.draw(gng.network, pos=node_pos)
    # plt.show()
    if args.save:
        wksp_folder = os.path.dirname(args.filename)
        outliers_file = os.path.join(wksp_folder, 'GNG_outliers.csv')
        outliers.to_csv(outliers_file)
    print('*** FINISH ***')
