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
from gng import GrowingNeuralGas
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
    args = argparser.parse_args()
    datasetOrig = pd.read_csv(args.filename, header=None)
    epochs: int = 500

    # data = np.array(datasetOrig.iloc[:, :-1])
    # labels = np.array(datasetOrig.iloc[:, -1])

    n_data = 2000
    n_outliers = 50
    dataset_type = 'moons'
    if dataset_type == 'blobs':
        data = datasets.make_blobs(n_samples=n_data, random_state=8)[0]
    elif dataset_type == 'moons':
        data = datasets.make_moons(n_samples=n_data, noise=.05)[0]
    elif dataset_type == 'circles':
        data = datasets.make_circles(n_samples=n_data, factor=.5, noise=.1)[0]
    data_outliers = datasets.make_circles(n_samples=n_data, factor=.5, noise=.1)[0]
    data[:n_outliers] = shuffle(data_outliers)[:n_outliers]
    labels = np.zeros(n_data)
    for i in range(n_outliers):
        labels[i] = 1

    plt.scatter(x=data[:, 0], y=data[:, 1], s=10, c='b', marker="s", label='data')
    plt.show()

    scaler = preprocessing.StandardScaler()
    data_scaled = scaler.fit_transform(data)
    eps = np.std(data_scaled)
    n_mature = 20
    numData = 0
    global_train_err = []
    global_test_err = []
    num_mature_neurons = []
    train_data, test_data = train_test_split(shuffle(data_scaled), test_size=0.25)
    gng = GrowingNeuralGas(amature=n_mature, alfac1=0.1, alfacN=0.01, startA=torch.tensor(test_data[0]),
                           startB=torch.tensor(test_data[1]), lambdaParam=20, alfaParam=0.5, dParam=0.995)

    print("Train, test size: {} {} - eps: {} - matureNeurons: {}".format(train_data.shape[0], test_data.shape[0], eps, n_mature))
    try:
        mean_train_error = np.inf
        mean_test_error = np.inf
        mature_neurons_ratio = 0
        for ep in range(epochs):
            start = time.time()
            for d in train_data:
                gng.forward(torch.tensor(d).view(1, -1))
                gng.CountSignal += 1
                # print("\r {} {}".format(gng.Units.shape[0], gng.CountSignal), end="")
            mature_torch_neurons = gng.Units
            # print("")
            if mature_torch_neurons is not None:
                mean_train_error = compute_global_error(mature_torch_neurons, torch.tensor(train_data), cuda=True)
                mean_test_error = compute_global_error(mature_torch_neurons, torch.tensor(test_data), cuda=True)
                actual_mature_neurons = mature_torch_neurons.shape[0]
                if len(num_mature_neurons) > 1:
                    mature_neurons_ratio = num_mature_neurons[-1] / actual_mature_neurons
                num_mature_neurons.append(actual_mature_neurons)
            print("\repoch [{}/{}] - Train euclidean error : {:.4f} - Test euclidean error : {:.4f} - #mature neurons: {} - Time :{} - Process:{}%"
                  .format(ep + 1, epochs, mean_train_error, mean_test_error, actual_mature_neurons, time_since(start), round(ep/epochs*100), 3), end="")

            if mature_neurons_ratio <= 0.95 and (actual_mature_neurons < 0.25*train_data.shape[0]):
                global_train_err.append(mean_train_error)
                global_test_err.append(mean_test_error)
            else:
                break

    except KeyboardInterrupt:
        mature_torch_neurons = gng.Units
        mature_neurons = mature_torch_neurons.data.numpy()

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
    mature_torch_neurons = gng.Units
    mature_neurons = mature_torch_neurons.data.numpy()

    min_distances = torch.transpose(min_distance_from_centroids(mature_torch_neurons, torch.tensor(data_scaled)),
                                    dim0=0, dim1=-1).numpy()
    avg_distances_from_K_centroids = torch.transpose(avg_k_distance_from_centroids(mature_torch_neurons, torch.tensor(data_scaled)),
                                                     dim0=0, dim1=-1).numpy()
    max_distances_from_K_centroids = torch.transpose(max_k_distance_from_centroids(mature_torch_neurons, torch.tensor(data_scaled)),
                                                     dim0=0, dim1=-1).numpy()
    outlier_K_factor_in_cluster = torch.transpose(outlier_factor_in_cluster(mature_torch_neurons, torch.tensor(data_scaled)),
                                                  dim0=0, dim1=-1).numpy()
    data_cluster_sparsity = cluster_sparsity(mature_torch_neurons, torch.tensor(data_scaled))

    outliers = pd.DataFrame(data=np.concatenate((min_distances.reshape(-1, 1),
                                                 avg_distances_from_K_centroids.reshape(-1, 1),
                                                 max_distances_from_K_centroids.reshape(-1, 1),
                                                 outlier_K_factor_in_cluster.reshape(-1, 1),
                                                 data_cluster_sparsity.reshape(-1, 1),
                                                 labels.reshape(-1, 1)),
                                                axis=1),
                            columns=['min_distances', 'avg_k_distances', 'max_k_distances', 'outlier_K_factor', 'cluster_sparsity','label'])

    fig, ax = plt.subplots()
    ax.scatter(x=data_scaled[n_outliers:, 0], y=data_scaled[n_outliers:, 1], s=10, c='b', marker="s", label='data')
    ax.scatter(x=data_scaled[:n_outliers, 0], y=data_scaled[:n_outliers, 1], s=12, c='G', marker="s", label='outliers')
    ax.scatter(x=mature_neurons[:, 0], y=mature_neurons[:, 1], s=10, c='r', marker="o", label='data')
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
