__authors__ = 'Antonio Ritacco'
__email__ = 'ritacco.ant@gmail.com'


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helpers import *
import os

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import datasets
import networkx as nx


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
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--filename', dest='filename', type=str)
    argparser.add_argument('--epochs', type=int, default=5)
    argparser.add_argument('--save', action='store_true')
    argparser.add_argument('--create', action='store_true')
    argparser.add_argument('--synth', action='store_true')
    argparser.add_argument('--test', action='store_true')
    args = argparser.parse_args()
    if not args.create:
        datasetOrig = pd.read_csv(args.filename, header=None)


    if args.create:
        n_data = 5000
        n_outliers = 250
        dataset_type = 'moons'
        outlier_type = 'blobs'

        if dataset_type == 'blobs':
            data = datasets.make_blobs(n_samples=n_data, random_state=8)[0]
        elif dataset_type == 'moons':
            data = datasets.make_moons(n_samples=n_data, noise=.02)[0]
        elif dataset_type == 'circles':
            data = datasets.make_circles(n_samples=n_data, factor=.5, noise=.02)[0]

        if outlier_type == 'blobs':
            data_outliers = datasets.make_blobs(n_samples=n_data, random_state=8, cluster_std=2)[0]
        elif outlier_type == 'moons':
            data_outliers = datasets.make_moons(n_samples=n_data, noise=.05)[0]
        elif outlier_type == 'circles':
            data_outliers = datasets.make_circles(n_samples=n_data, factor=.8, noise=.05)[0]

        #data_outliers = datasets.make_circles(n_samples=n_data, factor=.5, noise=.1)[0]
        # data_outliers = datasets.moons(n_samples=n_data, noise=.02)[0]
        data[:n_outliers] = shuffle(data_outliers)[:n_outliers]
        data[:n_outliers] = data[:n_outliers]
        labels = np.zeros(n_data)
        for i in range(n_outliers):
            labels[i] = 1

        plt.scatter(x=data[:, 0], y=data[:, 1], s=10, c='b', marker="s", label='data')
        plt.show()
    else:
        if args.test:
            data = np.array(datasetOrig)
        else:
            data = np.array(datasetOrig.iloc[:, :-1])
            labels = np.array(datasetOrig.iloc[:, -1])
        if args.synth:
            plt.scatter(x=data[:, 0], y=data[:, 1], s=10, c='b', marker="s", label='data')
            plt.show()

    scaler = preprocessing.StandardScaler()
    data_scaled = scaler.fit_transform(data)
    eps = np.std(data_scaled)
    n_mature = round(0.1*data_scaled.shape[0])
    igng = IncrementalGrowingNeuralGas(epsilon=eps, amature=n_mature, alfac1=0.5, alfacN=0.1, cuda=True)
    numData = 0
    global_train_err = []
    global_test_err = []
    num_mature_neurons = []
    train_data, test_data = train_test_split(shuffle(data_scaled), test_size=0.25)
    # train_data = shuffle(data_scaled)
    print("Train size: {} - n_vars: {} - matureNeurons: {}".format(train_data.shape[0], train_data.shape[1], n_mature))
    mature_neurons_ratio = 0
    train_err_ratio = 0
    test_err_ratio = 0
    actual_mature_neurons = 0
    mean_train_error = np.inf
    mean_test_error = np.inf
    start = time.time()
    ep = 0
    while ep < args.epochs:
        for d in train_data:
            igng._forward(torch.tensor(d).view(1, -1))
        mature_torch_neurons = igng.get_mature_neurons()
        num_embryo_neurons = len(igng.network)
        # print(mature_torch_neurons.shape[0], len(igng.network))
        # assert mature_torch_neurons.shape[0] == len(igng.network), " Not equal size network and neurons"
        if mature_torch_neurons is not None:
            pairwise_err_train = pairwise_distances(mature_torch_neurons, torch.tensor(train_data), cuda=True)
            mean_train_error = compute_global_error(pairwise_err_train)
            pairwise_err_test = pairwise_distances(mature_torch_neurons, torch.tensor(test_data), cuda=True)
            mean_test_error = compute_global_error(pairwise_err_test)
            actual_mature_neurons = mature_torch_neurons.shape[0]
            if actual_mature_neurons > 0:
                num_mature_neurons.append(actual_mature_neurons)
                global_train_err.append(mean_train_error)
                global_test_err.append(mean_test_error)
                if len(num_mature_neurons) > 1:
                    mature_neurons_ratio = num_mature_neurons[-2] / num_mature_neurons[-1]
                    train_err_ratio = global_train_err[-1]/global_train_err[-2]
                    test_err_ratio = global_test_err[-1]/global_test_err[-2]

        if mean_train_error is not np.inf:
            ep += 1
        print("epoch [{}/{}] - Train MEE : {:.4f} - Test MEE : {:.4F} - #num_embryo/mature neurons: {} {} - Epsilon : {:.4f} - Time :{} - Process:{}%"
              .format(ep, args.epochs, mean_train_error, mean_test_error, num_embryo_neurons,
                      actual_mature_neurons, igng.epsilon, time_since(start), round(ep/args.epochs*100), 3), end="\n")


        if igng.epsilon > eps/2:
            igng.epsilon = igng.epsilon - igng.epsilon * 0.15
            if actual_mature_neurons < 5:
                igng.epsilon = igng.epsilon - igng.epsilon * 0.2
        # if mature_neurons_ratio <= 0.95 and (actual_mature_neurons < 0.1*train_data.shape[0]):
        if mature_neurons_ratio > 0.99 and test_err_ratio > 0.95 and ep > 5:
            break
        # if test_err_ratio > 0.98:
        #     break
        # else:
        #     break

    # gng.fit_network(e_b=0.05, e_n=0.006, a_max=8, l=100, a=0.5, d=0.995, passes=10, plot_evolution=False)
    # gng.plot_clusters(clustered_data)
    mature_torch_neurons = igng.get_mature_neurons(training=False)
    pairwise_err = pairwise_distances(mature_torch_neurons, torch.tensor(data_scaled))
    torch_neurons = igng.Units
    if igng.cuda:
        mature_torch_neurons = mature_torch_neurons.cpu()
        torch_neurons = torch_neurons.cpu().data.numpy()

    mature_neurons = mature_torch_neurons.data.numpy()

    min_distances = torch.transpose(min_distance_from_centroids(pairwise_err),
                                    dim0=0, dim1=-1).numpy()
    print('min_distances DONE')

    avg_distances_from_K_centroids = torch.transpose(avg_k_distance_from_centroids(pairwise_err),
                                                     dim0=0, dim1=-1).numpy()
    print('avg_distances_from_K_centroids DONE')

    max_distances_from_K_centroids = torch.transpose(max_k_distance_from_centroids(pairwise_err),
                                                     dim0=0, dim1=-1).numpy()
    outlier_K_factor_in_cluster = torch.transpose(outlier_factor_in_cluster(pairwise_err),
                                                  dim0=0, dim1=-1).numpy()
    print('outlier_K_factor_in_cluster DONE')

    data_cluster_sparsity = cluster_sparsity(pairwise_err)
    print('data_cluster_sparsity DONE')

    lof_cluster = local_outlier_factor_cluster(pairwise_err, torch.tensor(data_scaled), k=20, k_lof=5)
    print('lof_cluster DONE')

    # abod_cluster = abod_cluster(pairwise_err, torch.tensor(data_scaled), k=20)
    # print('loci_cluster DONE')
    if args.test:
        outliers = pd.DataFrame(data=np.concatenate((min_distances.reshape(-1, 1),
                                                     avg_distances_from_K_centroids.reshape(-1, 1),
                                                     max_distances_from_K_centroids.reshape(-1, 1),
                                                     outlier_K_factor_in_cluster.reshape(-1, 1),
                                                     lof_cluster.reshape(-1, 1),
                                                     data_cluster_sparsity.reshape(-1, 1)),
                                                    axis=1),
                                columns=['min_distances', 'avg_k_distances', 'max_k_distances', 'outlier_K_factor',
                                         'lof_clusters', 'cluster_sparsity'])
    else:

        outliers = pd.DataFrame(data=np.concatenate((min_distances.reshape(-1, 1),
                                                     avg_distances_from_K_centroids.reshape(-1, 1),
                                                     max_distances_from_K_centroids.reshape(-1, 1),
                                                     outlier_K_factor_in_cluster.reshape(-1, 1),
                                                     lof_cluster.reshape(-1, 1),
                                                     data_cluster_sparsity.reshape(-1, 1),
                                                     # abod_cluster.reshape(-1, 1),
                                                     labels.reshape(-1, 1)),
                                                    axis=1),
                                columns=['min_distances', 'avg_k_distances', 'max_k_distances', 'outlier_K_factor',
                                         'lof_clusters', 'cluster_sparsity', 'label'])

    if args.create or args.synth:
        # wksp_folder = os.path.dirname(args.filename)
        # cluster_img = os.path.join(wksp_folder, dataset_type + '_' + outlier_type + 'neurons.png')
        fig, ax = plt.subplots()
        ax.scatter(x=data_scaled[:, 0], y=data_scaled[:, 1], s=10, c='b', marker="s", label='data')
        # ax.scatter(x=data_scaled[:n_outliers, 0], y=data_scaled[:n_outliers, 1], s=12, c='G', marker="s", label='outliers')
        ax.scatter(x=torch_neurons[:, 0], y=torch_neurons[:, 1], s=8, c='y', marker="o", label='neurons')
        ax.scatter(x=mature_neurons[:, 0], y=mature_neurons[:, 1], s=10, c='r', marker="o", label='matures')
        plt.show()


        node_pos = {}
        i = 0
        for neu in mature_neurons:
            node_pos[i] = (neu[0], neu[1])
            i += 1
        nx.draw(igng.network, pos=node_pos)
        plt.show()

    fig1, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MEE', color=color)
    ax1.plot(global_train_err, label='Train MEE', color='blue')
    # ax1.plot(global_test_err, label='Test MEE', color='red')
    ax1.legend()
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Num Mature Neurons', color=color)
    ax2.plot(num_mature_neurons, label='Mature neurons', linestyle='dashed', color='green')
    ax2.legend()
    fig1.tight_layout()
    plt.show()



    if args.save:
        wksp_folder = os.path.dirname(args.filename)
        file_name = os.path.basename(args.filename)
        if args.test:
            file_name = os.path.basename(args.filename)
            outliers_file = os.path.join(wksp_folder, file_name + '_IGNG_outliers.csv')
            outliers.to_csv(outliers_file, index=None)
        else:
            outliers_file = os.path.join(wksp_folder, file_name +'_IGNG_outliers.csv')
            outliers.to_csv(outliers_file, index=None)
        if args.create:
            data_file = os.path.join(wksp_folder, dataset_type +'_'+ outlier_type+'_input_data.csv')
            data_ = pd.DataFrame(data=np.concatenate((data_scaled, labels.reshape(-1, 1)), axis=1))
            data_.to_csv(data_file, header=None, index=None)

    print('*** FINISH ***')
