import time
import math
import torch
import collections
import numpy as np
"""
Some properties
1) ratio between number of element in the cluster and most populated cluster
2) min distance from nearest cluster
3) distances from k nearest centroids (min, max, avg)
"""

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def pairwise_distances(x, y):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matrix
    Output: dist is a NxM (Nx1 with the 0-axis sum) matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist



def compute_global_error(mature_neurons, data, cuda):
    if cuda:
        mature_neurons = mature_neurons.cuda()
        data = data.cuda()
    err = pairwise_distances(mature_neurons, data)
    mean_err = torch.mean(torch.min(err, dim=0, keepdim=True).values)
    return mean_err

def min_distance_from_centroids(clusters, data):
    err = pairwise_distances(clusters, data)
    vector_quantization_error = torch.min(err, dim=0, keepdim=True).values
    return vector_quantization_error

def avg_k_distance_from_centroids(clusters, data, k=5):
    err = pairwise_distances(clusters, data)
    vector_quantization_error = torch.mean(torch.topk(err, dim=0, k=k, largest=False).values, keepdim=True, dim=0)
    return vector_quantization_error

def max_k_distance_from_centroids(clusters, data, k=5):
    err = pairwise_distances(clusters, data)
    vector_quantization_error = torch.max(torch.topk(err, dim=0, k=k, largest=False).values, keepdim=True, dim=0).values
    return vector_quantization_error

def outlier_factor_in_cluster(clusters, data, k=5):
    err = pairwise_distances(clusters, data)
    nearest_vals = torch.min(err, dim=0, keepdim=True)
    nearest_units = nearest_vals.indices.numpy().flatten()
    min_errs = nearest_vals.values
    avgtopKerr_list = []
    for index in nearest_units:
        avgtopKerr = torch.mean(torch.topk(err[index, :], k=k, largest=False).values, keepdim=True, dim=0)
        avgtopKerr_list.append(avgtopKerr)
    avgtopKerr_tensor = torch.tensor(avgtopKerr_list)
    return torch.abs(min_errs.view(-1, 1) - avgtopKerr_tensor.view(-1, 1))

def cluster_sparsity(clusters, data):
    err = pairwise_distances(clusters, data)
    nearest_vals = torch.min(err, dim=0, keepdim=True)
    nearest_units = nearest_vals.indices.numpy().flatten()
    counter = collections.Counter(nearest_units)
    most_populated_cluster = collections.Counter(nearest_units).most_common(1)[0][1]
    counter_dict = dict(counter)
    data_density = []
    for index in nearest_units:
        data_density.append(1 - counter[index]/most_populated_cluster)
    return np.array(data_density)
