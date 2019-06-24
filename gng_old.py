# coding: utf-8

import numpy as np
from scipy import spatial
import networkx as nx
import time
import matplotlib.pyplot as plt
from sklearn import decomposition
import copy
import os
import torch
from helpers import compute_global_error, time_since

__authors__ = 'Adrien Guille, forked by Antonio Ritacco'
__email__ = 'adrien.guille@univ-lyon2.fr, ritacco.ant@gmail.com'

'''
Simple implementation of the Growing Neural Gas algorithm, based on:
A Growing Neural Gas Network Learns Topologies. B. Fritzke, Advances in Neural
Information Processing Systems 7, 1995.
'''


class GrowingNeuralGas:

    def __init__(self, input_data):
        self.network = None
        self.data = input_data
        self.units_created = 0

    def find_nearest_units(self, observation):
        distance = []
        for u, attributes in self.network.nodes(data=True):
            vector = attributes['vector']
            dist = spatial.distance.euclidean(vector, observation)
            distance.append((u, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        return ranking

    def prune_connections(self, a_max):
        listToRemoveE = []
        listToRemoveU = []
        for u, v, attributes in self.network.edges(data=True):
            if attributes['age'] > a_max:
                listToRemoveE.append([u, v])
        for u in self.network.nodes():
            if self.network.degree(u) == 0:
                listToRemoveU.append(u)
        try:
            self.network.remove_edges_from(listToRemoveE)
            self.network.remove_nodes_from(listToRemoveU)
        except:
            print('Error while removing...')
            print('Edges to remove', listToRemoveE)
            print('Nodes to remove', listToRemoveU)

    def fit_network(self, e_b, e_n, a_max, l, a, d, passes=1, test_data=None, global_train_err=None,
                    global_test_err=None, num_mature_neurons = None):
        # logging variables
        accumulated_local_error = []
        network_order = []
        network_size = []
        total_units = []
        self.units_created = 0
        # 0. start with two units a and b at random position w_a and w_b
        w_a = [np.random.uniform(-2, 2) for _ in range(np.shape(self.data)[1])]
        w_b = [np.random.uniform(-2, 2) for _ in range(np.shape(self.data)[1])]
        self.network = nx.Graph()
        self.network.add_node(self.units_created, vector=w_a, error=0)
        self.units_created += 1
        self.network.add_node(self.units_created, vector=w_b, error=0)
        self.units_created += 1
        # 1. iterate through the data
        sequence = 0
        start = time.time()
        for p in range(passes):
            # print('   Pass #%d' % (p + 1))
            np.random.shuffle(self.data)
            steps = 0
            for observation in self.data:
                # 2. find the nearest unit s_1 and the second nearest unit s_2
                nearest_units = self.find_nearest_units(observation)
                s_1 = nearest_units[0]
                s_2 = nearest_units[1]
                # 3. increment the age of all edges emanating from s_1
                for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
                    self.network.add_edge(u, v, age=attributes['age']+1)
                # 4. add the squared distance between the observation and the nearest unit in input space
                self.network.node[s_1]['error'] += spatial.distance.euclidean(observation, self.network.node[s_1]['vector'])**2
                # 5 .move s_1 and its direct topological neighbors towards the observation by the fractions
                #    e_b and e_n, respectively, of the total distance
                update_w_s_1 = e_b * (np.subtract(observation, self.network.node[s_1]['vector']))
                self.network.node[s_1]['vector'] = np.add(self.network.node[s_1]['vector'], update_w_s_1)
                update_w_s_n = e_n * (np.subtract(observation, self.network.node[s_1]['vector']))
                for neighbor in self.network.neighbors(s_1):
                    self.network.node[neighbor]['vector'] = np.add(self.network.node[neighbor]['vector'], update_w_s_n)
                # 6. if s_1 and s_2 are connected by an edge, set the age of this edge to zero
                #    if such an edge doesn't exist, create it
                self.network.add_edge(s_1, s_2, age=0)
                # 7. remove edges with an age larger than a_max
                #    if this results in units having no emanating edges, remove them as well
                self.prune_connections(a_max)
                # 8. if the number of steps so far is an integer multiple of parameter l, insert a new unit
                steps += 1
                if steps % l == 0:
                    sequence += 1
                    # 8.a determine the unit q with the maximum accumulated error
                    q = 0
                    error_max = 0
                    for u in self.network.nodes():
                        if self.network.node[u]['error'] > error_max:
                            error_max = self.network.node[u]['error']
                            q = u
                    # 8.b insert a new unit r halfway between q and its neighbor f with the largest error variable
                    f = -1
                    largest_error = -1
                    for u in self.network.neighbors(q):
                        if self.network.node[u]['error'] > largest_error:
                            largest_error = self.network.node[u]['error']
                            f = u
                    w_r = 0.5 * (np.add(self.network.node[q]['vector'], self.network.node[f]['vector']))
                    r = self.units_created
                    self.units_created += 1
                    # 8.c insert edges connecting the new unit r with q and f
                    #     remove the original edge between q and f
                    self.network.add_node(r, vector=w_r, error=0)
                    self.network.add_edge(r, q, age=0)
                    self.network.add_edge(r, f, age=0)
                    self.network.remove_edge(q, f)
                    # 8.d decrease the error variables of q and f by multiplying them with a
                    #     initialize the error variable of r with the new value of the error variable of q
                    self.network.node[q]['error'] *= a
                    self.network.node[f]['error'] *= a
                    self.network.node[r]['error'] = self.network.node[q]['error']
                # 9. decrease all error variables by multiplying them with a constant d
                error = 0
                for u in self.network.nodes():
                    error += self.network.node[u]['error']
                accumulated_local_error.append(error)
                network_order.append(self.network.order())
                network_size.append(self.network.size())
                total_units.append(self.units_created)
                for u in self.network.nodes():
                    self.network.node[u]['error'] *= d
                    if self.network.degree(nbunch=[u]) == 0:
                        print(u)
            # global_error.append(self.compute_global_error())
            neurons = []
            for k in self.network._node:
                neurons.append(self.network._node[k]['vector'])
            mature_torch_neurons = torch.tensor(neurons)
            mean_train_error = compute_global_error(mature_torch_neurons, torch.tensor(self.data), cuda=True)
            mean_test_error = compute_global_error(mature_torch_neurons, torch.tensor(test_data), cuda=True)
            global_train_err.append(mean_train_error)
            global_test_err.append(mean_test_error)
            actual_mature_neurons = mature_torch_neurons.shape[0]
            mature_neurons_ratio = num_mature_neurons[-1] / actual_mature_neurons
            num_mature_neurons.append(actual_mature_neurons)

            print(
                "\repoch [{}/{}] - Train euclidean error : {:.4f} - Test euclidean error : {:.4f} - #mature neurons: {} - Time :{} - Process:{}%"
                .format(p + 1, passes, mean_train_error, mean_test_error, actual_mature_neurons, time_since(start),
                        round(p / passes * 100), 3), end="")
            if mature_neurons_ratio > 0.98:
                break;




    def number_of_clusters(self):
        return nx.number_connected_components(self.network)

    def cluster_data(self):
        unit_to_cluster = np.zeros(self.units_created)
        cluster = 0
        for c in nx.connected_components(self.network):
            for unit in c:
                unit_to_cluster[unit] = cluster
            cluster += 1
        clustered_data = []
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s = nearest_units[0]
            clustered_data.append((observation, unit_to_cluster[s]))
        return clustered_data

    def reduce_dimension(self, clustered_data):
        transformed_clustered_data = []
        svd = decomposition.PCA(n_components=2)
        transformed_observations = svd.fit_transform(self.data)
        for i in range(len(clustered_data)):
            transformed_clustered_data.append((transformed_observations[i], clustered_data[i][1]))
        return transformed_clustered_data

    def compute_global_error(self):
        global_error = 0
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s_1 = nearest_units[0]
            global_error += spatial.distance.euclidean(observation, self.network.node[s_1]['vector'])**2
        return global_error



