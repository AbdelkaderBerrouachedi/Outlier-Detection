__authors__ = 'Antonio Ritacco'
__email__ = 'ritacco.ant@gmail.com'

import numpy as np
import torch.nn as nn
import torch
from torch_geometric.data import Data
from collections import defaultdict
from torch.nn.modules.distance import PairwiseDistance

from helpers import pairwise_distances
import networkx as nx



'''
Pytorch implementation of the Incremental Growing Neural Gas algorithm, based on:
An Incremental Growing Neural Gas Learns Topologies. Y. Prudent,
2005 IEEE International Joint Conference on Neural Networks
'''




class IncrementalGrowingNeuralGas:

    def __init__(self, epsilon, amature, alfac1, alfacN, cuda=False):
        self.Units = None
        self.Ages = None
        self.Connections = dict()
        self.epsilon = epsilon
        self.amature = amature
        self.alfac1 = alfac1
        self.alfacN = alfacN
        self.Error = 0
        self.cuda = cuda
        self.network = None

    def findWinning(self, x):
        if self.Units is None:
            val1 = None
            val2 = None
            index1 = None
            index2 = None
        else:
            if self.cuda:
                distance_vector = PairwiseDistance(self.Units.cuda(), x.cuda())
                distance_vector.to('cpu')
            else:
                distance_vector = PairwiseDistance(self.Units, x)
            if self.Units.shape[0] < 2:
                tuples = torch.topk(distance_vector, k=1, largest=False)
                val1 = tuples.values[0]
                index1 = tuples.indices[0]
                val2 = None
                index2 = None
            else:
                tuples = torch.topk(torch.reshape(distance_vector, (-1,)), k=2, largest=False)
                val1 = tuples.values[0]
                val2 = tuples.values[1]
                index1 = tuples.indices[0]
                index2 = tuples.indices[1]

        return {'val1': val1, 'index1': index1, 'val2': val2, 'index2': index2}


    def createConnection(self, bestUnit, newUnit):
        if bestUnit not in self.Connections.keys():
            self.Connections[bestUnit] = [newUnit]
        else:
            if newUnit not in self.Connections[bestUnit]:
                self.Connections[bestUnit].append(newUnit)

        if newUnit not in self.Connections.keys():
            self.Connections[newUnit] = [bestUnit]
        else:
            if bestUnit not in self.Connections[newUnit]:
                self.Connections[newUnit].append(bestUnit)


    def create_connection(self, best_unit, new_unit):
        self.network.add_node(best_unit)
        self.network.add_node(new_unit)




    def _forward(self, x):
        distDict = self.findWinning(x)
        if distDict['val1'] is None:
            self.network = nx.Graph()
            node_id = self.network.number_of_nodes()
            self.network.add_node(node_id, age=0)
            self.Units = x.clone().detach().requires_grad_(False)
        else:
            best_unit = distDict['index1'].item()
            if distDict['val1'] >= self.epsilon:
                node_id = self.network.number_of_nodes()
                self.network.add_node(node_id, age=0)
                self.Units = torch.cat((self.Units, x.clone().detach().requires_grad_(False)))
            else:
                if distDict['val2'] is None or distDict['val2'] >= self.epsilon:
                    node_id = self.network.number_of_nodes()
                    self.network.add_node(node_id, age=0)
                    self.Units = torch.cat((self.Units, x.clone().detach().requires_grad_(False)))
                    self.network.add_edge(best_unit, node_id, age=0)
                else:
                    second_best_unit = distDict['index2'].item()
                    self.Units[best_unit] += torch.reshape(self.alfac1 * (x - self.Units[best_unit]), (-1,))

                    if best_unit not in self.network.nodes():
                        self.network.add_node(best_unit, age=0)
                        for u in self.network.neighbors(best_unit):
                            self.Units[u] += torch.reshape(self.alfacN * (x - self.Units[u]), (-1,))
                            self.network._adj[best_unit][u]['age'] += 1
                    # self.network._node[u]


                    if second_best_unit in self.network.neighbors(best_unit):
                        self.network._adj[best_unit][second_best_unit]['age'] = 0
                    else:
                        self.network.add_edge(best_unit, second_best_unit, age=0)

                    ## <Remove edges too old>
                    edge_list_to_remove = []
                    for u, v, attrib in self.network.edges(data=True):
                        if attrib['age'] >= self.amature:
                            edge_list_to_remove.append([u, v])
                    self.network.remove_edges_from(edge_list_to_remove)
                    ##</Remove edges too old>

                    ## <Increasing nodes age of the winner unit>
                    for u in self.network.neighbors(best_unit):
                        self.network._node[u]['age'] += 1
                    ## </Increasing nodes age of the winner unit>

                    ## <Remove nodes too isolated>
                    node_list_to_remove = []
                    for n in self.network.nodes():
                        if len(self.network._adj[n]) < 1:
                            node_list_to_remove.append(n)
                    if node_list_to_remove:
                        list_to_keep = list(set(list(range(0, self.Units.shape[0]))) - set(node_list_to_remove))
                        self.Units = torch.index_select(self.Units, 0, torch.tensor(list_to_keep))
                        self.network.remove_nodes_from(node_list_to_remove)
                        old_index_nodes = [name for name in self.network.nodes()]
                        new_index_nodes = list(range(0, len(list_to_keep)))
                        mapping = dict(zip(old_index_nodes, new_index_nodes))
                        self.network = nx.relabel_nodes(self.network, mapping)
                    ## </Remove nodes too isolated>





    #
    # def forward(self, x):
    #     if self.cuda:
    #         x = x.cuda()
    #     distDict = self.findWinning(x)
    #     if distDict['val1'] is None or distDict['val1'] >= self.epsilon:
    #         if distDict['val1'] is None:
    #             self.Units = x.clone().detach().requires_grad_(False)
    #             self.Ages = torch.tensor([0.0], requires_grad=False)
    #         else:
    #             self.Units = torch.cat((self.Units, x.clone().detach().requires_grad_(False)))
    #             self.Ages = torch.cat((self.Ages, torch.tensor([0.0], requires_grad=False)))
    #     else:
    #         bestUnit = distDict['index1'].item()
    #         newUnit = self.Units.shape[0]
    #         if distDict['index2'] is not None:
    #             newUnit = distDict['index2'].item()
    #
    #         if distDict['val2'] is None or distDict['val2'] >= self.epsilon:
    #             self.Units = torch.cat((self.Units, x.clone().detach().requires_grad_(False)))
    #             self.Ages = torch.cat((self.Ages, torch.tensor([0.0], requires_grad=False)))
    #             # newUnit = self.Units.shape[0]
    #
    #         else:
    #
    #             self.Units[bestUnit] += torch.reshape(self.alfac1*(x-self.Units[bestUnit]), (-1,))
    #             if bestUnit not in self.Connections.keys():
    #                 self.Units[newUnit] += torch.reshape(self.alfacN * (x - self.Units[newUnit]), (-1,))
    #             else:
    #                 for index in self.Connections[bestUnit]:
    #                     self.Units[index] += torch.reshape(self.alfacN*(x-self.Units[index]), (-1,))
    #             self.createConnection(bestUnit, newUnit)
    #
    #             if distDict['index2'] not in self.Connections[distDict['index1']]:
    #                 self.Connections[distDict['index1']].append(distDict['index2'])
    #                 self.Ages[distDict['index1']] = 0.0
    #                 self.Ages[distDict['index2']] = 0.0
    #             else:
    #                 self.Connections[distDict['index1']].append(distDict['index2'])
    #
    #
    #             for index in self.Connections[bestUnit]:
    #                 self.Ages[index] += 1.0


    # def getMatureNeurons(self):
    #     neuronsList = []
    #     neuronIndexes = []
    #     i = 0
    #
    #     for age in self.Ages:
    #         if age >= self.amature:
    #             neuronsList.append(self.Units[i])
    #             neuronIndexes.append(i)
    #         i += 1
    #     if(len(neuronsList)>0):
    #         return torch.stack(neuronsList), neuronIndexes
    #     else:
    #         return None, None

    def get_mature_neurons(self, training=True):
        neuronsList = []
        i = 0
        if training:
            for node_id in self.network._node:
                if self.network._node[node_id]['age'] >= self.amature:
                    neuronsList.append(self.Units[node_id])
                i += 1
        else:
            neuron_to_remove = []
            for node_id in self.network._node:
                if self.network._node[node_id]['age'] < self.amature:
                    neuron_to_remove.append(node_id)
            if neuron_to_remove:
                list_to_keep = list(set(list(range(0, self.Units.shape[0]))) - set(neuron_to_remove))
                neuronsList = torch.index_select(self.Units, 0, torch.tensor(list_to_keep))
                self.network.remove_nodes_from(neuron_to_remove)
                old_index_nodes = [name for name in self.network.nodes()]
                new_index_nodes = list(range(0, len(list_to_keep)))
                mapping = dict(zip(old_index_nodes, new_index_nodes))
                self.network = nx.relabel_nodes(self.network, mapping)

        if training:
            if (len(neuronsList) > 0):
                return torch.stack(neuronsList)
            else:
                return None
        else:
            if (neuronsList.shape[0]>0):
                return neuronsList
            else:
                return None
