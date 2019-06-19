__authors__ = 'Antonio Ritacco'
__email__ = 'ritacco.ant@gmail.com'

import numpy as np
import torch.nn as nn
import torch
from collections import defaultdict
from torch.nn.modules.distance import PairwiseDistance


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






class IncrementalGrowingNeuralGas:

    def __init__(self, epsilon, amature, alfac1, alfacN):
        self.Units = None
        self.Ages = None
        self.Connections = dict()
        self.epsilon = epsilon
        self.amature = amature
        self.alfac1 = alfac1
        self.alfacN = alfacN

    def findWinning(self, x):
        if self.Units is None:
            val1 = None
            val2 = None
            index1 = None
            index2 = None
        else:
            distance_vector = pairwise_distances(self.Units, x)
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


    def forward(self, x):
        distDict = self.findWinning(x)
        if distDict['val1'] is None or distDict['val1'] >= self.epsilon:
            if distDict['val1'] is None:
                self.Units = x.clone().detach().requires_grad_(False)
                self.Ages = torch.tensor([0.0], requires_grad=False)
            else:
                self.Units = torch.cat((self.Units, x.clone().detach().requires_grad_(False)))
                self.Ages = torch.cat((self.Ages, torch.tensor([1.0], requires_grad=False)))
        else:
            bestUnit = distDict['index1'].item()
            newUnit = None
            self.Units.shape[0]
            if distDict['index2'] is not None:
                newUnit = distDict['index2'].item()

            if distDict['val2'] is None or distDict['val2'] >= self.epsilon:
                self.Units = torch.cat((self.Units, x.clone().detach().requires_grad_(False)))
                self.Ages = torch.cat((self.Ages, torch.tensor([0.0], requires_grad=False)))
                newUnit = self.Units.shape[0]

            else:
                self.Units[bestUnit] += torch.reshape(self.alfac1*(x-self.Units[bestUnit]), (-1,))
                if bestUnit not in self.Connections.keys():
                    self.Units[newUnit] += torch.reshape(self.alfacN * (x - self.Units[newUnit]), (-1,))
                else:
                    for index in self.Connections[bestUnit]:
                        self.Units[index] += torch.reshape(self.alfacN*(x-self.Units[index]), (-1,))

                self.createConnection(bestUnit, newUnit)
                # if(distDict['index2'] not in self.Connections[distDict['index1']]):
                #     self.Connections[distDict['index1']].append(distDict['index2'])
                #     self.Ages[distDict['index1']] = 0.0
                #     self.Ages[distDict['index2']] = 0.0
                # else:
                #     self.Connections[distDict['index1']].append(distDict['index2'])


                for index in self.Connections[bestUnit]:
                    self.Ages[index] += 1.0


    def getMatureNeurons(self):
        neuronsList = []
        neuronIndexes = []
        i = 0
        for age in self.Ages:
            if age >= self.amature:
                neuronsList.append(self.Units[i].data.numpy())
                neuronIndexes.append(i)
            i += 1
        return np.array(neuronsList), neuronIndexes

