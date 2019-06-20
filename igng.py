__authors__ = 'Antonio Ritacco'
__email__ = 'ritacco.ant@gmail.com'

import numpy as np
import torch.nn as nn
import torch
from collections import defaultdict
from torch.nn.modules.distance import PairwiseDistance

from helpers import pairwise_distances







class IncrementalGrowingNeuralGas:

    def __init__(self, epsilon, amature, alfac1, alfacN):
        self.Units = None
        self.Ages = None
        self.Connections = dict()
        self.epsilon = epsilon
        self.amature = amature
        self.alfac1 = alfac1
        self.alfacN = alfacN
        self.Error = 0

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


    def compute_global_error(self, mature_neurons, data):
        err = pairwise_distances(mature_neurons, data)
        mean_err = torch.mean(torch.min(err, dim=0, keepdim=True).values)
        return mean_err

    def getMatureNeurons(self):
        neuronsList = []
        neuronIndexes = []
        i = 0

        for age in self.Ages:
            if age >= self.amature:
                neuronsList.append(self.Units[i])
                neuronIndexes.append(i)
            i += 1
        if(len(neuronsList)>0):
            return torch.stack(neuronsList), neuronIndexes
        else:
            return None, None


