# coding: utf-8

import numpy as np
from scipy import spatial
import torch
from helpers import pairwise_distances

import copy
import os


__authors__ = 'Antonio Ritacco'
__email__ = 'ritacco.ant@gmail.com'

'''
Pytorch implementation of the Growing Neural Gas algorithm, based on:
A Growing Neural Gas Network Learns Topologies. B. Fritzke, Advances in Neural
Information Processing Systems 7, 1995.
'''




class GrowingNeuralGas:

    def __init__(self, amature, alfac1, alfacN, startA, startB, lambdaParam, dParam, alfaParam):
        self.amature = amature
        self.alfac1 = alfac1
        self.alfacN = alfacN
        self.lambdaParam = lambdaParam
        # 0.
        self.Units = torch.cat((startA.view(1, -1), startB.view(1, -1)), 0)
        # 0.
        self.local_errors = dict()
        self.CountSignal = 0
        self.incidence_matrix = None
        self.dParam = dParam
        self.alfaParam = alfaParam




    def forward (self, x):
        # 1., 2.

        distance_vector = pairwise_distances(self.Units, x)
        tuples = torch.topk(torch.reshape(distance_vector, (-1,)), k=2, largest=False)

        val1 = tuples.values[0]
        val2 = tuples.values[1]
        s1 = tuples.indices[0].numpy().flatten()[0]
        s2 = tuples.indices[1].numpy().flatten()[0]

        # 1., 2.

        # 3.
        # Increment the age of all edges emanating from s1
        self.increment_age(s1)
        # 3.

        # 4.
        self.increment_error(s1, val1, x , tuples)
        # 4.

        # 5.
        if self.incidence_matrix is not None:
            for i in range(self.incidence_matrix.shape[0]):
                if i == s1:
                    self.Units[s1] += torch.reshape(self.alfac1 * (x - self.Units[s1]), (-1,))
                else:
                    if self.incidence_matrix[i, s1, 0] == 1:
                        self.Units[i] += torch.reshape(self.alfacN * (x - self.Units[i]), (-1,))
        # 5.

        # 6.
        if self.incidence_matrix is not None:
            if self.incidence_matrix[s1, s2, 0] == 1:
                self.reset_age(s1, s2)
            else:
                self.createConnection(s1, s2)
        else:
            self.createConnection(s1, s2)
        # 6.

        # 7.
        self.remove_edges()
        # 7.

        # 8.
        if self.CountSignal > 0 and self.CountSignal % self.lambdaParam == 0:
            self.create_unit()
        # 8.
        for key in self.local_errors.keys():
            self.local_errors[key] = self.local_errors[key] * self.dParam

    def create_unit(self):
        q_index = torch.topk(torch.tensor(list(self.local_errors.values())), k=1).indices.numpy().flatten()[0]
        q_val = self.Units[q_index]

        f_index = -1
        max_f = 0
        for i in range(self.incidence_matrix.shape[0]):
            if self.incidence_matrix[q_index, i, 0] == 1:
                if i in self.local_errors.keys():
                    if self.local_errors[i] > max_f:
                        max_f = self.local_errors[i]
                        f_index = i
        if f_index > -1:
        # f_index = torch.topk(self.local_errors[torch.nonzero(self.incidence_matrix[q_index, :, 0] == 1)], k=1).indices.numpy().flatten()[0]
            f_val = self.Units[f_index]
            r_index = self.incidence_matrix.shape[0]
            r_val = (q_val + f_val)*0.5
            self.Units = torch.cat((self.Units, r_val.view(1, -1)), 0)
            self.incidence_matrix[q_index, f_index, :] = 0
            self.incidence_matrix[f_index, q_index, :] = 0
            self.createConnection(q_index, r_index)
            self.createConnection(f_index, r_index)
            self.local_errors[q_index] = self.local_errors[q_index]*self.alfaParam
            self.local_errors[f_index] = self.local_errors[f_index] * self.alfaParam
            self.local_errors[r_index] = self.local_errors[q_index]

    def remove_edges(self):
        # for i in range(self.incidence_matrix.shape[0]):
        #     for j in range(self.incidence_matrix.shape[1]):
        #         if self.incidence_matrix[i, j, 1] > self.amature:
        #             self.incidence_matrix[i, j, :] = 0
        #             self.incidence_matrix[j, i, :] = 0
        self.incidence_matrix[self.incidence_matrix[:, :, 1] > self.amature] = 0

    def reset_age(self, s1, s2):
        self.incidence_matrix[s1, s2, 1] = 0

    def increment_age(self, s1):
        if self.incidence_matrix is not None:
            self.incidence_matrix[s1, :, 1] += 1*self.incidence_matrix[s1, :, 0]
            self.incidence_matrix[:, s1, 1] += 1 * self.incidence_matrix[:, s1, 0]

    def createConnection(self, s1, s2):
        if self.incidence_matrix is None:
            self.incidence_matrix = torch.zeros([2, 2, 2])
            self.incidence_matrix[:, :, 0] = 1
        else:
            if s1 >= self.incidence_matrix.shape[0] or s2 >= self.incidence_matrix.shape[0]:
                self.incidence_matrix = torch.cat((self.incidence_matrix, torch.zeros([1, self.incidence_matrix.shape[0], 2])), 0)
                self.incidence_matrix = torch.cat((self.incidence_matrix, torch.zeros([self.incidence_matrix.shape[0], 1,  2])), 1)
            self.incidence_matrix[s1, s2, 0] = 1

    def increment_error(self, s1, val1, x, tuples):
        if s1 not in self.local_errors.keys():
            self.local_errors[s1] = torch.pow(val1, 2)
        else:
            self.local_errors[s1] += torch.pow(val1, 2)
