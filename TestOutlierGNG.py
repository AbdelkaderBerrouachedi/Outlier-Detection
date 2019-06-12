__authors__ = 'Antonio Ritacco'
__email__ = 'ritacco.ant@gmail.com'


import pandas as pd
import numpy as np
import argparse
from gng import GrowingNeuralGas
from AutoEncoder import AutoEncoder
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import torch.distributions.kl as kl
import torch
import torch.nn as nn

from torch.autograd import Variable


from sklearn import preprocessing

BETA = 0.01
NUM_EPOCHS = 100
batchSIZE = 32



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--filename', dest='filename', type=str)
    args = argparser.parse_args()
    datasetOrig = pd.read_csv(args.filename, header=None)
    data = np.array(datasetOrig)
    scaler = preprocessing.MinMaxScaler()
    data_scaled = scaler.fit_transform(data[:, :-1])

    gng = GrowingNeuralGas(input_data = data_scaled)
    gng.fit_network(e_b=0.05, e_n=0.006, a_max=8, l=100, a=0.5, d=0.995, passes=10, plot_evolution=False)
    clustered_data = gng.cluster_data()
    # gng.plot_clusters(clustered_data)

    
    print('Finish')
