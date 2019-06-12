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

RHO = 0.3
BETA = 0.01
NUM_EPOCHS = 100
batchSIZE = 128
encodedSIZE = 2
noisePERC = 0.05

def kl_divergence(p, q):
    '''
    args:
        2 tensors `p` and `q`
    returns:
        kl divergence between the softmax of `p` and `q`
    '''
    # p = F.softmax(p)
    # q = F.softmax(q)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


def train(model, data, numEpochs= NUM_EPOCHS, rho = None, numHiddenNeurons = 50 , numEncodingNeurons = 30):
    data = torch.Tensor(data)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-5)
    for epoch in range(numEpochs):
        # loss = 0
        MSE_loss = 0
        dataBatched = 0
        activationHiddenSum = 0
        activationEncodedSum = 0
        ##########  rho_hat = SOMMMARE ATTIVAZIONI DEI NEURONI NELLA BATCH CORRENTE (M*HIDDEN)
        ### PER OGNI NEURONE j fare KL_DIV
        for dataSample in data:
            noisy_sample = add_noise(dataSample)

            sample = Variable(dataSample).cuda()
            # noisy_sample = Variable(noisy_sample).cuda()

            # ===================forward=====================
            encoded, encoderLast, decoded = model(sample)
            MSE_loss += nn.MSELoss()(decoded, sample)
            if rho is not None:
                kl_loss = kl_divergence(rho, encoded)*BETA
                MSE_loss += kl_loss
            activationHiddenSum += torch.sum(encoded,dim=0).data.cpu()
            activationEncodedSum += torch.sum(encoderLast,dim=0).data.cpu()

            # else:
            #     loss += MSE_loss
            # MSE_loss += nn.MSELoss()(decoded, sample)
            dataBatched += 1
            # ===================backward====================
            if (dataBatched % batchSIZE == 0 and dataBatched > 0):
                MSE_loss = MSE_loss/batchSIZE
                activationHiddenSum = activationHiddenSum/batchSIZE
                activationEncodedSum = activationEncodedSum/batchSIZE

                # MSE_loss += kl_loss
                optimizer.zero_grad()
                MSE_loss.backward()
                optimizer.step()
                dataBatched = 0
                currentMSE_loss = MSE_loss.data.cpu()
                MSE_loss = 0
                # MSE_loss=0
            # ===================log========================
                print('\repoch [{}/{}], MSE_loss:{:.4f}, Activation Hidden sum : {:.4f}, Activation Encoded sum : {:.4f}'
                      .format(epoch + 1, numEpochs, currentMSE_loss, 
                      activationHiddenSum, activationEncodedSum), end="")
        # if epoch % 10 == 0:
        #     print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
        #           .format(epoch + 1, numEpochs, currentloss, currentMSE_loss))
        #     torch.save(model.state_dict(), './sim_autoencoder.pth')


def add_noise(sample):
    noise = (torch.randn(sample.size()) * noisePERC)
    noise = noise.type(torch.float32)
    noisy_sample = sample + noise
    return noisy_sample

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--filename', dest='filename', type=str)
    args = argparser.parse_args()
    datasetOrig = pd.read_csv(args.filename, header=None)
    data = np.array(datasetOrig)
    scaler = preprocessing.MinMaxScaler()
    data_scaled = scaler.fit_transform(data[:, :-1])

    # gng = GrowingNeuralGas(input_data = data_scaled)
    # gng.fit_network(e_b=0.05, e_n=0.006, a_max=8, l=100, a=0.5, d=0.995, passes=20, plot_evolution=False)
    # clustered_data = gng.cluster_data()
    # gng.plot_clusters(clustered_data)

    inputSize = data_scaled.shape[1]
    hiddenSize = inputSize
    encodedSize = encodedSIZE
    model = AutoEncoder(inputSize, hiddenSize, encodedSize)
    model.cuda()
    rho = None
    if RHO is not None:
        rho = torch.zeros([hiddenSize]).cuda()
        rho.fill_(RHO)
    train(model, data_scaled, rho=rho, numHiddenNeurons=hiddenSize)

    # print('\n')
    # sampleActual = Variable(torch.Tensor(data_scaled[0, :])).cuda()
    # resultEncoding = model.encoder(sampleActual).cuda()
    # print(resultEncoding)
    # sampleActual = Variable(torch.Tensor(data_scaled[1, :])).cuda()
    # resultEncoding = model.encoder(sampleActual).cuda()
    # print(resultEncoding)
    # sampleActual = Variable(torch.Tensor(data_scaled[12, :])).cuda()
    # resultEncoding = model.encoder(sampleActual).cuda()
    # print(resultEncoding)
    # sampleActual = Variable(torch.Tensor(data_scaled[17, :])).cuda()
    # resultEncoding = model.encoder(sampleActual).cuda()
    # print(resultEncoding)


    sampleActual = Variable(torch.Tensor(data_scaled)).cuda()

    encoded, encoderLast, decoded = model(sampleActual)
    diffDecodedActual = torch.sum(torch.abs(sampleActual - decoded),dim=1)
    
    diffDecodedActual = diffDecodedActual/torch.max(diffDecodedActual)
    resultToCPU = encoderLast.data.cpu().numpy()
    datasetResult = pd.DataFrame()
    datasetResult['X'] = pd.Series(resultToCPU[:, 0], index=datasetOrig.index)
    datasetResult['Y'] = pd.Series(resultToCPU[:, 1], index=datasetOrig.index)
    # datasetResult['Z'] = pd.Series(resultToCPU[:, 2], index=datasetOrig.index)
    datasetResult['Diff'] = pd.Series(diffDecodedActual.data.cpu(), index=datasetOrig.index)
    datasetResult['Target'] = pd.Series(datasetOrig.iloc[:, -1], index=datasetOrig.index)
    resultFile = str(args.filename)+'_'+'ResultEncoding.csv'
    datasetResult.to_csv(resultFile)
    print('Finish')

    # #%%
    # plt.plot(datasetOrig['X'],datasetOrig['Y'])
    # plt.show()
