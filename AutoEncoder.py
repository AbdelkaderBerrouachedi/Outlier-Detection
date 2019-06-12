import torch
import torch.nn as nn
from torch.autograd import Variable


class AutoEncoder(nn.Module):

    def __init__(self, inputSize, hiddenSize, encodedSize):

        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inputSize, hiddenSize,bias=True),
            # nn.ReLU(True),
            # nn.Linear(hiddenSize, encodedSize),
            # nn.ReLU(True),
            # nn.Linear(16, 2),
            nn.Sigmoid())

        self.encoderLast = nn.Sequential(
            nn.Linear(hiddenSize, encodedSize,bias=True),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encodedSize, hiddenSize,bias=True),
            nn.Sigmoid(),
            # nn.Linear(encodedSize, hiddenSize),
            # nn.ReLU(True),
            nn.Linear(hiddenSize, inputSize,bias=True),
            nn.Sigmoid())


    def forward(self, x):
        encoded = self.encoder(x)
        encoderLast = self.encoderLast(encoded)
        decoded = self.decoder(encoderLast)
        return encoded, encoderLast, decoded

