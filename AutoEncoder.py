import torch
import torch.nn as nn
from torch.autograd import Variable


class AutoEncoder(nn.Module):

    def __init__(self, inputSize, hiddenSize):

        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            # nn.ReLU(True),
            # nn.Linear(hiddenSize, encodedSize),
            # nn.ReLU(True),
            # nn.Linear(16, 2),
            nn.ReLU(True))

        self.encoder3D = nn.Sequential(
            nn.Linear(hiddenSize, 3),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, hiddenSize),
            nn.ReLU(True),
            # nn.Linear(encodedSize, hiddenSize),
            # nn.ReLU(True),
            nn.Linear(hiddenSize, inputSize),
            nn.Sigmoid())


    def forward(self, x):
        encoded = self.encoder(x)
        encoded3D = self.encoder3D(encoded)
        decoded = self.decoder(encoded3D)
        return encoded, encoded3D, decoded

