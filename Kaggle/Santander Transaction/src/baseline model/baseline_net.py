import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)


class Net(nn.Module):
    """
    A basic deep neural network.

    Attributes
    ----------
    neural_net : `torch.nn.Sequential`
        The deep neural network
    """

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=50),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=50, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, X):
        """
        Passes X through the Net.
        """

        return self.neural_net(X).view(-1)
