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

    def __init__(self, input_size, hidden_dim, p=0.3):
        super(Net, self).__init__()

        # Hidden dimension since the data is not correlated
        # we may use each feature of each input as its own input
        self.hidden_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # The deep neura network
        self.neural_net = nn.Sequential(
            nn.Linear(in_features=input_size//2*hidden_dim, out_features=50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(in_features=50, out_features=50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(in_features=50, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, X):
        """
        Passes X through the Net.
        """
        batch_size = len(X)

        # Unsqueeze the data do pass through the hidden dimension
        orig_features = X[:, :200].unsqueeze(2)  # Tensor(batch_size, 200, 1)
        new_features = X[:, 200:].unsqueeze(2)  # Tensor(batch_size, 200, 1)

        # Concat the squeezed features
        # Tensor(batch_size, 200, 2)
        X = torch.cat([orig_features, new_features], dim=2)

        # Pass through the hidden dimension
        X = self.hidden_net(X)  # Tensor(batch_size, 200, hidden_dim)

        # Reshape back to the 2D shape
        X = X.view(batch_size, -1)  # Tensor(batch_size, 200*hidden_dim)

        # Pass through the deep neural net
        return self.neural_net(X).view(-1)
