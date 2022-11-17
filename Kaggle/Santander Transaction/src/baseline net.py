import os
import numpy as np
import pandas as pd
import datetime
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from get_data import train_loader, val_loader, test_loader

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

    def fit(self, train_loader, val_loader, optimizer, criterion, epochs=10):
        """
        Trains the Net with the train loader and
        computes the performance metrics after each epoch using
        both train loader and validation loader.

        Parameters
        ----------
        train_loader : `torch.DataLoader`
            The training DataLoader instance
        val_loader : `torch.DataLoader`
            The validation DataLoader instance
        optimizer : `torch.optim.Object`
            The optimizer to tweak the weights and biases
        criterion : `torch.nn.Object`
            The loss function
        epochs : int
            The amount of epochs to train the Net

        Returns
        ----------
        hist : `Dict`
            The dictionary with all epochs' losses and accuracies on
            both training loader and validation loader
        """

        self.hist = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        self.optimizer = optimizer
        self.criterion = criterion

        for epoch in range(epochs):
            for batch_id, (X_train, y_train) in enumerate(train_loader):
                self.zero_grad()
                # Forward pass
                preds = self.forward(X_train)
                loss = self.criterion(preds, y_train)

                # Backward pass
                loss.backward()
                self.optimizer.step()
            # End of train
            # Start evaluating the model's performance
            train_loss, train_acc = self.evaluate(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            self.hist['train_loss'].append((train_loss))
            self.hist['train_acc'].append(train_acc)
            self.hist['val_loss'].append(val_loss)
            self.hist['val_acc'].append(val_acc)

            # Print log
            print(
                f'Epoch {epoch} - Train loss: {train_loss} - Train acc: {train_acc} ---- Validation loss {val_loss} - Validation acc: {val_acc}')

        return self.hist

    def evaluate(self, eval_loader):
        """
        Computes the loss and accuracy of the Net for the given evaluation loader.

        Parameters
        ----------
        eval_loader : `torch.DataLoader`
            The Evaluation DataLoader instance

        Returns
        ----------
        total_loss : float
            The loss value of the Net
        total_acc : float
            The accuracy value of the Net
        """

        right_preds = 0
        total_preds = 0
        total_loss = 0
        for X, y in eval_loader:
            preds = self.forward(X)
            total_loss += self.criterion(preds, y).item()
            preds = torch.round(preds, decimals=0)
            right_preds += (preds == y).sum().item()
            total_preds += len(y)
        return round(total_loss, 6), round(right_preds/total_preds, 6)


# Instantiate the model
model = Net(200)

# Hyperparams
epochs = 50
lr = 3e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Train the model
hist = model.fit(train_loader, val_loader, optimizer, criterion, epochs)

# Save the ouputs to a log file
ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, 'training_logs')
logs_file = os.path.join(PATH, 'training_logs.csv')
model_name = f'model_{datetime.date.today().strftime("%Y%m%d")}'
train_loss, train_acc = model.evaluate(train_loader)
val_loss, val_acc = model.evaluate(val_loader)
model_file = os.path.join(PATH, f'{model_name}.pkl')

df = pd.DataFrame([[model_name, train_loss, train_acc, val_loss, val_acc]],
                  columns=['model_name', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

# Check to see if logs_file exists so not to overwrite data
if not os.path.exists(logs_file):
    mode = 'w'
else:
    mode = 'a'
df.to_csv(logs_file, mode=mode, index=False)

# Save the final model to a pickle file
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
