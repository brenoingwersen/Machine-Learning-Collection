import os
import sys
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from baseline_net import Net
from get_data import train_loader, val_loader, test_loader

TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
ROOT = os.path.abspath(__file__)
for i in range(3):
    ROOT = os.path.dirname()
PATH = os.path.join(ROOT, 'best_models')

# Instantiate the model
model = Net(200)

# Hyperparams
epochs = 100
lr = 3e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
patience = 10  # Amount of epochs to get a better model performance on the validation set

# Train one epoch function


def train_one_epoch(epoch, writer):
    running_loss = 0
    n_instances = 0
    n_correct = 0  # The number of correctly guessed instances to compute the accuracy

    for i, data in enumerate(train_loader):
        inputs, labels = data

        # Zero gradients
        optimizer.zero_grad()

        # Make predictions
        outputs = model.forward(inputs)

        # Compute the loss and gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjust weights and biases
        optimizer.step()

        # Gather data
        running_loss += loss.item()
        n_instances += len(inputs)
        n_correct += (torch.round(outputs, decimals=0) == labels).sum().item()

    # Report
    avg_loss = round(running_loss/n_instances, 6)
    avg_acc = round(n_correct/n_instances, 6)
    writer.add_scalar('Train Loss/Epoch', avg_loss, epoch)
    writer.add_scalar('Train Acc/Epoch', avg_acc, epoch)
    return avg_loss, avg_acc, writer


# The training loop
writer = SummaryWriter(f'{PATH}/baseline_model_{TIMESTAMP}')
best_avg_loss = 1e6
n_patience = 0
last_val_loss = 1e6

for epoch in range(1, epochs + 1):
    model.train(True)
    avg_train_loss, avg_train_acc, writer = train_one_epoch(epoch, writer)

    # Disble learning
    model.train(False)

    # Eval the model's performance on the validation set
    running_loss = 0
    n_instances = 0
    n_correct = 0
    for data in val_loader:
        inputs, labels = data

        # Forwards the data through the network
        outputs = model.forward(inputs)

        # Compute the loss and size of training instances
        running_loss += criterion(outputs, labels).item()
        n_instances += len(inputs)
        n_correct += (torch.round(outputs, decimals=0) == labels).sum().item()

    # Compute the mean loss and accuracy per instance
    avg_loss = round(running_loss/n_instances, 6)
    avg_acc = round(n_correct/n_instances, 6)

    # Report
    print(
        f'Epoch{epoch}: Train Loss {avg_train_loss} Train Acc: {avg_train_acc} - Val Loss {avg_loss} Val Acc: {avg_acc}')

    # Save the file to disk
    writer.add_scalar('Validation Loss/Epoch', avg_loss, epoch)
    writer.add_scalar('Train Acc/Epoch', avg_acc, epoch)
    writer.flush()

    # Apply early stopping
    # If the loss on the validation set does not decrease for a patience
    # amount of epochs, then shutdown the training process
    if last_val_loss <= avg_loss:
        n_patience += 1
        if n_patience + 1 == patience:
            break  # Terminates the training loop
    else:
        n_patience = 0

    # Keep the actual model as best model params in case early stopping is activated
    best_model_params = model.state_dict()
    best_optim_params = optimizer.state_dict()

# Save the best model after training is complete
MODEL_PATH = os.path.join(PATH, f'baseline_model_{TIMESTAMP}_{epoch}')
torch.save({
    'epoch': epoch,
    'model_state_dict': best_model_params,
    'optimizer_state_dict': best_optim_params,
    'loss': avg_train_loss},
    MODEL_PATH)
