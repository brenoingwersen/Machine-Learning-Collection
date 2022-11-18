import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

BATCH_SIZE = 32  # Batch size of the train set
ROOT = os.path.abspath(__file__)
for i in range(3):
    ROOT = os.path.dirname(ROOT)
PATH = os.path.join(ROOT, 'data')

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.X = data.drop(
            columns=['ID_code', 'target'], errors='ignore').values
        self.X = torch.tensor(self.X, dtype=torch.float32)

        self.has_target = False
        if 'target' in data.columns:
            self.has_target = True
            self.y = data['target'].values
            self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.has_target:
            return self.X[index], self.y[index]
        return self.X[index]


train_file = os.path.join(PATH, 'train.csv')
train_df_full = pd.read_csv(train_file)

# Spliting the train set into train and validation
train_df, val_df = train_test_split(train_df_full, test_size=0.2)

# Applying standardization to the training set
# This sets the mean to 0 and std to 1 for all features in the dataset
feature_cols = train_df.drop(columns=['ID_code', 'target']).columns
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
train_set = Dataset(train_df)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# Normalizing the validation
val_df[feature_cols] = scaler.transform(val_df[feature_cols])
val_set = Dataset(val_df)
val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

# Same steps for the test set
test_file = os.path.join(PATH, 'test.csv')
test_df = pd.read_csv(test_file)
test_df[feature_cols] = scaler.transform(test_df[feature_cols])
test_set = Dataset(test_df)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
