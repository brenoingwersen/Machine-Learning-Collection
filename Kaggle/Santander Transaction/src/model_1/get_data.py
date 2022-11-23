import sys
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

# Creating new features


class Pipe():
    """
    A base class to add new features.

    Features added:
    - var_i_u = int
        Boolean column with a flag 1 for a unique value for the `var_i` variable and 0 if not.

    """

    def __init__(self):
        self.is_fitted = False

    def fit(self, df):
        """
        Fits the data and creates a dict obj with the unique values per feature.
        """
        self.uniques = {}
        for col in df.drop(columns=['ID_code', 'target'], errors='ignore').columns:
            # Store the unique values per col
            self.uniques[col] = df[col].unique()

        self.is_fitted = True

    def transform(self, df):
        """
        Applies the transformations.
        """
        # Check wether the fit method has already been called
        assert self.is_fitted == True, 'Call the fit or fit_transform method to fit the data'

        # Copy the dataframe to add the new features
        new_df = df.copy(deep=True)
        new_df.drop(columns=['ID_code', 'target'],
                    errors='ignore', inplace=True)

        # Loop through the columns
        for col, uniques in self.uniques.items():
            # Create a flag col for unique values
            new_col = f'{col}_u'
            new_df[col] = new_df[col].map(lambda x: 1 if x in uniques else 0)
            new_df.rename(columns={col: new_col})

        # Merge the original features to the created ones
        return pd.concat([df, new_df], axis=1)

    def fit_transform(self, df):
        # Fits the data
        self.fit(df)

        # Transforms the data
        return self.transform(df)


# Load the training data
train_file = os.path.join(PATH, 'train.csv')
test_file = os.path.join(PATH, 'test.csv')
train_df_full = pd.read_csv(train_file, nrows=1e4)
test_df = pd.read_csv(test_file)

# Combine all data
all_data_df = pd.concat([train_df_full, test_df], axis=0, ignore_index=True)

# Get the original feature columns to apply normalization
feature_cols = all_data_df.drop(columns=['ID_code', 'target']).columns

# Apply the feature creation
# Instantiate the pipe
pipe = Pipe()
all_data_df = pipe.fit_transform(all_data_df)

# Re-split into train and test
mask = all_data_df['ID_code'].isin(train_df_full['ID_code'])
train_df_full = all_data_df.loc[mask]
test_df = all_data_df.loc[~mask]

# Spliting the train set into train and validation
train_df, val_df = train_test_split(train_df_full, test_size=0.2)

# Instantiate the StandardScaler
scaler = StandardScaler()

# Apply standardization
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

# Creates the training loader instance
train_set = Dataset(train_df)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# Apply the above steps to the validation set
val_df[feature_cols] = scaler.transform(val_df[feature_cols])
val_set = Dataset(val_df)
val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

# Apply the above steps to the test set
test_df[feature_cols] = scaler.transform(
    test_df[feature_cols])  # Scale the original features
test_set = Dataset(test_df)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
