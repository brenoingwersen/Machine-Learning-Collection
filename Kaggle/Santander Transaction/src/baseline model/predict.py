import os
import pandas as pd
import torch
from get_data import test_loader
from baseline_net import Net

ROOT = os.path.abspath(__file__)
for i in range(3):
    ROOT = os.path.dirname(ROOT)
PATH = os.path.join(ROOT, 'best_models')

# Load the test.csv file to get the ID_codes
TEST_PATH = os.path.join(ROOT, 'data')
TEST_FILE = os.path.join(TEST_PATH, 'test.csv')
test_df = pd.read_csv(TEST_FILE, usecols=['ID_code'])

# Load the checkpoint dict
checkpoint = torch.load(os.path.join(
    PATH, 'baseline_model_20221117_214149_100'))

# Instantiate the model
model = Net(200)

# Load the model's weights and biases
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model mode to eval only
model.train(False)

# Loop through the test_loader
for X_test in test_loader:
    preds = model.forward(X_test)

# Convert the probabilities to binary classes
preds = preds.to(torch.int8).view((-1, 1)).detach().numpy()

# Add the predictions to the test set
test_df['target'] = preds

# Save predictions
PRED_PATH = os.path.join(ROOT, 'predictions')
PRED_FILE = os.path.join(PRED_PATH, 'predictions_baseline_model.csv')
test_df.to_csv(PRED_FILE, index=False)
