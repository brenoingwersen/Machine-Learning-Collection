# Santander Customer Transaction Prediction :credit_card:
***Kaggle Competition*** :trophy:

## Project introduction
This is a <a href="https://www.kaggle.com/c/santander-customer-transaction-prediction">Kaggle Competition</a> challenge released by Santander to identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. 

According to the company the data provided for this competition has the same structure as the real data we have available to solve this problem.

## About the dataset
There's not much to be said about the dataset. This is a binary classification task to predict 1 for a true positive and 0 for a true negative label. The data has 200 features and is devided into a a train set and test set. The features are named "var_i" whereas i represents the order number of the ith variable. The company opted not to reveal business information regarding the features nor the labels. Finally, the test set is a .csv file with only features and instances to post the final predictions to the Kaggle's challenge page to obtain the final model's generalization score.

## Proposed solution

Apply Deeplearning to create a DNN (Deep Neural Net) using PyTorch. 

PyTorch is a Python package that provides two high-level features:

- Tensor computation (like NumPy) with strong GPU acceleration
- Deep neural networks built on a tape-based autograd system

## Problem approach

1. Gather the data and run an EDA (Exploratory Data Analysis) on a Notebook file to inspect for any preprocessing needs (missing data, feature engineering, etc.), outliers detection and to create some plots and check for feature to feature and feature to target correlations.
2. Build a baseline DNN using PyTorch. This baseline net will provide us with the minimum values of the performance metrics once I start tweaking the DNN to add more layers, change neuron count, add batchnorm, change the optimizer, etc. So this will be a really simple DNN.
3. Tweak the DNN.
4. TBD

## Results
| Model | Accuracy |
| --- | --- |
| Baseline model | 0.678 |
