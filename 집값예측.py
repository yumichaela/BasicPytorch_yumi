import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('./data/reg.csv', index_col=[0])

X = df.drop('Price', axis=1).to_numpy()
Y = df['price'].to_numpy().reshape((-1,1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self)
        return self.len

trainsets = TensorData(X_train, Y_train)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=32, suffle=True)
testsets = TensorData(X_test, Y_test)
testloader = torch.utils.data.DataLoader(trainsets, batch_size=32, suffle=False)

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 50, bias=True)
