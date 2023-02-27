import torch
from matplotlib import pyplot as plt

x = torch.FloatTensor(range(5).unsqueeze(1))
y = 2 * x + torch.rand(5, 1)
num_features = x.shape[1]

w = torch.randn(num_features, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 1e-3