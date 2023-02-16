import torch
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 

torch.manual_seed(777)
np.random.seed(777)

# 데이터 셋 준비 
x = np.random.randn(100)
w = 2.0
b = 1.0 
y = x * 3.0 + 1.0 +  np.random.normal(0.0, 2, size=100)

xt = torch.FloatTensor(x)
yt = torch.FloatTensor(y)

xt = xt.unsqueeze(1)
yt = yt.unsqueeze(1)

class LinearRegression(nn.Module):
    def __init__(self):
        self.w = torch.ones(1, requires_grad=True) 
        self.b = torch.zeros(1, requires_grad=True)
        self.optim = torch.optim.SGD([self.w, self.b], lr=0.0001) 
    
    def forward(self, x):
        y_hat = x * self.w + self.b 
        return y_hat
        
    def train(self, x, y, total_epochs):
        for epoch in range(total_epochs):
            for i in range(len(x)):
                y_hat = x[i] * self.w + self.b
                err = torch.mean((y_hat - y[i]) ** 2)
                
                self.optim.zero_grad() 
                err.backward() 
                self.optim.step() 
                
            print("Epoch : {}, Error : {}".format(epoch, err))

model = LinearRegression() 

model.train(xt, yt, 31)
plt.scatter(x, y) 
plt.xlabel("x")
plt.ylabel("y")
point1 = (-4, -4 * model.w.item() + model.b.item()) # x값이 적당히 작을때 직선 위의 점
point2 = (4, 4 * model.w.item() + model.b.item()) # x값이 적당히 클때 직선 위의 점
plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r', label="Predict")
point1 = (-4, -4 * w + b) # x값이 적당히 작을때 직선 위의 점
point2 = (4, 4 * w + b) # x값이 적당히 클때 직선 위의 점
plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'g', label="true")
plt.legend()
plt.grid() 
plt.savefig("lec2_3.png") 
plt.close() 
                        
     