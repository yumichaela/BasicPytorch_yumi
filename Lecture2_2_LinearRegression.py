import torch
import numpy as np 
import matplotlib.pyplot as plt 

torch.manual_seed(777)
np.random.seed(777)

x = np.random.randn(100)
w = 2.0
b = 1.0 
y = x * 3.0 + 1.0 +  np.random.normal(0.0, 2, size=100)

# print(np.min(x))
# print(np.max(x))

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.grid() 
plt.savefig("lec2_2_1.png") # 초기값일 때의 직선을 그립니다. 
plt.close() 

# Step 1. 데이터 준비하기 
xt = torch.FloatTensor(x)
yt = torch.FloatTensor(y)
print(xt.size())
print(yt.size())

xt = xt.unsqueeze(1) # pytorch에 사용하기 위해서는 2차원으로 변환이 필요합니다. 
yt = yt.unsqueeze(1)

print(xt.size())
print(yt.size())

wt = torch.ones(1, requires_grad=True) 
bt = torch.zeros(1, requires_grad=True)

optim = torch.optim.SGD([wt, bt], lr=0.0001) 

y_hat = xt[0] * wt + bt
err = torch.mean((y_hat - yt[0]) ** 2)

optim.zero_grad() 
err.backward() 
optim.step() 

total_epochs = 31  #모든 데이터를 한 번 다 보는 주기를 epoch
cnt = 2
for epoch in range(total_epochs):
    for i in range(len(x)):
        y_hat = xt[i] * wt + bt
        err = torch.mean((y_hat - yt[i]) ** 2)
        
        optim.zero_grad() 
        err.backward() 
        optim.step() 
    
    print("Epoch : {}, Error : {}".format(epoch, err))

    if epoch % 10 == 0:
        plt.scatter(x, y) 
        plt.xlabel("x")
        plt.ylabel("y")
        point1 = (-4, -4 * wt.item() + bt.item()) # x값이 적당히 작을때 직선 위의 점
        point2 = (4, 4 * wt.item() + bt.item()) # x값이 적당히 클때 직선 위의 점
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r', label="Predict")
        point1 = (-4, -4 * w + b) # x값이 적당히 작을때 직선 위의 점
        point2 = (4, 4 * w + b) # x값이 적당히 클때 직선 위의 점
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'g', label="True")
        plt.grid() 
        plt.legend()
        plt.title("Epoch : {}".format(epoch))
        plt.savefig("lec2_2_{}.png".format(cnt)) 
        plt.close() 
        cnt += 1
