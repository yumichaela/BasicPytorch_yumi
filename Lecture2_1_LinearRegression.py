import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_diabetes 

# Step 1. 데이터 셋을 준비합니다. 
# Step 1.1. diabetes 데이터 셋을 불러옵니다. 
diabetes = load_diabetes() 


# Step 1.2. diabetes 데이터 셋을 확인합니다. 
'''
print("diabetes 특성 데이터의 모양 : ", diabetes.data.shape)
print("diabetes 타겟 데이터의 모양 : ", diabetes.target.shape)
# diabetes 특성 데이터의 크기 :  (442, 10) -> 442개의 샘플이 있고 10개의 특징이 있다는 뜻입니다. 
# diabetes 타겟 데이터의 크기 :  (442,) -> 442개의 샘플이 있고 타겟이 1차원이라는 뜻입니다. 

print(diabetes.data[0:3])  # 0번째에서 2번째까지 총 3개의 샘플을 확인합니다. 
# diabetes.data[0:3]의 크기 : (3, 10)
# diabetes.data[0:3]과 diabetes.data[:3], diabetes.data[0:3, :], diabetes.data[:3, :]은 동일합니다. 
print(diabetes.target[:3])
'''

# Step 1.3. diabetes 데이터 셋의 일부만 취합니다. 
x = diabetes.data[:, 2]  # 3번째 feature를 취합니다. 
y = diabetes.target 

print(np.min(x))  # x 값의 최솟값은 -0.0902752958985185
print(np.max(x))  # x 값의 최댓값은 0.17055522598066

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.grid() 
# plt.show()
plt.savefig("lec2_1_1.png")
plt.close() 

# Step 2. weight와 bias를 초기화 합니다. 
w = 1.0 
b = 1.0 

plt.scatter(x, y) 
plt.xlabel("x")
plt.ylabel("y")
point1 = (-0.1, -0.1 * w + b) # x값이 적당히 작을때 직선 위의 점
point2 = (0.15, 0.15 * w + b) # x값이 적당히 클때 직선 위의 점
plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r')
plt.grid() 
plt.savefig("lec2_1_2.png") # 초기값일 때의 직선을 그립니다. 
plt.close() 

# Step 3. 첫 번째 예측과 오차 역전법을 이용한 첫 번째 업데이트 입니다. 
# 어떤 식으로 업데이트 하는지 확인하기 위해 한 번을 수행해봅니다. 
y_hat = x[0] * w + b 
err = y[0] - y_hat 
w_rate = x[0]
w_new = w + w_rate * err 
b_new = b + 1 * err 

total_epochs = 100  #모든 데이터를 한 번 다 보는 주기를 epoch
for epoch in range(total_epochs):
    for i in range(len(x)):
        y_hat = x[i] * w + b 
        err = y[i] - y_hat
        w_rate = x[i]
        w = w + w_rate * err 
        b = b + 1 * err 
        
plt.scatter(x, y) 
plt.xlabel("x")
plt.ylabel("y")
point1 = (-0.1, -0.1 * w + b) 
point2 = (0.15, 0.15 * w + b) 
plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r')
plt.grid() 
plt.savefig("lec2_1_3.png") 
plt.close() 
