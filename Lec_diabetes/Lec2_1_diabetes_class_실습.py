import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

diabetes_data = diabetes.data[:, 2]
diabetes_target = diabetes.target

np.random.seed(777)

class LinearRegression:
    def __init__(self):
        self.w = 1.0
        self.b = 1.0
    
    def forward(self, x):
        y_hat = x * self.w + self.b
        return y_hat

    def cal_error(self, y, y_hat):
        err = y - y_hat
        return err

    def train(self, x, y, total_epochs):
        for k in range(total_epochs):
            for i in range(len(x)):
                y_hat = self.forward(x[i])
                err = self.cal_error(y[i], y_hat) #backward 과정
                self.w = self.w + x[i] * err # step 과정
                self.b = self.b + 1 * err 
                    

plt.scatter(diabetes_data, diabetes_target)
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.savefig("lec2_2_1.png")
plt.close()

model = LinearRegression()
model.train(diabetes_data, diabetes_target, 100)
        


plt.scatter(diabetes_data, diabetes_target)
plt.xlabel("x")
plt.ylabel("y")
point1 = (-0.1, -0.1 * model.w + model.b) # x값이 적당히 작을때 직선 위의 점
point2 = (0.15, 0.15 * model.w + model.b) # x값이 적당히 클때 직선 위의 점
plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r')
plt.grid() 
plt.savefig("lec2_2_2.png") # 초기값일 때의 직선을 그립니다. 
plt.close()
