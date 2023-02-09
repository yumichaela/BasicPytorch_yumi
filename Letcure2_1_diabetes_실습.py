import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

# print("diabetes 특성 데이터의 모양 : ", diabetes.data.shape)
# print("diabetes 타겟 데이터의 모양 : ", diabetes.target.shape)


x = diabetes.data[:, 2]
y = diabetes.target

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.savefig("lec2_1_1.png")
plt.close()

w = 1.0
b = 1.0

y_hat = x[0] * w + b

# print(y_hat)
# print(y[0])


err = y[0] - y_hat
w = w + x[0] * err
b = b + 1 * err

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
point1 = (-0.1, -0.1 * w + b)
point2 = (0.15, 0.15 * w + b)
plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'or')
plt.grid()
plt.savefig("lec2_1_2.png")
plt.close()
