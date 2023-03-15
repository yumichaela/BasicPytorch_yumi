# 1. 데이터셋 불러오기
from sklearn.datasets import load_iris
iris = load_iris()

print(iris.data.shape, iris.target.shape)
print(iris.feature_names)

sepal_length = iris.data[:, 0]
sepal_width = iris.data[:, 1]
petal_length = iris.data[:, 2]
petal_width = iris.data[:, 3]

import matplotlib.pyplot as plt

plt.scatter(sepal_length, petal_length, marker="o", color='b')
plt.grid()
plt.xlabel("sepal_length")
plt.ylabel("petal_length")
plt.title("'sepal_length - petal_length' in IRIS dataset")
plt.show()

plt.scatter(sepal_width, petal_width, marker="o", color='r')
plt.grid()
plt.xlabel("sepal_width")
plt.ylabel("petal_width")
plt.title("'sepal_width - petal_width' in IRIS dataset")
plt.show()

# 2. 오차 역전파를 이용한 가중치, 절편 업데이트
w = 1
b = 1
x = sepal_length 
y = petal_length

for x_i, y_i in zip(x, y):
  y_hat = x_i * w + b
  err = y_i - y_hat
  w_rate = x_i 
  w = w + w_rate * err * 0.001
  b = b + 1 * err * 0.001

plt.scatter(x, y)
pt1 = (4.0, 4.0 * w + b)
pt2 = (8.0, 8.0 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()


w = 1   # 여러 에포크를 반복하기
b = 1

x = sepal_length 
y = petal_length

for i in range(1000):
  for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i 
    w = w + w_rate * err * 0.001
    b = b + 1 * err * 0.001

print(w, b)
plt.scatter(x, y)
pt1 = (4.0, 4.0 * w + b)
pt2 = (8.0, 8.0 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

# 공식을 이용한 정답
import numpy as np

x = sepal_length
y = petal_length

x_bar = np.mean(x)
y_bar = np.mean(y)
s_xx = np.std(x)
s_xy = np.cov(x, y)[0][1]

w = s_xy/s_xx
b = y_bar - w + x_bar

print(w, b)

plt.scatter(x, y)
pt1 = (4.0, 4.0 * w + b)
pt2 = (8.0, 8.0 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
# 여기까지 위와 똑같은 결과임

class Neuron:
    def __init__(self):
        self.w = 1.0
        self.b = 1.0

    def forpass(self, x):
        y_hat = x * self.w + self.b
        return y_hat
       
    def backdrop(self, x, err, lr=1e-2):
        w_grad = x * err * lr
        b_grad = 1 * err * lr
        return w_grad, b_grad
    
    def fit(self, x, y, epochs=1000):
        for i in range(epochs):
            for x_i, y_i in zip(x, y):
            y_hat = self.forpass(x_i)
            err = -(y_i - y_hat)
            w_grad, b_grad = self.backdrop(x_i, err)
            self.w -= w_grad
            self.b -= b_grad

x = sepal_length
y = petal_length

m = Neuron()
m.fit(x, y, 10000)

plt.scatter(x, y)
pt1 = (4.0, 4.0 * w + b)
pt2 = (8.0, 8.0 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

    
