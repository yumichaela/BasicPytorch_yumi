import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

x = diabetes.data[:, 2]
y = diabetes.target

# plt.scatter(x, y)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid()
# plt.savefig("lec2_1_1.png")
# plt.close()

w = 1.0
b = 1.0
# 1. 무작위로 w와 b를 정함

y_hat = x[0]* w + b
# 2. x에서 샘플 하나를 정하여 y_hat을 계산함

err = y[0] - y_hat
# 3. y_hat과 선택한 샘플의 진짜 y를 비교함

w = w + x[0] * err
b = b + 1 * err
# 4. y_hat과 y가 더 가까워지도록 w, b를 조정함


# 5. 반복함!

for k in range(100):
    for i in range(len(x)):
        y_hat = x[i]* w + b
        err = y[i] - y_hat
        w = w + x[i] * err
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