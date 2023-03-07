### 1. Iris 데이터 준비
# 라이브러리와 iris 데이터 불러오기
import torch
from torch import nn, optim
from sklearn.datasets import load_iris
iris = load_iris()

# iris는 (0,1,2)의 세 가지 종류를 분류하는 문제이므로 (0,1)의 두 개의 데이터만 사용한다
# ???? 학습용, 테스트용 나누기 ?????
x = iris.data[:100]
y = iris.target[:100]

# Numpy의 ndarray를 Pytorch의 Tensor로 변환
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


### 2. 모델 작성
# iris 데이터는 4차원
net = nn.Linear(4, 1)

# 시그모이드 함수를 적용해서 두 클래스의 분류를 위한 크로스 엔트로피를 계산
loss_fn = nn.BCEWithLogitsLoss()

# SGD(약간 큰 학습률)
optimizer = optim.SGD(net.parameters(), lr=0.25)


### 3. 파라미터 최적화를 위한 반복 루프
losses = []
for epoch in range(100):
    optimizer.zero_grad()               # 전 회의 backward매서드로 계산된 경사값을 초기화
    y_pred = net(x)                     # 선형 모델로 y 예측값을 계산
    loss = loss_fn(y_pred.view_as(y),y) # MSE loss를 사용한 미분값 계산
    loss.backward()
    optimizer.step()                      # 경사를 갱신
    losses.append(loss.item())          # 수렴 확인을 위한 loss를 기록
    print(loss.item())

from matplotlib import pyplot as plt
plt.plot(losses)
plt.show()


### 4.모델 작성?????????????
h = net(x)
prob = nn.functional.sigmoid(h)
y_pred = prob > 0.5 
(y.byte() == y_pred.view_as(y)).sum().item()