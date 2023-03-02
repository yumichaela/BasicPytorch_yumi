import torch
from matplotlib import pyplot as plt

# (이해를 돕기 위해)일변수 데이터를 생성하여 이용함. 즉 (x, y) 형태로 2차원 상에서 표현 가능한 데이터임
x = torch.FloatTensor(range(5)).unsqueeze(1) # 리스트 range(5)를 이용해 텐서로 만듬 / 원래 크기는 1차원인 torch.Size(5)라서 행렬 계산을 위해 2차원 배열로 만들어줌.
                                             # unsqueeze(1) : 1번째 위치의 차원을 늘려주는 역할 > 최종 x의 크기는 > torch.Size(5, 1)이 됨 / cf. (만약 unsqueeze(0)이면? > torch.Size(1, 5)가 됨)
y = 2 * x + torch.rand(5, 1) # y는 실제 값으로 임의로 5개 만들어줌
num_features = x.shape[1] # 변수의 갯수를 저장함 / x의 크기가 torch.Size([5,1])이므로 > 인스턴스 개수가 5개 > 변수(피쳐) 개수가 1개 > 따라서 x.shape[1]은 변수의 개수임

w = torch.randn(num_features, 1, requires_grad = True) # torch.randn : 모르는 초깃값을 무작위로 줌
b = torch.randn(1, requires_grad = True) # w, b값은 역전파를 통해 최적값을 찾는 것이므로 > w, b에 requires_grad를 True로 활성화시킴

# 가중치 업데이트
learning_rate = 1e-3
optimizer = torch.optim.SGD([w, b], lr=learning_rate) # torch.optim.SGD 내부에 변수를 리스트로 묶고 > 적절한 학습률을 정해줌

loss_stack = [] # 매 epoch마다 손실함수값을 저장하기 위해 빈 리스트를 생성함
for epoch in range(1001):
    optimizer.zero_grad() # 최적화는 계산을 누적시키기 때문에 > 매 에폭마다 누적된 값을 이 함수르 초기화함
    y_hat = torch.matmul(x, w) + b # 회귀식 모델을 이용하여 예측값 산출
    loss = torch.mean((y_hat - y)**2) # 예측값, 실제값을 이용하여 손실함수 계산 > Mean Squared Error 함수 사용함 (그래서 **2 제곱합)
    loss.backward() # 역전파의 기준을 손실함수로 정함
    optimizer.step() # 미리 정의한 optimizer를 이용하여 최적화 시행함
    loss_stack.append(loss.item()) # 그래프를 그리기 위해 > 손실 함수값만 loss_stack에 하나씩 넣음 > item()사용 하지 않으면 > loss 텐서 전체를 저장하게 됨

    if epoch % 100 == 0:
        print(f'Epoch {epoch}:{loss.item()}') # 에폭이 100으로 나눠 떨어질때마다 손실 함수값 저장함

with torch.no_grad(): # 최적화를 사용하지 않음 > required_grad 비활성화함
    y_hat = torch.matmul(x, w) + b

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(loss_stack)
plt.title("Loss")
plt.subplot(122)
plt.plot(x, y, '.b')
plt.plot(x, y_hat, 'r-')
plt.legend(['ground truth', 'predictoin'])
plt.title("Prediction")
plt.show()
plt.close()


