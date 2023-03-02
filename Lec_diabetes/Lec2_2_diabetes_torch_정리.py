import torch 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# 이전 수업에서 했던 그림 그리는 것을 함수로 만들어 사용합니다. 
def plot(x, y, w, b, step, min_val=-0.1, max_val=0.15, name="plot"):
    plt.scatter(x, y) 
    plt.xlabel("x")
    plt.ylabel("y")
    point1 = (min_val, min_val * w.data.numpy() + b.data.numpy()) # x값이 적당히 작을때 직선 위의 점
    point2 = (max_val, max_val * w.data.numpy() + b.data.numpy()) # x값이 적당히 클때 직선 위의 점
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r')
    plt.grid() 
    plt.savefig(name+"{}.png".format(step)) # 초기값일 때의 직선을 그립니다. 
    plt.close() 

torch.manual_seed(777) # cpu 연산 무작위 고정
np.random.seed(777) # numpy 연산 무작위 고정 (시드가 무작위로 변한다면 학습 모델의 결과를 확인할 때 무엇이 문제인지 판별하기 어려워짐)

# Step 1 데이터 준비
diabetes = load_diabetes()
x = diabetes.data[:, 2]
y = diabetes.target

xt = torch.FloatTensor(x)  # numpy array를 torch의 tensor로 변환함
yt = torch.FloatTensor(y)

w = torch.ones([1], requires_grad=True) # requires_grad=True : 해당 텐서를 기준으로 모든 연산들을 추적하여 Gradient라고 하는 미분값의 모임(배열)을 계산할 수 있게 함.
b = torch.ones([1], requires_grad=True)

optimizer = torch.optim.SGD([w, b], lr=1e-2) # lr : "Learning Rate"(미분값을 얼마나 이동시킬 것인가 = 하강에 대한 보폭을 정해주는 값)
# SGD : 확률적 경사하강법 : (데이터 전체를 한 번에 사용하는게 아니라) 나눠서 사용함(데이터가 많을 경우 연산량이 많아 모델의 학습 속도가 느려짐 or 메모리 부족 문제 보완 목적)
loss_fn = torch.nn.MSELoss() # mean squared error를 구하는 loss function. 

# 1회 어떻게 작동하는지 체크하는 코드입니다. 
# y_hat = xt[0] * w + b 
# loss = loss_fn(y_hat, yt[0])

# optimizer.zero_grad() # gradients값들을 추후에 backward를 해줄때 계속 더해주기 때문에, 항상 zero로 만들고 시작해야함. 안하면, 의도한 방향과 다른 결과가 나옴. 
# + (기본적으로 변화도는 더해지기 때문에, 중복 계산을 막기 위해 반복할 때마다 명시적으로 0을 설정함.)
# loss.backward() # <얘를 호출해서, 예측 손실(prediction loss)을 역전파한다. Pytorch는 각 매개변수에 대한 변화도를 저장함.
# optimizer.step() # 위에서 수집한 변화도로 매개변수를 조정함.

# 모든 데이터 셋에 대해서 해봅시다. 
cnt = 0 # 해당 문자가 몇 번 반복했는지 알 수 있는 카운트 변수
idx = [i for i in range(len(x))]  # 이 문법에 대해서 찾아보세요!

"""
idx = [i for i in range(len(x))] 
위 문장은 아래 for문과 같은 의미입니다 
idx = list() 
for i in range(len(x)):
    idx.append(i)
"""
for k in range(100):
    idx = np.random.permutation(idx)  
    # permutaion은 안에 순서를 바꾸어줍니다. 
    epoch_loss = 0
    for i in idx:
        """ 
        idx의 순서가 바뀌어 들어간다는 것은 epoch마다 학습에 사용되는 샘플의 순서가 바뀐다는 뜻입니다. 
        """
        y_hat = xt[i] * w + b
        loss = loss_fn(y_hat, yt[i].unsqueeze(0))

        epoch_loss += loss.data.numpy()
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
    
    epoch_loss = epoch_loss / len(idx)
    print("Epoch : {}, Loss : {}".format(k, epoch_loss))
        
    if k % 20 == 0:
        plot(x, y, w, b, cnt, name="lec2_2_")
        cnt += 1


