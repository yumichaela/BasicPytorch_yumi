# 연쇄 법칙 : 마치 오차가 로지스틱 손실 뉴런을 거꾸로 거치면서 > 역전파 되는 듯 한 모양임
# 이진 분류(0 또는 1로만 출력) > 로지스틱 손실 함수를 각 w(가중치), b(절편)에서 미분 하는 것 = 선형 회귀 > 제곱 오차 함수를 미분한 것과 동일한 Form

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# 1. 데이터 파악(상자 수염 그래프)
# print(cancer.data.shape, cancer.target.shape) # data : 2차원 배열 (569개의 데이터, 30개의 특성 / target : 1차원 배열(569개)
cancer.data[:3]

# plt.boxplot(cancer.data)
# plt.xlabel("feature")
# plt.ylabel("value")
# plt.show()

# 2. 타겟 데이터 확인, 훈련 데이터 준비
np.unique(cancer.target, return_counts=True)   # unique() : cancle.target 중 고유한 값만 뽑아서 출력해줌
# (array[0, 1]), array([212, 357]) > 0에 대한 값(음성 클래스):212 / 1에 대한 값(양성 클래스):357
x = cancer.data
y = cancer.target

# 로직스틱 회귀를 위한 뉴런 만들기
# 3. 훈련 테스트와 테스트 세트 나누기 (훈련이 테스트보다 많아야함)
from sklearn.model_selection import train_test_split   # train_test_split(): x를 train과 test로 각각 나누어주는 함수
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)  # stratify=y:양, 음성 클래스의 비율을 동일하게 유지 / test_size = 0.2:테스트 20퍼 정도

# 4. 분할 결과 확인
print(x_train.shape, x_test.shape)
np.unique(y_train, return_counts=True)   # 훈련 세트의 비율이 적절히 잘 나눠져 있는지 확인

# 5. 로지스틱 뉴런 구현하기
class LogisticNeuron:
    
    def __init__(self):
        self.w = None
        self.b = None
        # 가중치, 절편을 미리 초기화 하지 않음 (입력되는 데이터 크기에 맞춰 나중에 초기화 할 것임)

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b   # 정방향 계산 > 넘파이 배열 사용 > 차원은 바뀌지 X, (벡터or행렬 내의)원소끼리 계산만 해줌(for문 필요 X)
        return z
    
    def backdrop(self, x, err):
        w_grad = x * err   # 가중치에 대한 그래디언트 계산
        b_grad = 1 * err   # 절편에 대한 그래디언트 계산
        return w_grad, b_grad
    
# 6. 나머지 메서드 구현하기
    def activation(self, z):
        z = np.clip(z, -100, None)      # 안전한 np.exp()계산을 위해
        a = 1 / (1 + np.exp(-z))        # 시그모이드 계산
        return a

    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])    # 가중치 초기화 / np.ones : 모든 배열을 1로 채움
        self.b = 0                      # 절편 초기화
        for i in range(epochs):         # epochs만큼 반복
            for x_i, y_i in zip(x, y):  # 모든 샘플에 대해 반복
                z = self.forpass(x_i)   # 정방향 계산
                a = self.activation(z)  # 활성화 함수 적용 = 시그모이드 함수 통과함 / z라는 선형 방적식의 결과를 사용함(위에 있음)0
                err = -(y_i - a)        # 오차 계산
                w_grad, b_grad = self.backdrop(x_i, err)   # 역방향 계산
                self.w -= w_grad        # 가중치 업데이트 
                self.b -= b_grad        # 절편 업데이트

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]   # 선형 함수 적용
        # 새로운 'x'라는 데이터가 들어오면 > x에 있는 데이터를 하나씩 꺼내서 x_i에 집어넣음 > 얘네를 forpass 매소드에 다 통과시킴 > 이 결과를 리스트에 다 채워서 리스트에 만듬 > 결론:리스트 값을 다 꺼내서, 리스트 결과로 만듬
        a = self.activation(np.array(z))       # 활성화 함수 적용(시그모이드 함수) / 위에서 z를 넘길때 numpy 배열 사용 했으므로, 넘파이 배열로 바꿔서 a 값에 집어넣음
        return a > 0.5                         # 계단 함수 적용 = 0.5보다 크면 'True' > 양성으로 나옴



# 7. 모델 훈련 and 결과 확인하기
neuron = LogisticNeuron()
neuron.fit(x_train, y_train)

print(np.mean(neuron.predict(x_test) == y_test))   # 예측값 vs 결과를 True, Flase 형태로 비교함(각 위치에 있는 원소 값이 같은지) / True=1, Flase=0임 > 모두 더해서 > 원소 갯수로 나누면? > 맞춘 확률이 됨








# Ver.2_손실 함수 결괏값 저장 기능 추가하기(결과값을 보는데 시간 넘 오래걸릴 때 > 중간 중간 loss값이 줄어드는지 확인하면 좋음)

class SingleLayer:

    def __init__(self):
        self.w = None
        self.b = None
        # 가중치, 절편을 미리 초기화 하지 않음 (입력되는 데이터 크기에 맞춰 나중에 초기화 할 것임)
        self.losses = []   # **Ver.2** 중간 중간 손실 함수가 잘 저장 되는지 저장하는 List를 만듬

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b
        return z
    
    def backdrop(self, x, err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def activation(self, z):
        z = np.clip(z, -100, None)
        a = 1 / (1 + np.exp(-z))
        return a

    def fit(self, x, y, epochs=100):                                                    
        self.w = np.ones(x.shape[1])    # 가중치 초기화 / np.ones : 모든 배열을 1로 채움
        self.b = 0                      # 절편 초기화
        for i in range(epochs):         # epochs만큼 반복
            loss = 0
            indexes = np.random.permutation(np.len(x))   # **Ver.2**인덱스를 섞음 (확률적 경사하강법이므로, 1개의 샘플을 무작위로 뽑아 쓰기 때문.)
            for i in indexes:  # 모든 샘플에 대해 반복
                z = self.forpass(x[i])   # 정방향 계산
                a = self.activation(z)  # 활성화 함수 적용 = 시그모이드 함수 통과함 / z라는 선형 방적식의 결과를 사용함(위에 있음)0
                err = -(y[i] - a)        # 오차 계산
                w_grad, b_grad = self.backdrop(x[i], err)   # 역방향 계산
                self.w -= w_grad        # 가중치 업데이트 
                self.b -= b_grad        # 절편 업데이트 
                a = np.clip(a, 1e-10, 1-1e-10)   # 넘파이 클립 함수 : 안전한 로그 계산을 위해 클리핑한 후 손실을 누적함 (***log함수에서 [0일때 : -무한대 / 1일때 : 0]이기 때문에, 1의 -10승(1e-10 = 0에 가까움), 1 빼기 1의 -10승(1에 가까움))
                loss += -(y[i] * np.log(a) + (1 - y[i] * np.log(1-a)))

            self.losses.append(loss/len(y))

    def predict(self, x):
        z = [self.forpass(x_i) in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
    
    layer = SingleLayer()
    layer.fit(x_train, y_train)
    layer.score(x_test, y_test)

    plt.plot(layer.losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    