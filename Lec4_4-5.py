
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


plt.boxplot(cancer.data)
plt.xlabel("feature")
plt.ylabel("value")
plt.show()


# 타겟 데이터 확인, 훈련 데이터 준비
np.unique(cancer.target, return_counts=True)
x = cancer.data
y = cancer.target


# 로직스틱 회귀를 위한 뉴런 만들기
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42) 

print(x_train.shape, x_test.shape)
np.unique(y_train, return_counts=True) 


# class LogisticNeuron:
    
#     def __init__(self):
#         self.w = None
#         self.b = None

#     def forpass(self, x):
#         z = np.sum(x * self.w) + self.b 
#         return z
    
#     def backdrop(self, x, err):
#         w_grad = x * err  
#         b_grad = 1 * err  
#         return w_grad, b_grad
    
# # 나머지 메서드 구현하기
#     def activation(self, z):
#         z = np.clip(z, -100, None)     
#         a = 1 / (1 + np.exp(-z))       
#         return a

#     def fit(self, x, y, epochs=100):
#         self.w = np.ones(x.shape[1])    
#         self.b = 0                    
#         for i in range(epochs):     
#             for x_i, y_i in zip(x, y): 
#                 z = self.forpass(x_i)  
#                 a = self.activation(z)  
#                 err = -(y_i - a)      
#                 w_grad, b_grad = self.backdrop(x_i, err)  
#                 self.w -= w_grad      
#                 self.b -= b_grad   

#     def predict(self, x):
#         z = [self.forpass(x_i) for x_i in x] 
#         a = self.activation(np.array(z)) 
#         return a > 0.5 


# # 모델 훈련 and 결과 확인하기
# neuron = LogisticNeuron()
# neuron.fit(x_train, y_train)
# print(np.mean(neuron.predict(x_test) == y_test))




# Ver.2_손실 함수 결괏값 저장 기능 추가하기

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

