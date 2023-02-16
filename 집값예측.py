import pandas as pd # pandas : 데이터를 데이터 프레임 형태로 다룰수 있음(보다 안정적이고 쉬움), 다양한 통계 함수 + 시각화 기능 갖춤
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error # MSE(MeanSquaredError) : 미분 계산이 쉬워서 모델 최적화에 자주 사용하는 손실 함수임. But 제곱근 씌워 나와서 정확하지X > 루트 씌운게 > RMSE
import matplotlib.pyplot as plt

# 데이터 세트 만들기
df = pd.read_csv('./data/reg.csv', index_col=[0]) #reas_csv : 스케일링된 집값 데이터 불러옴 / index_col=[0] : csv파일의 1번째 열의 데이터 인덱스를 제외하고, 데이터 프레임 만듬

#데이터 프레임을 Numpy 배열로 만들기
X = df.drop('Price', axis=1).to_numpy() # (데이터 프레임) df에서 Price를 제외한 나머지를 변수로 사용함 / axis=1 : 열을 의미함 > Price를 열 기준을 제외하겠다는 뜻임
Y = df['price'].to_numpy().reshape((-1,1)) # Price를 타겟값 Y로 사용 하겠다는 뜻임

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5) # 전체 데이터를 50:50으로 > 학습 데이터 + 평가 데이터로 나눔

# 텐서 데이터 만들기
class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self)
        return self.len

trainsets = TensorData(X_train, Y_train)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=32, suffle=True)
testsets = TensorData(X_test, Y_test)
testloader = torch.utils.data.DataLoader(trainsets, batch_size=32, suffle=False)

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.
