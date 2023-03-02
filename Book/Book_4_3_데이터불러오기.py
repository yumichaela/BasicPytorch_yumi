import torch
import torchvision #이미지 관련된 파이토치 Library
import torchvision.transforms as tr # 이미지 전처리 기능 Lib
from torch.utils.data import DataLoader, Dataset # 데이터를 모델에 사용할 수 있게 정리 해주는 Lib
import numpy as np
import matplotlib.pyplot as plt

transf = tr.Compose([tr.Resize(16),tr.ToTensor()]) 
# tr.Compose 내에 원하는 전처리를 차례대로 넣어주면 됨
# ex) 16x16으로 이미지 크기 변환 후 > 텐서 타입으로 변환함 / tr.Resize(16) : 원본 이미지의 너비, 높이가 다를 경우 각각 지정해줌

trainset = torchvision.datasets.CIFAR10(root = './data', train=True, download=True, transform=transf)
testset = torchvision.datasets.CIFAR10(root = './data', train=False, download=True, transform=transf)
# torchvision.datasets에서 제공하는 CIFAR10  