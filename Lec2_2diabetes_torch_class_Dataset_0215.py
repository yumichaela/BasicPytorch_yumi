import torch 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

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

torch.manual_seed(777)
np.random.seed(777)

# Step 1 데이터 준비
diabetes = load_diabetes()
x = diabetes.data[:, 2]
y = diabetes.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


xt = torch.FloatTensor(x)  # numpy array를 torch의 tensor로 변환합니다. 
yt = torch.FloatTensor(y)

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1,1)

    def forward(self, x):
        return self.model(x) # 모델의 역할 분담을 클스로 나눠줌


class Trainer(torch.nn.Module):
    def __init__(self, x_train, y_train, batch_size, lr=1e-2):
        super().__init__()
        self.model = LinearRegression() 
        """
        Linear 함수로 w (1차원), b (1차원) 텐서를 만들어줍니다. 
        
        w = torch.ones([1], requires_grad=True)
        b = torch.ones([1], requires_grad=True)
        이거 대신 Linear를 사용합니다. 
        """
        self._set_optimizer(lr)
        self._set_loss()
        self._set_dataset(x_train, y_train, batch_size)

    def _set_optimizer(self, lr):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        
    def _set_loss(self):
        self.loss_fn = torch.nn.MSELoss()

    def _set_dataset(self, x_train, y_train, batch_size):
        trainsets = TensorData(x_train, y_train)
        self.dataloader = torch.utils.data.DataLoader(trainsets, batch_size = batch_size, shuffle=True)
        
    def train(self, epoch, fig_name):
    
        for i in range(epoch):
            epoch_loss = 0
            cnt = 0 
            for a, b in self.dataloader:
                pred_y = self.model(a.unsqueeze(1)) #[batch_size, 1]
                loss = self.loss_fn(b.unsqueeze(1), pred_y)

                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step() 

                epoch_loss = epoch_loss + loss.data.numpy()
                epoch_loss = epoch_loss / a.size(0)
                cnt += 1
        
            epoch_loss = epoch_loss / cnt
            print("Epoch : {}, Loss : {}".format(i+1, epoch_loss))
        
        breakpoint()
        
#         for k in range(epoch):
#             idx = np.random.permutation(idx)  
#             # permutaion은 안에 순서를 바꾸어줍니다. 
#             epoch_loss = 0
#             for i in idx:
#                 """ 
#                 idx의 순서가 바뀌어 들어간다는 것은 epoch마다 학습에 사용되는 샘플의 순서가 바뀐다는 뜻입니다. 
#                 """
#                 y_hat = self.model(xt[i].unsqueeze(0))
#                 loss = self.loss_fn(y_hat, yt[i].unsqueeze(0))

#                 epoch_loss += loss.data.numpy()
#                 self.optimizer.zero_grad()
#                 loss.backward() 
#                 self.optimizer.step() 
            
#             epoch_loss = epoch_loss / len(idx)
#             print("Epoch : {}, Loss : {}".format(k, epoch_loss))
                
#             if (k + 1) % 20 == 0:
#                 plot(x, y, self.model.weight[0], self.model.bias[0], cnt, name=fig_name)
#                 cnt += 1
                


# my_model = LinearRegression(lr=1e-2)
# my_model.train(epoch=100, fig_name="lec2_3_")

trainer = Trainer(x_train, y_train, 4, 1e-3)
trainer.train(10, "test")