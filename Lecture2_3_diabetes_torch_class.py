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

torch.manual_seed(777)
np.random.seed(777)

# Step 1 데이터 준비
diabetes = load_diabetes()
x = diabetes.data[:, 2]
y = diabetes.target

xt = torch.FloatTensor(x)  # numpy array를 torch의 tensor로 변환합니다. 
yt = torch.FloatTensor(y)

class LinearRegression(torch.nn.Module):
    def __init__(self, lr=1e-2):
        super().__init__()
        self.model = torch.nn.Linear(1, 1) 
        """
        Linear 함수로 w (1차원), b (1차원) 텐서를 만들어줍니다. 
        
        w = torch.ones([1], requires_grad=True)
        b = torch.ones([1], requires_grad=True)
        이거 대신 Linear를 사용합니다. 
        """
        self._set_optimizer(lr)
        self._set_loss()

    def _set_optimizer(self, lr):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        
    def _set_loss(self):
        self.loss_fn = torch.nn.MSELoss()
        
    def train(self, x, y, xt, yt, epoch, fig_name):
        cnt = 0 
        idx = [i for i in range(len(x))]  
        
        for k in range(epoch):
            idx = np.random.permutation(idx)  
            # permutaion은 안에 순서를 바꾸어줍니다. 
            epoch_loss = 0
            for i in idx:
                """ 
                idx의 순서가 바뀌어 들어간다는 것은 epoch마다 학습에 사용되는 샘플의 순서가 바뀐다는 뜻입니다. 
                """
                y_hat = self.model(xt[i].unsqueeze(0))
                loss = self.loss_fn(y_hat, yt[i].unsqueeze(0))

                epoch_loss += loss.data.numpy()
                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step() 
            
            epoch_loss = epoch_loss / len(idx)
            print("Epoch : {}, Loss : {}".format(k, epoch_loss))
                
            if (k + 1) % 20 == 0:
                plot(x, y, self.model.weight[0], self.model.bias[0], cnt, name=fig_name)
                cnt += 1
                


my_model = LinearRegression(lr=1e-2)
my_model.train(x, y, xt, yt, epoch=100, fig_name="lec2_3_")

        

