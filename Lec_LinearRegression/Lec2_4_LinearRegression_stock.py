import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


stock = pd.read_csv(filepath_or_buffer="kospi_kosdak.csv", encoding='utf-8')
x = stock["Kospi"].to_numpy().astype(float)
y = stock["Kosdak"].to_numpy().astype(float)

def minmaxscale(x, amin=None, amax=None):
    if amin is None:
        amin = np.min(x)
    if amax is None:
        amax = np.max(x)
    y = (x-amin) / (amax-amin+1e-5) # 0이 되는 것을 방지
    return y, amin, amax

def minmaxscale_reverse(y, amin=None, amax=None):
    x = y * (amax-amin+(1e-5)) + amin
    return x


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

x_train, amin_x, amax_x = minmaxscale(x_train)
y_train, amin_y, amax_y = minmaxscale(y_train)

x_test, _, _ = minmaxscale(x_test, amin_x, amax_x)
y_test, _, _ = minmaxscale(y_test, amin_y, amax_y) # 이것 때문에 min, max 값을 받았음

torch.manual_seed(777)
np.random.seed(777)

xt = torch.FloatTensor(x_train)
yt = torch.FloatTensor(y_train)

xt = xt.unsqueeze(1)
yt = yt.unsqueeze(1)

print(xt.size(), yt.size())

class LinearRegression(torch.nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__() # nn.Module 쓰려면 필요함
        self.model = torch.nn.Linear(d_input, d_output)

    def forward(self, x):
        output = self.model(x) # torch.nn.Module을 상속받는 클래스들은 forward 함수를 생략할 수 있음
        return output

class Trainer:
    def __init__(self, d_in, d_out, lr):
        self._build(d_in, d_out) # 모델을 정의해줌  
        self._set_optimizer(lr)
        self._set_loss()

    def _build(self, d_in, d_out):
        self.model = LinearRegression(d_in, d_out) # 위에서 정의해주고 > 밑에 가져와서 씀

    def _set_optimizer(self, lr):
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr = lr)

    def _set_loss(self):
        self.loss = torch.nn.MSELoss()

    def run(self, xt, yt, epoch):
        cnt = 0
        idx = [i for i in range(len(xt))]

        for k in range(epoch):
            idx = np.random.permutation(idx)
            epoch_loss = 0
            for i in idx:
                y_hat = self.model(xt[i].unsqueeze(0))
                loss = self.loss(y_hat, yt[i].unsqueeze(0))

                epoch_loss += loss.data.numpy()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss = epoch_loss / len(idx)

            print("Epoch : {}, Loss : {}".format(k, epoch_loss))

            cnt += 1

    def predict(self, x):
        y = self.model(x)
        return y

trainer = Trainer(1, 1, lr = 1e-2)
trainer.run(xt, yt, epoch=100)    

plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test)
plt.xlabel("kospi")
plt.ylabel("kosdak")

point1 = (0, 0*trainer.model.model.weight.data.numpy()[0] + trainer.model.model.bias.data.numpy()[0])
point2 = (1, 1*trainer.model.model.weight.data.numpy()[0] + trainer.model.model.bias.data.numpy()[0])

# plt.plot([point1[0],point2[0]], [point1[1], point2[1]]) 
# plt.show()      

x = 2417.68
x,_,_ = minmaxscale(x, amin_x, amax_x)
x = torch.FloatTensor([x]).unsqueeze(1) # x에 대괄호 안씌워주면 그 갯수가 나옴
result = trainer.predict(x).squeeze(1)
result = minmaxscale_reverse(result.data.numpy(), amin_y, amax_y)
print(result)
