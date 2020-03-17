import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)



class Net(nn.Module):
    def __init__(self,in_feature,out):
        super(Net,self).__init__()
        #construct net
        self.hidden1=nn.Linear(in_feature,100)
        self.hidden2=nn.Linear(100,20)
        self.predict=nn.Linear(20,out)

    def forward(self, x):
        x=F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x=self.predict(x)
        return x


net=Net(1,1)
#优化器
optimizer=torch.optim.SGD(params=net.parameters(),lr=0.1)
loss_func=nn.MSELoss()

#进入gpu中运算
if torch.cuda.is_available():
    net.cuda(device=0)
    x=x.to("cuda")
    y=y.to("cuda")
print(x.size())

plt.ion()   # 画图
plt.show()
for i in range(500):
    prediction=net(x)
    loss=loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.cpu().data.numpy(), y.cpu().data.numpy())
        plt.plot(x.cpu().data.numpy(), prediction.cpu().data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.cpu().data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.01)
print(loss)
