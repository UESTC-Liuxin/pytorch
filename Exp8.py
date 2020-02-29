import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))                # noisy y data (tensor), shape=(100, 1)

BATCH_SIZE=32
LR=0.01
EPOCH=12


#定义数据集
torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

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

if __name__ == '__main__':
    net_SGD=Net(1,1)
    net_Momentum=Net(1,1)
    net_RMSprop=Net(1,1)
    net_Adam=Net(1,1)
    nets=[net_SGD,net_Momentum,net_RMSprop,net_Adam]

    opt_SGD     = torch.optim.SGD(params=net_SGD.parameters(),lr=LR)
    opt_Momentum= torch.optim.SGD(params=net_Momentum.parameters(),lr=LR,momentum= 0.8)
    opt_RMSprop = torch.optim.RMSprop(params=net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam    = torch.optim.Adam(params=net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
    optimizers=[opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]

    loss_func=nn.MSELoss()
    loss_his=[[],[],[],[]]

    for epoch in range(EPOCH):
        for step,(batch_x,batch_y) in enumerate(loader):
            for net,opt,l_his in zip(nets,optimizers,loss_his):
                output=net(batch_x)
                loss=loss_func(output,batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.item())
        labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
        for i, l_his in enumerate(loss_his):
            plt.plot(l_his, label=labels[i])
        plt.legend(loc='best')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.ylim((0, 0.2))
        plt.show()
