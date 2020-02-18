'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def print_module(m):
    print(m)
net=nn.Module()
net.add_module('conv1',nn.Conv2d(2,2,5))
net.add_module('linear',nn.Linear(2,5))
# net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(print_module) # 将init_weights()函数应用于模块的所有子模块
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
         # 当前的nn.Conv2d模块就被赋值成为Model模块的一个子模块，成为“树结构”的叶子
         # Conv2d相当于定义了一个卷积层，输入1 channel，输出20 channel，卷积核尺寸为5*5
        self.conv1 = nn.Conv2d(1, 5, 2)
        #这里注意，实际的weights.size()=[8*45];因为计算公式是y=xA.T+bias
        self.fc1 = nn.Linear(45, 8)
        #buffer参数注册
        self.ratio=self.register_buffer('ratio',torch.randn(1))
        #对fc1层进行注册
        self.fc1.register_forward_hook(self.forward_hook)

    def forward_hook(self,module, fea_in, fea_out):
        print('forward end')

    def forward(self, x):
       x = F.relu(self.conv1(x))
       x= x.view(1,-1)
       x=F.relu(self.fc1(x))
       return x



model =Model()
#将模型转移到指定设备
# model.cuda(device=0)
model.forward(torch.rand(1,1,4,4))
#将特定的module转移到指定设备
model.to(torch.device("cuda:0"))
