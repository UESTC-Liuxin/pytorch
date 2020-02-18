import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1, 5, 2),
            nn.ReLU()
        )
        self.lin=nn.Sequential(
            nn.Linear(45, 8),
            nn.ReLU()
        )
        #buffer参数注册
        self.ratio=self.register_buffer('ratio',torch.randn(1))
        #对fc1层进行注册
        self.lin.register_forward_hook(self.forward_hook)

    def forward_hook(self,module, fea_in, fea_out):
        print('forward end')

    def forward(self, x):
       x = self.conv(x)
       x= x.view(1,-1)
       x=self.lin(x)
       return x



model =Model()
#将模型转移到指定设备
# model.cuda(device=0)
model.forward(torch.rand(1,1,4,4))
#将特定的module转移到指定设备
model.to(torch.device("cuda:0"))