import torch
import torch.nn as nn
import torch.nn.functional as F
'''
一个神经网络的典型训练过程如下：

定义包含一些可学习参数（或者叫权重）的神经网络
在输入数据集上迭代
通过网络处理输入
计算损失（输出和正确答案的距离）
将梯度反向传播给网络的参数
更新网络的权重，一般使用一个简单的规则：weight = weight - learning_rate * gradient
'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = xAT + b,进行线性变换
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #num_flat_features：获取所有特征值，也就是数的个数
        #进行一个一维展开，也就是全连接层
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(net)
#
# params = list(net.parameters())
# for i in params:
#     print(i.size())  # conv1's .weight

#手动输入
input = torch.randn(1, 1, 32, 32)
out = net(input)
# print(out)
#清空参数的梯度缓存
net.zero_grad()
output = net(input)
target = torch.randn(10)  # 本例子中使用模拟数据
target = target.view(1, -1)  # 使目标值与数据值形状一致
criterion = nn.MSELoss()
loss = criterion(output, target)
# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)
# loss.backward()
# learning_rate = 0.01
# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)
import torch.optim as optim

# 创建优化器（optimizer）
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练的迭代中：
optimizer.zero_grad()   # 清零梯度缓存
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 更新参数