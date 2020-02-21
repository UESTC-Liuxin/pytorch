[TOC]

# 搭建第一个神经网络

## Module模块

模块（Module）是所有神经网络模型的基类，新建的网络应该继承于它。

### 基本属性

Moudle可以用add_module(name,module)添加模块；

apply(function)将function函数应用于每个子模块和父模块；

```python
def print_module(m):
    #定义一个打印module信息的函数
    print(m)
#定义一个net实例
net=nn.Module()
#添加module(name,module)
net.add_module('conv1',nn.Conv2d(2,2,5))
net.add_module('linear',nn.Linear(2,5))
# net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(print_module) # 将init_weights()函数应用于模块的所有子模块
>>>>>
Conv2d(2, 2, kernel_size=(5, 5), stride=(1, 1))
Linear(in_features=2, out_features=5, bias=True)
Module(
  (conv1): Conv2d(2, 2, kernel_size=(5, 5), stride=(1, 1))
  (linear): Linear(in_features=2, out_features=5, bias=True)
)

```

Pytorch模型中有两种参数：parameter与buffer，前者会在反向传播中更新，比如weights和bias；后者不会更新。

model.cpu()/cuda(device=)可以将整个模型的parameter与buffer都进行cpu/gpu转移。（注意：==在转移时一定要注意，输入也必须是gpu/cpu上的==）

eval()/train(mode=True):可将model置于测试/训练状态。

state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)(注意,==只有那些参数可以训练的layer才会被保存到模型的state_dict中,如卷积层,线性层等等==)；优化器对象Optimizer也有一个state_dict,它包含了优化器的状态以及被使用的超参数(如lr, momentum,weight_decay等)。

register_backward_hook(hook)：在模块上注册一个挂载在反向操作之后的钩子函数。（挂载在backward之后这个点上的钩子函数）。对于每次输入，当模块关于此次输入的反向梯度的计算过程完成，该钩子函数都会被调用一次。`hook(module, grad_input, grad_output) -> Tensor or None`

```python
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

```

## Sequential

一种顺序容器。传入Sequential构造器中的模块会被按照他们传入的顺序依次添加到Sequential之上。相应的，一个由模块组成的顺序词典也可以被传入到Sequential的构造器中。上面的moudle可换个方式写。

```python
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
        self.lin.register_forward_hook(self.forward_hook)

    def forward_hook(self,module, fea_in, fea_out):
        print('forward end')

    def forward(self, x):
       x = self.conv(x)
       x= x.view(1,-1)
       x=self.lin(x)
       return x

```

## minist数据集手写数字识别网络搭建

### 网络框图

<img src="https://pytorch.org/tutorials/_images/mnist.png" style="zoom:80%;" />

### 基本方法介绍

1.**Conv2d**

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

   利用指定大小的二维卷积核对输入的多通道二维输入信号进行二维卷积操作的卷积层。


Conv2d的参数:

- **in_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)) – 输入通道个数
- **out_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)) – 输出通道个数
- **kernel_size** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)) – 卷积核大小
- **stride** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)*,* *optional*) –卷积操作的步长。 默认： 1
- **padding** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)*,* *optional*) – 输入数据各维度各边上要补齐0的层数。 默认： 0
- **dilation** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)*,* *optional*) –卷积核各元素之间的距离。 默认： 1
- **groups** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – 输入通道与输出通道之间相互隔离的连接的个数。 默认：1
- **bias** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 如果被置为 `True`，向输出增加一个偏差量，此偏差是可学习参数。 默认：`True`

简单解释一下==dilation==参数和==groups==参数：dilation是指的卷积过程中，卷积核有空洞，会漏掉一些值；groups是指每个卷积核需要多少层，比如输入为4 channels，设groups=2，那么每两个通道会用一个卷积层，以一共两层，所以必须channels/groups，必须整除。

<img src="https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/dilation.gif" style="zoom: 33%;" />
$$
H_{o u t}=\left\lfloor\frac{H_{i n}+2 \times \text { padding }[0]-\text { dilation }[0] \times\left(\text { kernel }_{-} \operatorname{size}[0]-1\right)-1}{\operatorname{stride}[0]}+1\right]
$$

2. **nn.ConvTranspose1d**:

   ```python
   class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
   ```

   反卷积，卷积过程通常以大尺寸输入得到了一个小的尺寸的输出，然而反卷积就是以小尺寸输入获得大尺寸输出。具体的原理和解释可以查看这篇文章：[转置卷积(transposed convolution)/反卷积(deconvolution)](https://blog.csdn.net/lanadeus/article/details/82534425)

3. **MaxPool2d**：

   对输入的多通道信号执行二维最大池化操作。

4. **Linear**
   ```python
class torch.nn.Linear(in_features, out_features, bias=True)
   ```
线性变换，主要用在全连接层线性变换的：$y=xA^T+bias,其中，A指的是权重需要转置$

Parameters:

   - **in_features** – size of each input sample
   - **out_features** – size of each output sample
   - **bias** – If set to False, the layer will not learn an additive bias. Default: `True`

### 开始搭建 

一个神经网络的典型训练过程如下：

1. 定义包含一些可学习参数（或者叫权重）的神经网络
2. 在输入数据集上迭代
3. 通过网络处理输入
4. 计算损失（输出和正确答案的距离）
5. 将梯度反向传播给网络的参数
6. 更新网络的权重，一般使用一个简单的规则：weight = weight - learning_rate * gradient

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
'''

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
#手动输入
#通常神经网络的输入都是(batch_size,channel,H,W)
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


# 创建优化器（optimizer）
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练的迭代中：
optimizer.zero_grad()   # 清零梯度缓存
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 更新参数
```

