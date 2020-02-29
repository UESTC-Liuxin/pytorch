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

#### 网络搭建

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
        self.conv1 = nn.Sequential( # ->(1,28,28)
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2, #if stride=1 , padding=(kernel_size-1)/2
            ),# ->(6,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # ->(6,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 14, 5,1,2),  # ->(14,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #->(14,7,7)
        )
        # an affine operation: y = xAT + b,进行线性变换
        self.fc1 =nn.Sequential(
            nn.Linear(14*7*7, 120),
            nn.ReLU()
        )
        self.fc2 =nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        x = self.fc1(x)
        x=self.fc2(x)
        out=self.fc3(x)
        #同时返回输出和最后一层qian'd
        return out
```

#### 数据集的加载

在数据集方面，pytorch提供了两个模块

- torch.utils.data

- torchvision

 先介绍比较重要的torch.utils.data，这是一个设计、加载、采样数据集的模块，其中最核心的就是torch.utils.data.DataLoader，我们从这个class介绍。

**torch.utils.data.DataLoader** (*dataset*, *batch_size=1*, *shuffle=False*, *sampler=None*, *batch_sampler=None*, *num_workers=0*, *collate_fn=None*, *pin_memory=False*, *drop_last=False*, *timeout=0*, *worker_init_fn=None*, *multiprocessing_context=None*）

包含了一个数据集和一个采样器，并提供了给定数据集上的迭代性。并同时支持map-style和iterable-style的数据集，可以单/多线程进行读取，定制读取顺序和可选的batch以及内存锁页。

**Parameters：**

**dataset** ([*Dataset*](https://pytorch.org/docs/1.3.1/data.html#torch.utils.data.Dataset)) – dataset from which to load the data.

**batch_size** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – how many samples per batch to load (default: `1`).

**shuffle** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – set to `True` to have the data reshuffled at every epoch (default: `False`).

**sampler** ([*Sampler*](https://pytorch.org/docs/1.3.1/data.html#torch.utils.data.Sampler)*,* *optional*) – defines the strategy to draw samples from the dataset. If specified, `shuffle` must be `False`.

**batch_sampler** ([*Sampler*](https://pytorch.org/docs/1.3.1/data.html#torch.utils.data.Sampler)*,* *optional*) – like `sampler`, but returns a batch of indices at a time. Mutually exclusive with `batch_size`, `shuffle`, `sampler`, and `drop_last`.

**num_workers** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – how many subprocesses to use for data loading. `0` means that the data will be loaded in the main process. (default: `0`)

**collate_fn** (*callable**,* *optional*) – merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.

**pin_memory** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If `True`, the data loader will copy Tensors into CUDA pinned memory before returning them. If your data elements are a custom type, or your `collate_fn` returns a batch that is a custom type, see the example below.

**drop_last** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – set to `True` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If `False` and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: `False`)

**timeout** (*numeric**,* *optional*) – if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: `0`)

**worker_init_fn** (*callable**,* *optional*) – If not `None`, this will be called on each worker subprocess with the worker id (an int in `[0, num_workers - 1]`) as input, after seeding and before data loading. (default: `None`

- **dataset** 

在pytorch中，一共支持两种数据集：

map-style datasets

映射形式数据集实现`__getitem__`以及`__len__`函数，实现了从下标键值到数据样本之间的映射。例如，在图像领域，它可能是这样一个数据集读取模式`dataset[idx]`即读取了第`idx-`个图像对象，以及其对应的标签。通常，我们使用此类数据集，==在类中没有实现`__iter__`方法==，这是与后者最直观的区别，从源代码上可以一眼认出。

iterable-style datasets

递推数据集是[`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) 类的实例，主要实现了`__iter__()`函数，可以递推式读取数据。这种形式的数据集在数据成流式到达时适用。例如一个这样的数据集可以调用`iter(dataset)`来从数据集或者远程服务器，甚至实时生成的数据中返回一个数据流。这个一般不常用。

dataset的基类主要是由迭代器`__getitem__`构成，有多个子类继承于基类，添加了`__iter__()`、`__len()__`等方法，数据集中的数据都为tensor，dataset的来源可以是输入的tensor，利用`TensorDataset`也可以是自定义的dataset的子类，以tuple形式（data，label）的形式在重写的`__getitem__`中返回。

- **shuffle** 

布尔类型，是否打乱随机数据集的顺序，一般来说都是True，缺省值为F。

- **sampler** 、**batch_sampler** 

前者是基于整个数据集的采样器，后者是针对于每个batch内的采样器，要求**shuffle** 参数为F。[具体关系](https://www.cnblogs.com/marsggbo/p/11308889.html)

- **num_workers** 


数据加载器的线程数，可用多线程加载数据。

- **collate_fn**

这是一个回调函数接口，缺省值为`default_collate(batch):`函数，主要作用就是将数据和标签都转换为tensor，并增加一个batch的维度，比batch_size=5,输入为(3,28,28)，输出为(0/1/2/3/4,3,28,28)这样的tensor。每个数据对，都存放在list中，返回应该也是一个list。我们可以自定义一些操作，比如对标签的重塑之类的。

- **pin_memory** 

pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。而显卡中的显存全部是锁页内存！

当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False。因为pin_memory与电脑硬件性能有关，pytorch开发者不能确保每一个炼丹玩家都有高端设备，因此pin_memory默认为False。

- **drop_last**

布尔类型，由于我们的数据/batch_size可能不是整数，那么就会造成最后一个batch数量不足，如果设置为T，就会抛弃最后一个batch。

常用的参数就这几个，其余值，一般都作为缺省值就可以了。

一般来说，数据集来源于自定义数据集和官方的数据集，在torchvision中提供了众多的常见数据集以及下载渠道。这里就不重点介绍了。

```python
# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root=data_root,    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=True,          # 没下载就下载, 下载了就不用再下了
)
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
val_data = torchvision.datasets.MNIST(
    root=data_root,
    train=False
)
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)

val_x = torch.unsqueeze(val_data.data, dim=1).type(torch.FloatTensor)[:2000]/255
# print(test_data.data.size())
val_y = val_data.targets[:2000]
```

这里简单介绍一下，第16行的代码，之所以要在dim=1这里增加一维，是因为本身每张图像是灰度图像，经过读取后，size()=(batchs,H,W)，而训练数据经过train_loader出来数据是（batchs，channels，H, W)这样形式的，所以需要增加一维。同时要进行归一化到（0，1）。[:2000]是指取前2000张。

#### 优化器与损失

```python
#opt
optimizer=torch.optim.Adam(net.parameters(),lr=LR)
#loss
loss_func=  nn.CrossEntropyLoss()
```

在损失函数方面，我们选择了交叉熵函数，关于交叉熵函数的输入输出，以图片分类为例：[CrossEntropyLoss的简单解释](https://blog.csdn.net/qq_22210253/article/details/85229988)，总的来说，input应该是2维数组，dim=0维应当是各图片，dim=1的那一维是图片的类别预测值（非one-hot形式）。target应该是1维的数组，value=类别排列的下标值。

#### 训练与结果输出

```python
if __name__ == '__main__':
    #training
    for epoch in range(EPOCH):
        for step,(b_x,b_y) in enumerate(train_loader):
            if torch.cuda.is_available():
                b_x=b_x.cuda()
                b_y=b_y.cuda()
            out=net(b_x)
            #计算损失时，直接用非one-hot的展开与准确标签做计算，标签也不是one-hot的，就直接是一个数
            loss=loss_func(out,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%50 ==0:
                test_output=net(val_x)
                #利用max求出（value,index) 并将index转为numpy
                pred_y=torch.max(test_output,1)[1].cpu().data.numpy()
                #下面这个操作就很叼，直接将每个对应元素做逻辑相等转换为int再相加，除以总数就是准确率
                accuracy = float((pred_y == val_y.data.numpy()).astype(int).sum()) / float(val_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.2f' % accuracy)
            save_model(net)
```

其中，评估时，是将整个2000张图片一块放了进去。

[完整代码](https://github.com/UESTC-Liuxin/pytorch/blob/master/Exp9.py)