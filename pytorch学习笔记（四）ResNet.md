# ResNet以及在CIFAR上实现分类

## ResNet介绍

ResNet全名Residual Network残差网络。Kaiming He 的《Deep Residual Learning for Image Recognition》获得了CVPR最佳论文。他提出的深度残差网络在2015年可以说是洗刷了图像方面的各大比赛，以绝对优势取得了多个比赛的冠军。而且它在保证网络精度的前提下，将网络的深度达到了152层，后来又进一步加到1000的深度。论文的开篇先是说明了深度网络的好处：特征等级随着网络的加深而变高，网络的表达能力也会大大提高。因此论文中提出了一个问题：是否可以通过叠加网络层数来获得一个更好的网络呢？作者经过实验发现，单纯的把网络叠起来的深层网络的效果反而不如合适层数的较浅的网络效果。

Resnet网络的提出者Balduzzi D 利用实验设计了在已经训练的浅层网络上，添加identity mapping，按照常理来说，至少添加identity mapping不应该会比未修改的浅层网络表现效果更差，但是根据实验数据表明，并非如此。

| ![img](md_img/Exp13_1.png)                                   |
| ------------------------------------------------------------ |
| 20层网络与增加层数的56层网络训练错误收敛图 （此图来源于：Deep Residual Learning for Image Recognition） |

随着迭代次数增加，20层网络与56层网络的错误率都在收敛，56层网络的收敛速度明显低于20层，这和预期结果相同，但当迭代次数增加到50000次时，两种网络都收敛至一个稳定值，不再有明显上升或下降，但收敛值却没有如预期的那样，应该呈现出收敛至同一值，56层训练网络的错误率收敛值明显高于20层，也就是说，前者的训练效果远不如前者。作者把这种现象称为degradation problem(降级)。

==注意：这种问题并不是随着网络加深造成的梯度消失或者梯度爆炸，虽然残差网络依然可以解决这两个问题，到那时这两个问题通常都被BN和Relu激活函数得到解决。==

degradation problem的出现证明主流的训练方法存在一定的缺陷，而出现降级的根本原因，并不明确，在“The Shattered Gradients Problem: If resnets are the answer, then what is the question?”中提出了一种说法：神经网络越来越深的时候，反传回来的梯度之间的相关性会越来越差，最后接近白噪声。因为图像是具备局部相关性的，那其实可以认为梯度也应该具备类似的相关性，这样更新的梯度才有意义，如果梯度接近白噪声，那梯度更新可能根本就是在做随机扰动。

[知乎答案](https://www.zhihu.com/question/64494691/answer/271335912)王峰那一个答案（很巧合，后来发现这个人原来是我师兄）。

### 残差模块

何恺明团队提出的残差网络结构 将$F(x)$替换为$H(x)=F(x)+x$，非线性网络结构实际上学习的是$H(x)-x$这样一个残差，而为什么要做这么呢？因为，在极端情况下，返回的损失已经很小，效果已经到达了网络的极限，最终学习的$F(x)$为0，也就是说整个二层网络不会造成任何影响，最终是一个恒等映射，那么至少，网络不会存在更差的情况，同时，如果不使用残差网络结构，这一层的输出F'(5)=5.1 期望输出 H(5)=5 ,如果想要学习H函数，使得F'(5)=H(5)=5,这个变化率较低，学习起来是比较困难的。但是如果设计为H(5)=F(5)+5=5.1，进行一种拆分，使得F(5)=0.1，那么学习目标是不是变为F(5)=0，一个映射函数学习使得它输出由0.1变为0，这个是比较简单的。也就是说引入残差后的映射对输出变化更敏感了。进一步理解：如果F'(5)=5.1 ,现在继续训练模型，使得映射函数F'(5)=5。(5.1-5)/5.1=2%，也许你脑中已经闪现把学习率从0.01设置为0.0000001。浅层还好用，深层的话可能就不太好使了。如果设计为残差结构呢？5.1变化为5，也就是F(5)=0.1变化为F(5)=0.这个变化率增加了100%。引入残差后映射对输出变化变的更加敏感了，这也就是为什么ResNet虽然层数很多但是收敛速度也不会低的原因。明显后者输出变化对权重的调整作用更大，所以效果更好。残差的思想都是去掉相同的主体部分，从而突出微小的变化，看到残差网络我第一反应就是差分放大器。这也就是当网络模型我们已经设计到一定的深度，出现了精准度下降，如果使用残差结构就会很容易的调节到一个更好的效果，即使你不知道此刻的深度是不是最佳，但是起码准确度不会下降。

![](https://img-blog.csdn.net/2018042621572485?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N1bnFpYW5kZTg4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### BN层

在resnet的实际结构中，实际上在每一个卷积之后都紧接了一个BN层。

[论文](https://arxiv.org/abs/1502.03167)当中提到的几个优点：

1）加快训练速度，这样我们就可以使用较大的学习率来训练网络。

2）提高网络的泛化能力。

3）BN层本质上是一个归一化网络层，可以替代局部响应归一化层（LRN层）。

4）可以打乱样本训练顺序（这样就不可能出现同一张照片被多次选择用来训练）论文中提到可以提高1%的精度。

![](https://img-blog.csdn.net/20180820151214146?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RvbmtleV8xOTkz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

BN的基本思想其实相当直观：因为深层神经网络在做非线性变换前的激活输入值（就是那个x=WU+B，U是输入）随着网络深度加深或者在训练过程中，其分布逐渐发生偏移或者变动，**之所以训练收敛慢，一般是整体分布逐渐往非线性函数的取值区间的上下限两端靠近（对于Sigmoid函数来说，意味着激活输入值WU+B是大的负值或正值），所以这导致反向传播时低层神经网络的梯度消失，这是训练深层神经网络收敛越来越慢的本质原因**，而BN就是通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到均值为0方差为1的标准正态分布，其实**就是把越来越偏的分布强制拉回比较标准的分布，这样使得激活输入值落在非线性函数对输入比较敏感的区域，这样输入的小变化就会导致损失函数较大的变化，意思是这样让梯度变大，避免梯度消失问题产生，而且梯度变大意味着学习收敛速度快，能大大加快训练速度。**

其实一句话就是：==**对于每个隐层神经元，把逐渐向非线性函数映射后向取值区间极限饱和区靠拢的输入分布强制拉回到均值为0方差为1的比较标准的正态分布，使得非线性变换函数的输入值落入对输入比较敏感的区域，以此避免梯度消失问题**==。因为梯度一直都能保持比较大的状态，所以很明显对神经网络的参数调整效率比较高，就是变动大，就是说向损失函数最优值迈动的步子大，也就是说收敛地快。BN说到底就是这么个机制，方法很简单，道理很深刻。（https://www.cnblogs.com/eilearn/p/9780696.html）。

由于如果把所有的输入都放在了激活函数的线性区，那么整个深层网络就变成了线性变化的堆叠，如果你稍微对深度卷积网路有一些了解的化，你就会知道，这样是没有意义的，因此$y_{i} \leftarrow \gamma \widehat{x}_{i}+\beta \equiv \mathrm{B} \mathrm{N}_{\gamma, \beta}\left(x_{i}\right)$的意义在于，将整个输入又稍微向着非线性区向着进行了扩展和偏移。$ \gamma,\beta$都式学习来的。

## 手动实现ResNet

<img src="https://pic1.zhimg.com/80/v2-1dfd4022d4be28392ff44c49d6b4ed94_720w.jpg" style="zoom:150%;" />

<center>resnet18/34/50/101/152</center>

### 残差模块实现

首先，resnet18/34与resnet50/101/152的残差结构有一些区别![](https://pic4.zhimg.com/80/v2-9f4ebe0a9ea229144fbbf7e7015a4f57_720w.jpg)

resnet18/34的残差模块是由两个64通道的3X3卷积构成，而resnet50/101/152就是用两组1X1卷积和一个3X3卷积构成。

根据上表，可以看出，resnet把所有的残差结构分为四组，每组有多个残差模块（例如resnet18一共就是8个，每组4个），每一种深度的resnet网络都定义了一个expansion系数，每组残差模块的输入层的channel都等于输入的expansion倍。resnet18(50)的expansion系数等于1(4).

以resnet18为例，残差模块的快速连接分为两种情况：

- 残差模块的输入与3x3卷积的输入维度相同，那么直接相加就好

- 如果不同，那么需要进行一个维度的变换，尺寸的变换依靠stride进行实现，由于都是能改变尺寸的都是3X3卷积
  $$
  卷积后的尺寸：H_1=\frac{H_0+2P-K}{2}=\frac{H_0}{2}+\frac{2-3}{2}
  $$

  $$
  变换后的尺寸：H_2=\frac{H_0-K}{2}=\frac{H_0}{2}+\frac{-1}{2}
  $$

  是相等的。

  深度的变换就用卷积核个数就可以了。最后要加上BN。

  ==注意：深度的不同只发生在4组残差模块组合的每一组的第一个，因为上一组发生了深度的改变，尺度的改变是因为从第二组开始，3X3卷积的stride由1变成了2，相当于都减了一半。==

相加操作：`out += self.shortcut(x)`实现。

BasicBlock：18/34

```python
# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

Bottleneck：50/101/152  1X1 、3X3的组合

```python
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
```

### 残差网络

```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks,input_size,num_classes=10,):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        f=lambda x: x//32
        self.linear = nn.Linear(512 * block.expansion*f(input_size[0]*f(input_size[1])), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            # 这里解释一下为什么要有个expansion，因为resnet的残差结构当中最后一个1x1的卷积的filters=前两个的4倍
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out =self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```

这里的代码没有太多要解释的，主要就是_make_layer这个函数，构造一组残差模块为一层，四个参数（残差模块，输出通道，残差模块个数，stride）。`strides = [stride] + [1] * (num_blocks - 1)`这句代码的意思是，除了每组第一个的stride需要定以外，其余的都是1，因为每一组卷积之后，尺寸只减少一半。

然后就是关于输入尺寸的问题，原本第一层的7X7卷积，在后面改成了3X3卷积，那么尺度全连接层的参数应当是inputsize//7，如果尺寸大于32，就会报错，于是源码修改一下：

```python
        f=lambda x: x//32
        self.linear = nn.Linear(512 * block.expansion*f(input_size[0]*f(input_size[1])), num_classes)
```

适应尺度。

## 训练与测试

基本的操作还是与上一个帖子一致，就是把网络更换为resnet18，但是最后的测试结果发现，还是只有85%左右，那么我将持续尝试更多的方法去提高模型的效率。。！！！！！！加油。

## 总结

resnet真的是一个具有重大影响力的突破，解决了很多问题，同时也表现出了不俗的性能，在其余应用中，我们也还是在可以将resnet网络作为其backbone，用于提取高维特征。

另外，还是想说一句：何恺明牛逼！！！王峰师兄牛逼！！！



