# 在CIFAR上的优化调参

# 前言

这次的实验主要是为了针对笔记（三）和笔记（四）上的后续的操作，同时也是为了撰写高级数字图像处理的论文而做理论和数据准备。

## 网络结构更换

### 自定义网络

第一次实验是在自己设计的一个7层网络上进行的，2层卷机层，2层池化层，3层全连接层。训练过程未做任何处理，最后结果在40%。包括后来调整了很多参数，最后只能达到下面的效果。

<img src="DATA/L2/C1.PNG" alt="C1" style="zoom: 67%;" /><img src="DATA/L2/C2.PNG" alt="C1" style="zoom: 67%;" />
         <img src="DATA/L2/C3.PNG" alt="C3" style="zoom: 67%;" /><img src="DATA/L2/C4.PNG" alt="C4" style="zoom: 67%;" />

从上面的图可以明显的看到，此模型的训练误差不能收敛到一个较低值，训练准确率 也上不去，说明此模型的复杂度完全不能够解决此问题，需要更换更深层次的网络。

### ResNet18/50网络

网络更换为了Resnet18时，最初在训练集上能达到80%的acc.

  <img src="DATA/resnet18_base/1.PNG" alt="1" style="zoom:63%;" /><img src="DATA/resnet18_base/2.PNG" alt="2" style="zoom:55%;" />    

<img src="DATA/resnet18_base/3.PNG" alt="3" style="zoom:55%;" /><img src="DATA/resnet18_base/4.PNG" alt="4" style="zoom:55%;" />

其实已经明显看出，模型已经在出现了严重的过拟合现象。这个时候应该想到的是进行过拟合处理，但是在做过拟合处理之前，我还进行了一个处理，就是lr_scheduler。

### DenseNet网络

DenseNet还未来得及研究，实现是通过torchvision集成的API，最后进行了相同的训练。但是效果不及resnet。

### 总结

网络的深度和宽度，往往决定了能解决问题的复杂程度，但对于同一任务来说，如果任务复杂程度，远远低于了网络的复杂程度，那么持续选择更复杂的网络，并不会得到一个更好的结果，反而增加计算量。

## 学习率适应

由于学习过程如果一直用同一学习率，学习率过小很可能网络陷入局部最优，学习率过大很可能难以收敛到一个比较到最优指，因此，通常会采用一些学习率计划在训练中调整学习率。

### StepLR

- batch size=64 ;step=20;gamma=0.2
<img src="DATA/batchsize64_15_g0.05/1.png" alt="1" style="zoom: 90%;" />  <img src="DATA/batchsize64_15_g0.05/2.png" alt="2" style="zoom:90%;" />
  <img src="DATA/batchsize64_15_g0.05/3.png" alt="3" style="zoom:90%;" />    <img src="DATA/batchsize64_15_g0.05/4.png" alt="4" style="zoom:90%;" /> 

- batch size=64 ;step=15;gamma=0.1

<img src="DATA/batchsize128_20_g0.2/1.png" alt="1" style="zoom:90%;" /><img src="DATA/batchsize128_20_g0.2/2.png" alt="1" style="zoom:90%;" />

<img src="DATA/batchsize128_20_g0.2/3.png" alt="3" style="zoom: 90%;" /><img src="DATA/batchsize128_20_g0.2/4.png" alt="4" style="zoom:90%;" />

### 自定义lr_scheduler

根据网上的案例，这次做出重新调整，自定义lr_scheduler。

```python
def adjust_learning_rate(optimizer,lr):
    lr=lr*0.1
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

```

- batch size=200 ;step=[130,180,200];gamma=0.1

<img src="DATA/batchsize128_cust_1/1.png" alt="1" style="zoom:50%;" />    <img src="DATA/batchsize128_cust_1/2.png" alt="2" style="zoom:52%;" />

<img src="DATA/batchsize128_cust_1/3.png" alt="3" style="zoom:50%;" /> <img src="DATA/batchsize128_cust_1/4.png" alt="4" style="zoom:55%;" />

明显看到，自定义的学习率函数表现出了一定的提升，但是整体效果还是不达标。

## 过拟合处理

根据前面的测试与优化，明显从测试集的loss情况和训练集的loss情况，已经可以看出，模型出现了严重的过拟合现象，因此我尝试了一些常规的过拟合处理，包括正则化，dropout，以及训练集图像的预处理（裁剪，反转等）。

### 加入L2正则化

<img src="DATA/L2/1.png" alt="1" style="zoom:50%;" /><img src="DATA/L2/2.png" alt="2" style="zoom:50%;" />

<img src="DATA/L2/3.png" alt="3" style="zoom:50%;" /><img src="DATA/L2/4.png" alt="4" style="zoom:50%;" />

以上的图片显示了，在训练的

###  训练图像预处理

<img src="DATA/L2/5.PNG" alt="5" style="zoom:50%;" /><img src="DATA/L2/6.PNG" alt="6" style="zoom:50%;" />

<img src="DATA/L2/8.PNG" alt="8" style="zoom:50%;" /><img src="DATA/L2/7.PNG" alt="7" style="zoom:50%;" />

### dropout

由于本身在resnet网络中，加入了BN层，那么dropout就没有必要再添加，同时我也做了测试，加了dropout后，效果反而降低了。

