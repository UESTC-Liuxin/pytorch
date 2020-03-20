[TOC]

## CIFAR图片分类实现

## 数据获取

这里数据获取依旧使用torchvision，后面会尝试做一点自己的数据集。这里关注一下归一化这个问题,在没有使用BN的情况下，我们通常会对数据进行归一化或者标准化，好处我大概总结了一下：

- 计算机大数吞小数的情况，也就是数值方面的问题
- 在训练中，针对激活函数的非线性性，在区间限制内才有比较好的非线性性。
- 梯度的数量级可能变化非常大，同时可能会存在数值问题
- 权值太大，学习率就必须非常小，也会引发数值问题
- 收敛会更快

关于归一化的一些理解：https://blog.csdn.net/program_developer/article/details/78637711

```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),#每个channel，mean，std=0.5
     # transforms.CenterCrop
     ] #3个通道的平均值和方差都取0.5
)

#训练数据集
trainset =torchvision.datasets.CIFAR10(
    root=data_root,
    train=True,
    download=False,
    transform=transform
)
trainloader =  Data.DataLoader(
    dataset=trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

#测试数据集
testset=torchvision.datasets.CIFAR10(
    root=data_root,
    train=False,
    transform=transform
)
testloader=Data.DataLoader(
    dataset=testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           
```

