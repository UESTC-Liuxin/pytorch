import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import visdom
import numpy as np


# vis = visdom.Visdom()
# vis.text('Hello, world!')
# vis.image(np.ones((3, 10, 10)))
#get win/linux path of dataset
root=os.getcwd()
data_root=os.path.join(root,'DATA')

# Hyper Parameters
EPOCH = 10           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 8
LR = BATCH_SIZE*0.0125          # 学习率

transform = transforms.Compose(
    [transforms.ToTensor()
     # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
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

if __name__ == '__main__':
    # 随机获取训练图片
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.size())
    vis = visdom.Visdom()
    vis.text(classes[0],win='win1')
    # print(img.size())
    # torch.transpose(img,)
    vis.images(images,win='win1')
    labels=testset.targets
    vis.histogram(labels,win='win2').title('testset')