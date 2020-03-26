import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from visdom import Visdom
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

# vis = visdom.Visdom()
# vis.text('Hello, world!')
# vis.image(np.ones((3, 10, 10)))
#get win/linux path of dataset
root=os.getcwd()
data_root=os.path.join(root,'DATA')

# Hyper Parameters
EPOCH = 10           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 8
LR = BATCH_SIZE*0.00125          # 学习率

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
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

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet,self).__init__()
        self.Conv=nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
                padding=2
            ),#if stride=1 , padding=(kernel_size-1)/2
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5
            ),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.Fc=nn.Sequential(
            nn.Linear(16*6*6,20),
            nn.ReLU(),
            nn.Linear(20,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )

    def forward(self, x):
        x=self.Conv(x)
        x=x.view(x.size()[0],-1)
        x=self.Fc(x)
        return x



def data_analyze(dataset):
    #数据分布
    fig1=plt.figure()
    targets=dataset.targets
    plt.hist(targets,bins=10,rwidth=0.8)
    plt.title('dataset histogram')
    plt.xlabel('class_id')
    plt.ylabel('class_num')
    #图片抽样查看
    fig2=plt.figure()
    images=dataset.data[:20]
    for i in np.arange(1,21):
        plt.subplot(5,4,i)
        plt.text(10,10,'{}'.format(targets[i-1]),fontsize=20,color='g')
        plt.imshow(images[i-1])
    fig2.suptitle('Images')
    plt.show()


viz = Visdom(server='http://[::1]', port=8097,env='Cifar Loss')
# viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
# viz.line([0.], [0.], win='test_acc', opts=dict(title='test acc'))
if __name__ == '__main__':
    net = CifarNet()
    writer = SummaryWriter( comment="myresnet")
    #绘制网络框图
    with writer:
        writer.add_graph(net, (torch.rand(1,3,32, 32),))
    optimizer = torch.optim.SGD(net.parameters(), lr=LR,momentum=0.9)
    loss_func = nn.CrossEntropyLoss()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # 随机获取训练图片
    for epoch in range(EPOCH):
        Loss=[]
        #训练损失visdom可视化
        # train_win='train{}_loss'.format(epoch)
        # viz.line([0.], [0.], win=train_win, opts=dict(title=train_win))
        for step,(batch_x,batch_y) in enumerate(trainloader):
            batch_x,batch_y=batch_x.to(device),batch_y.to(device)
            # print(batch_x.size())
            out=net(batch_x)
            #计算损失时，直接用非one-hot的展开与准确标签做计算，标签也不是one-hot的，就直接是一个数
            loss=loss_func(out,batch_y)
            optimizer.zero_grad()
            loss.backward()
            Loss.append(loss.data)
            optimizer.step()
            # viz.line([loss.item()], [step], win=train_win, update='append')
            if step%500==0:
                print('epoch:', epoch, 'step:', step, 'loss:{:.3f}'.format(loss.data))
                writer.add_scalar('Train', loss, step)

        # test_win = 'test{}_acc'.format(epoch)
        # viz.line([0.], [0.], win=test_win, opts=dict(title=test_win))
        total=0
        correct=0
        #TEST
        for step1, (batch_x, batch_y) in enumerate(testloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_x=net(batch_x)
            _, predicted = torch.max(pred_x.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            acc=correct/total
            # viz.line([acc], [step1], win=test_win, update='append')
            writer.add_scalar('Test', acc, step1)

    # data_analyze(trainset)