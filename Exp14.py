import os
import time
import torch
import torchvision
from torchvision.models import densenet121
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from resnet import ResNet18

# vis = visdom.Visdom()
# vis.text('Hello, world!')
# vis.image(np.ones((3, 10, 10)))
#get win/linux path of dataset
root=os.getcwd()
data_root=os.path.join(root,'DATA')

# Hyper Parameters
EPOCH = 300           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 128
# LR = BATCH_SIZE*0.00125          # 学习率
LR=0.1

transform_train = transforms.Compose(
    [
     transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
     transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
     # transforms.CenterCrop
     ] #3个通道的平均值和方差都取0.5
)

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
     # transforms.CenterCrop
     ] #3个通道的平均值和方差都取0.5
)
#训练数据集
trainset =torchvision.datasets.CIFAR10(
    root=data_root,
    train=True,
    download=True,
    transform=transform_train
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
    transform=transform_test
)
testloader=Data.DataLoader(
    dataset=testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



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

def adjust_learning_rate(optimizer,lr):
    lr=lr*0.1
    for param_group in optimizer.param_groups:
        param_group['lr']=lr


if __name__ == '__main__':
    net = densenet121(drop_rate=0.5,num_classes=10)
    epoch_list=[130,150,200,250]
    lr_index=0
    TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = 'logs/densenet/' + TIMESTAMP
    writer = SummaryWriter(log_dir)
    #绘制网络框图
    with writer:
        writer.add_graph(net, (torch.rand(1,3,32, 32),))
    optimizer = torch.optim.SGD(net.parameters(), lr=LR,momentum=0.9,weight_decay=5e-4)
    # lr_sch=torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)
    loss_func = nn.CrossEntropyLoss()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # 随机获取训练图片
    for epoch in range(EPOCH):
        Loss=[]
        train_total=0
        train_correct=0
        net.train()
        # read the lr of this epoach
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if epoch>=epoch_list[lr_index]:
            lr_index+=1
            adjust_learning_rate(optimizer,lr)

        for step,(batch_x,batch_y) in enumerate(trainloader):
            batch_x,batch_y=batch_x.to(device),batch_y.to(device)
            # print(batch_x.size())
            out=net(batch_x)
            #record train_set acc
            _, predicted = torch.max(out.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            train_acc=train_correct/train_total
            # 计算损失时，直接用非one-hot的展开与准确标签做计算，标签也不是one-hot的，就直接是一个数
            loss=loss_func(out,batch_y)
            optimizer.zero_grad()
            loss.backward()
            Loss.append(loss.data)
            optimizer.step()
            # viz.line([loss.item()], [step], win=train_win, update='append')
            if step%50==0:
                print('epoch:', epoch, 'step:', step, 'lr:{:.5f}'.format(lr),
                      'loss:{:.3f}'.format(loss.data))

        writer.add_scalar('Train_loss', sum(Loss)/len(Loss), epoch)
        writer.add_scalar('Train_acc', train_acc, epoch)
        writer.add_scalar('learning rate', lr, epoch)
        # lr_sch.step()

        # test_win = 'test{}_acc'.format(epoch)
        # viz.line([0.], [0.], win=test_win, opts=dict(title=test_win))
        test_Loss = []
        test_total=0
        test_correct=0
        #TEST
        with torch.no_grad():
            net.eval()
            for step1, (batch_x, batch_y) in enumerate(testloader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred_x=net(batch_x)
                _, predicted = torch.max(pred_x.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                test_acc=test_correct/test_total
                loss=loss_func(pred_x, batch_y)
                test_Loss.append(loss.data)
            # viz.line([acc], [step1], win=test_win, update='append')
        writer.add_scalar('Test_acc', test_acc, epoch)
        writer.add_scalar('Test_loss', sum(test_Loss) / len(test_Loss), epoch)

    # data_analyze(trainset)