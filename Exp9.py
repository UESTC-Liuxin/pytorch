import torch
import os
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

#get win/linux path of dataset
root=os.getcwd()
data_root=os.path.join(root,'DATA')
# Mnist digits dataset

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root=data_root,    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=True,          # 没下载就下载, 下载了就不用再下了
)
# print(train_data.train_data.size())                 # (60000, 28, 28)
# print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
# for step,(b_x,b_y) in enumerate(train_loader):
#     print(b_x.dim())
#     aaa
# pick 2000 samples to speed up testing
val_data = torchvision.datasets.MNIST(
    root=data_root,
    train=False
)
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)

val_x = torch.unsqueeze(val_data.data, dim=1).type(torch.FloatTensor)[:2000]/255
# print(test_data.data.size())
val_y = val_data.targets[:2000]
if torch.cuda.is_available():
   val_x= val_x.cuda()
   # val_y = val_y.cuda()

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

def save_model(model=None):
    if model:
        model_path = os.path.join(root, 'model','Exp9.pth')
        #保存模型的参数,内存小，速度快
        torch.save(model.state_dict(),model_path)

net=Net()
#opt
optimizer=torch.optim.Adam(net.parameters(),lr=LR)
#loss
loss_func=  nn.CrossEntropyLoss()

if torch.cuda.is_available():
    torch.nn.DataParallel(net,device_ids=[0]).cuda()

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