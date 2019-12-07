import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import os
import random

# 可再现
# torch.manual_seed(1)

EPOCH = 3
BATCH_SIZE = 64
# 学习率
LR = 0.001
# 是否需要下载
DOWNLOAD_MNIST = False
# 数据大小
n_train = 64
n_test = 16

# 检查是否已经下载MNIST
if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

# 训练集
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
# 训练集加载器，每个batch大小是(BATCH_SIZE, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
# 测试集
test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=transforms.Compose([
        # transforms.RandomAffine(30),
        transforms.ToTensor()])
)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)


# 数据集大小
# (60000, 28, 28)
# print(train_data.train_data.size())
# (60000)
# print(train_data.train_labels.size())
# (10000, 28, 28)
# print(test_data.test_data.size())
# (10000)
# print(test_data.test_labels.size())


# 显示一张图片
# plt.imshow(test_data.test_data[0], cmap='gray')
# plt.show()


class Net(torch.nn.Module):
    '''
    两层神经网络，输入层为28*28，即每张图片大小，隐藏层包含hidden个神经元和一个ReLU层，
    输出层为10个（即十个数字分别的权值），采取全连接
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        '''
        变量在神经网络中的传递
        :param x: 输入数据
        :return: 输出权值
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


net = Net()
# 采取Adam作为优化器
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
EntropyLoss = torch.nn.CrossEntropyLoss()

# 训练循环
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        if step > n_train: break

        output = net(b_x)
        loss = EntropyLoss(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每若干步进行一次test
        if step % 5 == 0:
            # 正确数量
            correctNum = 0
            # 错误数量
            totalNum = 0
            # 分batch测试
            for tn, (t_x, t_y) in enumerate(test_loader):
                if tn > n_test: break
                test_output = net(t_x)
                # 得到分类结果
                pred_y = torch.max(test_output, 1)[1].numpy()
                correctNum += sum(pred_y == t_y.detach().numpy())
                totalNum += t_y.size(0)

            accuracy = correctNum / totalNum
            print('Epoch:', epoch, '| train loss: %10.4f' % loss.detach().numpy(),
                  '| test accuracy: %.3f' % accuracy)

torch.save(net, './cnn.pt')
