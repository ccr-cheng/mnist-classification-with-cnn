import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

matplotlib.use('TkAgg')

import torch
from torch import nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms


class Net(torch.nn.Module):

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
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


# 测试集
test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=transforms.ToTensor()
)
test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)

# 数据集大小
# (10000, 28, 28)
# print(test_data.test_data.size())
# (10000)
# print(test_data.test_labels.size())


net = torch.load('./cnn.pt')
net.eval()

iterator = iter(test_loader)
t_x, t_y = next(iterator)


def ShowImg():
    global t_x, t_y, results

    t_x, t_y = next(iterator)
    plt.imshow(t_x[0, 0], cmap='gray')
    plt.title('{}'.format(t_y[0]), fontsize=40, color='red')
    plt.xticks([], [])
    plt.yticks([], [])
    canvas.draw()

    results.set('')
    num.set('')


def PredictNum():
    output = torch.softmax(net(t_x)[0], 0) * 100
    pred_y = torch.argmax(output)
    pred = '识别结果:\n'
    for i in range(10):
        pred += '{}:{:>8.2f}%\n'.format(i, output[i])

    global results, num
    results.set(pred)
    num.set('{}'.format(pred_y))

    return pred_y


def FindWrong():
    while True:
        ShowImg()
        pred_y = PredictNum()
        if pred_y != t_y: break


window = tk.Tk()
window.title('卷积神经网络手写数字识别')
window.geometry('900x550')
results = tk.StringVar()
num = tk.StringVar()

fig = plt.figure(figsize=(5, 5))
canvas = FigureCanvasTkAgg(fig, window)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

GenTest = tk.Button(window, text='点击随机抽取手写数字样本', font=10,
                    width=50, height=2, command=ShowImg).pack()
Predict = tk.Button(window, text='点击进行识别', font=10,
                    width=50, height=2, command=PredictNum).pack()
Find = tk.Button(window, text='点击找出一个识别错误的例子', font=10,
                 width=50, height=2, command=FindWrong).pack()
Result = tk.Label(window, textvariable=results, font=('Courier New', 18),
                  width=15, height=20).pack()
PredNum = tk.Label(window, textvariable=num, font=('Courier New', 50),
                   fg='blue').place(x=800, y=210)

window.mainloop()
