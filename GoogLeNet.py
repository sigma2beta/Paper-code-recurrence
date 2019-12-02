import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torch as t
import sys
from utils import train
sys.path.append('..')


# 定义一个卷积加relu激活函数和一个batchnorm作为一个基本的层结构
# torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中
def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(True)
    )
    return layer


# 定义inception模块
class Inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(Inception, self).__init__()
        # 第一条路线1*1 conv
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        # 第二条路线1*1+3*3
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)  # 3x3后输入channel变为1x1卷积之后的out2_1
        )

        # 第三条路线1*1+5*5
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

        # 第四条路线maxpool
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output


# 模型搭建好后，设置一个虚拟的输入检测模型是否存在问题
# test_net = Inception(3, 64, 48, 64, 64, 96, 32)
# test_x = Variable(torch.zeros(1, 3, 96, 96))
# print("input shape:{} x {} x {}".format(test_x.shape[1], test_x.shape[2], test_x.shape[3]))
# test_y = test_net(test_x)
# print("output shape:{} x {} x {}".format(test_y.shape[1], test_y.shape[2], test_y.shape[3]))


class GoogLeNet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(GoogLeNet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channel=64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )

        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )

        self.block3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )

        self.block4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )

        self.block5 = nn.Sequential(
           Inception(832, 256, 160, 320, 32, 128, 128),
           Inception(832, 384, 192, 384, 48, 128, 128),
           nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

# 测试GoogLeNet的搭建是否成功
#rgb = t.randn(1, 3, 256, 256)
#net = GoogLeNet(3, 10, verbose=True)
#out = net(rgb)
#print(out.shape)

#test_net2 = GoogLeNet(3, 10, True)
#test_x2 = Variable(torch.zeros(1, 3, 358, 480))
#print("input shape:{} x {} x {}".format(test_x2.shape[1], test_x2.shape[2], test_x2.shape[3]))
#test_y2 = test_net2(test_x2)
#print("output shape:{} x {}".format(test_y2.shape[0], test_y2.shape[1]))


def data_tf(x):
    x = x.resize((96, 96), 2)  # 将图片放大到 96 x 96
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化，这个技巧之后会讲到
    x = x.transpose((2, 0, 1))  # 将 channel 放到第一维，只是 pytorch 要求的输入方式
    x = torch.from_numpy(x)
    return x


train_set = CIFAR10('./data', train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10('./data', train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

net = GoogLeNet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
train(net, train_data, test_data, optimizer, criterion)