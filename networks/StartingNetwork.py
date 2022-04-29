import torch
import torch.nn as nn

class Residual(torch.nn.Module):
    """
    Residual block.
    """

    def __init__(self, in_channels, out_channels, use_1x1Conv=False, stride=1):
        super(Residual, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()

        # Shortcut
        if use_1x1Conv:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut:
            x = self.shortcut(x)
        out += x
        out = self.relu(out)

        return out


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        # batch size = 32, channels = 3, height = 224, width = 224
        # Input tensor size = 32x224x224x3

        # Conv2D Input-Output Size:
        # Hout = [(Hin - kernel_size + 2*padding) / stride] + 1
        # Wout = [(Win - kernel_size + 2*padding) / stride] + 1

        # Stem Layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        
        # input size = 112x112x64
        # output size = 56x56x64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Layers
        # if (stride != 1) or (self.in_channels != out_channels): use 1x1 conv shortcut
        self.res1 = Residual(64, 64, stride=1) 
        self.res2 = Residual(64, 128, stride=2, use_1x1Conv=True)
        self.res3 = Residual(128, 256, stride=2, use_1x1Conv=True)
        self.res4 = Residual(256, 512, stride=2, use_1x1Conv=True)
        
        # batch size = 32, channels = 512, height = 7, width = 7
        # Global average pooling
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

        # batch size = 32, channels = 512, height = 1, width = 1
        # Fully connected layer
        self.flatten = nn.Flatten()

        # batch size = 32, channels = 512
        # reshape
        self.fc = nn.Linear(512, 5005)

        # batch size = 32, channels = 5005
        # Softmax
        self.softmax = nn.Softmax()


    def forward(self, x):
        # Stem Layers
        # print("Before Stem", x.size())
        x = self.conv1(x)
        x = self.maxpool(x)

        # Residual Layers
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        # Global average pooling
        # print("After Res", x.size())
        x = self.avgpool(x)
        # print("After AvgPool", x.size())
        x = self.flatten(x)
        # print("After flatten", x.size())
        x = self.fc(x)
        # print("After FC", x.size())
        x = self.softmax(x)
        # print("After Softmax", x.size())
        return x