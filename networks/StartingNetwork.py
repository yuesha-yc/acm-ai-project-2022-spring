import torch
import torch.nn as nn

class Residual(torch.nn.Module):
    """
    Residual block.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                )
            )

    def forward(self, x):
        # Route 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Route 2
        if hasattr(self, "shortcut"):
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut

        # ReLU activation
        out = self.relu(out)

        return out


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        # Input tensor size = 224x224x3

        # Conv2D Input-Output Size:
        # Hout = [(Hin - kernel_size + 2*padding) / stride] + 1
        # Wout = [(Win - kernel_size + 2*padding) / stride] + 1

        # Stem Layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        
        # input size = 112x112x64
        # output size = 56x56x64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Layers
        self.res1 = Residual(64, 128, stride=2)
        self.res2 = Residual(128, 256, stride=2)
        self.res3 = Residual(256, 512, stride=2)
        self.res4 = Residual(512, 1024, stride=2)

        # Global average pooling
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        # Fully connected layer
        self.fc = nn.Linear(1024, 1)
        # Softmax
        self.softmax = nn.Softmax()


    def forward(self, x):
        # Stem Layers
        x = self.conv1(x)
        x = self.maxpool(x)
        # Residual Layers
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        # Global average pooling
        print(x.size())
        x = self.avgpool(x)
        print(x.size())
        x = self.fc(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    print("Starting Network")
    # With square kernels and equal stride
    # non-square kernels and unequal stride and with padding
    # m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

    # m = nn.Conv3d(16, 33, 3, stride=2)
    # input = torch.randn(20, 16, 10, 50, 100)
    # output = m(input)
    # print(output.size())

    # m = nn.Conv1d(16, 33, 3, stride=2)
    # input = torch.randn(20, 16, 50)
    # output = m(input)
    # print(output.size())

    m = nn.Conv2d(3, 64, kernel_size=7, stride=2)
    input = torch.randn(1, 3, 224, 224)
    output = m(input)
    print(output.size())
    print(output)