import torch
import torch.nn as nn

# Residual Block
class Residual(torch.nn.Module):
    """
    Residual block.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, "shortcut"):
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.relu(out)

        return out


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(224 * 224 * 3, 1)
        # add cnn layers here
        self.conv1 = nn.Sequential(
            nn.Conv3d(224 * 224 * 3, 224 * 244 * 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(3),
            nn.ReLU()
        )
        

        
        self.sigmoid = nn.Sigmoid()
        



    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    # With square kernels and equal stride
    # non-square kernels and unequal stride and with padding
    # m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

    m = nn.Conv3d(16, 33, 3, stride=2)
    input = torch.randn(20, 16, 10, 50, 100)
    output = m(input)
    print(output.size())

    m = nn.Conv1d(16, 33, 3, stride=2)
    input = torch.randn(20, 16, 50)
    output = m(input)
    print(output.size())