from torch import nn


class LCAL(nn.Module):
    """"""
    def __init__(self, Dulbrn):
        super(LCAL, self).__init__()

        self.dulbrn = Dulbrn

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.dulbrn, (3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(self.dulbrn, self.dulbrn * 2, (3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out_channels = [self.dulbrn, self.dulbrn * 2]

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        features = [f1, f2]
        return features