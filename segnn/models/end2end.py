import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class CAN(nn.Module):
    def __init__(self, channels=32, num_classes=7):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(3, channels, (3, 3), padding=1, dilation=1),
            nn.LeakyReLU())
        self.c2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3),  padding=2, dilation=2),
            nn.LeakyReLU())
        self.c3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3),  padding=4, dilation=4),
            nn.LeakyReLU())
        self.c4 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3),  padding=8, dilation=8),
            nn.LeakyReLU())
        self.c5 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3),  padding=1, dilation=1),
            nn.LeakyReLU())
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = x.permute([0, 2, 3, 1])
        x = self.fc(x)
        x = x.permute([0, 3, 1, 2])
        return x
