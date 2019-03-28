import torch
import torch.nn as nn


class ZeroNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Linear(321, 321)

    def forward(self, x):
        shape = [*x.shape]
        shape[1] = 7
        x = torch.zeros(shape)
        x = self.A(x)
        return x
