import torch
from torch import nn


class ExampleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)
