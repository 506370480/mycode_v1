import torch.nn as nn
import torch


class ConvModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((1, 2)),
        )

    def forward(self, x):
        y = self.layers(x)
        return y


class CNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=1408):
        super().__init__()
        self.layers = nn.Sequential(
            ConvModel(in_dim),
            nn.Flatten(),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)



if __name__ == '__main__':
    model = ConvModel(2)
    x = torch.randn((128, 2, 1, 180))
    y = model(x)
    print(y.shape)
