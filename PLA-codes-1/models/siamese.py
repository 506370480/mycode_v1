import torch.nn as nn
import torch
import torch.nn.functional as F


class SiameseNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        y1 = self.layers(x1)
        y2 = self.layers(x2)
        mid = abs(y1-y2)
        final = self.final(mid)
        return final



class SiameseNetV2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(64, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((1, 2)),
            nn.Flatten(),
            nn.Linear(720, 16),
            nn.ReLU(True)
        )
        self.final = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        y1 = self.layers(x1)
        y2 = self.layers(x2)
        mid = abs(y1-y2)
        final = self.final(mid)
        return final


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


if __name__ == '__main__':
    model = SiameseNetV2(1)
    x = torch.ones((1, 1, 1, 360))
    y = model(x, x)
    