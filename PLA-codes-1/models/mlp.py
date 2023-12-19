import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super().__init__()
        layer_sizes=[in_dim, 256, 512, 512, out_dim]
        # layer_sizes=[in_dim, 256, out_dim]
        self.layers = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.layers.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.layers.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, x):
        y = self.layers(x)
        return torch.sigmoid(y)


if __name__ == '__main__':
    model = MLP(360)
    print(list(model.modules()))