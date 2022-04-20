import torch
from torch import nn, Tensor
from torch.nn import functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, softplus_at_end=False):
        super().__init__()
        self.num_layers = num_layers
        self.softplus_at_end = softplus_at_end
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        if self.softplus_at_end:
            x = F.softplus(x)

        return x
