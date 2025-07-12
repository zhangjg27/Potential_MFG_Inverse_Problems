import torch
from torch import nn

class PeriodicKernel(nn.Module):
    def __init__(self):
        super(PeriodicKernel, self).__init__()
    def forward(self, x, y, sigma):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        return torch.exp(-dist_sq / (2 * sigma**2))
    def kappa(self, x, y, sigma):
        return self.forward(x, y, sigma)




def gram_matrix_assembly(X, kernel=None, kernel_parameter=0.6):
    if kernel is None:
        kernel = PeriodicKernel()
    gram_matrix = kernel.kappa(X, X, kernel_parameter)
    return gram_matrix