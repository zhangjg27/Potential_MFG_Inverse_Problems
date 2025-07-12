import torch
from torch import nn
class PeriodicKernel(nn.Module):
    def __init__(self):
        super(PeriodicKernel, self).__init__()

    def forward(self, x, y, sigma):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        diff = x - y
        diff = torch.cos(2*torch.pi * diff)-1
        dist_sq = (diff ).sum(dim=2)
        return torch.exp(dist_sq / (sigma ** 2))
    def kappa(self, x, y, sigma):
        return self.forward(x, y, sigma)


class GaussianKernel(nn.Module):
    def __init__(self):
        super(GaussianKernel, self).__init__()

    def forward(self, x, y, sigma):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    def kappa(self, x, y, sigma):
        return self.forward(x, y, sigma)

class GaussianDKernel(nn.Module):
    def __init__(self):
        super(GaussianDKernel, self).__init__()

    def derivative(self, x, y, sigma):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        kernel = torch.exp(-dist_sq / (2 * sigma ** 2))
        diff=diff.squeeze(2)
        grad_kernel = -diff * kernel / sigma ** 2
        return grad_kernel

    def kappa(self, x, y, sigma):
        return self.derivative(x, y, sigma)


class GaussianDDKernel(nn.Module):
    def __init__(self):
        super(GaussianDDKernel, self).__init__()

    def second_derivative(self, x, y, sigma):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        kernel = torch.exp(-dist_sq / (2 * sigma ** 2))
        term1 = 1 / sigma ** 2
        term2 = dist_sq / sigma ** 4
        second_grad_kernel = (-term1 + term2) * kernel
        return second_grad_kernel

    def kappa(self, x, y, sigma):
        return self.second_derivative(x, y, sigma)


def gram_matrix_assembly(X,Y, kernel=None, kernel_parameter=0.9):
    if kernel is None:
        kernel = PeriodicKernel()  # 实例化周期核
    elif kernel == 'gaussian':
         kernel = GaussianKernel()
    elif kernel == 'gaussiand':
         kernel = GaussianDKernel()
    elif kernel == 'gaussiandd':
         kernel = GaussianDDKernel()
    gram_matrix = kernel.kappa(X, Y, kernel_parameter)
    return gram_matrix

