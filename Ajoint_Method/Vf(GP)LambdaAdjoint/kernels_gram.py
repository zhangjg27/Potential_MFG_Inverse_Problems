import torch
import torch.nn as nn


class PeriodicKernel(nn.Module):
    def __init__(self):
        super(PeriodicKernel, self).__init__()
    def forward(self, x, y, sigma):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        diff = x - y
        diff = torch.cos(2*torch.pi * diff)-1
        dist_sq = (diff).sum(dim=2)
        return torch.exp(dist_sq / (sigma ** 2))
    def kappa(self, x, y, sigma):
        return self.forward(x, y, sigma)

class MaternKernel(nn.Module):
    def __init__(self, sigma=1.0, rho=1.0, epsilon=1e-6):
        super(MaternKernel, self).__init__()
        self.sigma = sigma
        self.rho = rho
        self.epsilon = epsilon

    def kappa(self, x, y):
        x_expanded = x.unsqueeze(1)
        y_expanded = y.unsqueeze(0)
        d = torch.sqrt(((x_expanded - y_expanded) ** 2).sum(dim=2) + self.epsilon)
        sqrt_5_d_rho = torch.sqrt(torch.tensor(5.0)) * d / self.rho
        exp_component = torch.exp(-1*sqrt_5_d_rho)
        poly_component = 1 + sqrt_5_d_rho + (5 * d ** 2) / (3 * self.rho ** 2)

        return self.sigma ** 2 * poly_component * exp_component

class MaternKernelD(nn.Module):
    def __init__(self, sigma=1.0, rho=1.0, epsilon=1e-6):
        super(MaternKernelD, self).__init__()
        self.sigma = sigma
        self.rho = rho
        self.epsilon = epsilon

    def kappa(self, x, r):
        x_expanded = x.unsqueeze(1)
        r_expanded = r.unsqueeze(0)
        diff= x_expanded - r_expanded
        d_squared = ((x_expanded - r_expanded) ** 2).sum(dim=2) + self.epsilon
        d = torch.sqrt(d_squared)
        sqrt_5 = torch.sqrt(torch.tensor(5.0))
        sqrt_5_d_rho = sqrt_5 * d / self.rho
        exp_component = torch.exp(-1*sqrt_5_d_rho)
        factor1 = -sqrt_5 / self.rho * (1 + sqrt_5_d_rho + 5 * d_squared / (3 * self.rho ** 2))
        factor2 = (sqrt_5 / self.rho + 10 * d / (3 * self.rho ** 2))
        derivative = self.sigma ** 2 * (factor1 + factor2) * exp_component * (diff.squeeze(2) / d)

        return derivative

class MaternKernelDD(nn.Module):
    def __init__(self, sigma=1.0, rho=1.0, epsilon=1e-6):
        super(MaternKernelDD, self).__init__()
        self.sigma = sigma
        self.rho = rho
        self.epsilon = epsilon

    def kappa(self, x, r):
        x_expanded = x.unsqueeze(1)
        r_expanded = r.unsqueeze(0)
        diff = x_expanded - r_expanded
        d_squared = ((diff) ** 2).sum(dim=2) + self.epsilon
        d = torch.sqrt(d_squared)
        sqrt_5 = torch.sqrt(torch.tensor(5.0))
        sqrt_5_d_rho = sqrt_5 * d / self.rho
        exp_component = torch.exp(-1*sqrt_5_d_rho)
        term1 = (-sqrt_5 / self.rho * (1 + sqrt_5_d_rho + 5 * d_squared / (3 * self.rho ** 2)) +
                (sqrt_5 / self.rho + 10 * d / (3 * self.rho ** 2))) * (-sqrt_5 / self.rho)
        term2 = (-sqrt_5 / self.rho) * (sqrt_5 / self.rho+10 * d / (3 * self.rho ** 2)) + (10 / (3 * self.rho ** 2))
        part1 = self.sigma ** 2 * (term1 + term2) * exp_component
        part2_terms = (-sqrt_5 / self.rho * (1 + sqrt_5_d_rho + 5 * d_squared / (3 * self.rho ** 2)) +
                      (sqrt_5 / self.rho + 10 * d / (3 * self.rho ** 2)))
        part2 = self.sigma ** 2 * part2_terms * exp_component * (1 / d - (diff.squeeze(2)) ** 2 / d.pow(3))
        second_derivative = part1
        return second_derivative


def gram_matrix_assembly(X, Y, kernel_type=None, kernel_parameter=0.6, sigma=1.0, rho=1.0, epsilon=1e-3):
    if kernel_type is None:
        kernel = PeriodicKernel()
    elif kernel_type == 'matern':
        kernel = MaternKernel(sigma=sigma, rho=rho, epsilon=epsilon)
    elif kernel_type == 'maternd':
        kernel = MaternKernelD(sigma=sigma, rho=rho, epsilon=epsilon)
    elif kernel_type == 'materndd':
        kernel = MaternKernelDD(sigma=sigma, rho=rho, epsilon=epsilon)
    else:
        raise ValueError("Unsupported kernel type specified!")
    if kernel_type is None:
       return kernel.kappa(X, Y, kernel_parameter)
    else:
        return kernel.kappa(X, Y)