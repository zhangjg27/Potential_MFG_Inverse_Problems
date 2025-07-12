import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def sampled_pts_grid_torch(N, domain):
    d = domain.size(0)
    N = int(N)
    axes = [torch.linspace(domain[i, 0], domain[i, 1], N) for i in range(d)]
    grid = torch.meshgrid(axes, indexing='ij')
    grid_points = torch.stack(grid, dim=-1).reshape(-1, d)
    return grid_points

def sampled_pts_rdm_torch(N, domain):
    torch.manual_seed(46)
    d = domain.size(0)
    X_domain = torch.empty(N, d)
    for i in range(d):
        X_domain[:, i] = torch.rand(N) * (domain[i, 1] - domain[i, 0]) + domain[i, 0]
    return X_domain

def sampled_points_fun(N, domain=torch.tensor([[0.0, 1.0], [0.0, 1.0]])):
    def function(x, y):
        return -0.5 * (torch.sin(2 * torch.pi * x) + torch.sin(2 * torch.pi * y))
    X = sampled_pts_rdm_torch(N, domain)
    function_values = function(X[:, 0], X[:, 1])
    plot_sampled_points_separate_torch1(X)
    return X, function_values

def sampled_points_fun_observe_torch(N, k, alpha,lambda_val, domain=torch.tensor([[0., 1.], [0., 1.]])):
    torch.manual_seed(46)
    observed = int(N * N / k / k)
    function = lambda x, y: -0.5 * (torch.sin(2 * torch.pi * x) + torch.sin(2 * torch.pi * y))
    X = sampled_pts_grid_torch(N, domain)
    selection_mask = torch.zeros(len(X), dtype=torch.bool)
    indices = torch.randperm(len(X))[:observed]
    selection_mask[indices] = True
    observepts = X[selection_mask]
    m_exponent = 1 / alpha
    function_values = function(X[:, 0], X[:, 1])
    base =0.5 *(torch.sin(2 * torch.pi * X[:, 0]) + torch.sin(2 * torch.pi * X[:, 1])) - lambda_val
    m_values = torch.pow(torch.clamp(base, min=0), m_exponent)
    plot_sampled_points_separate_torch(X, observepts, function_values, m_values)
    plot_function_values(X, function_values)
    return X, function_values, m_values, selection_mask

def plot_function_values(X, function_values):
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(function_values, torch.Tensor):
        function_values = function_values.numpy()
    grid_x, grid_y = np.mgrid[np.min(X[:, 0]):np.max(X[:, 0]):100j, np.min(X[:, 1]):np.max(X[:, 1]):100j]
    grid_z = griddata(X[:, :2], function_values, (grid_x, grid_y), method='cubic')
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('V(x,y)', fontsize=14)
    plt.title('3D Surface Plot of Real V(x,y)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig('afig6Vfmatern.jpg', format='jpg', dpi=300)
    plt.show()


def plot_sampled_points_separate_torch(X, observepts, function_values, m_values):
    X_np = X.numpy()
    observepts_np = observepts.numpy()
    grid_x, grid_y = np.mgrid[np.min(X_np[:, 0]):np.max(X_np[:, 0]):100j, np.min(X_np[:, 1]):np.max(X_np[:, 1]):100j]
    grid_z = griddata(X_np[:, :2], m_values, (grid_x, grid_y), method='cubic')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')
    plt.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel("X", labelpad=10, fontsize=14)
    ax.set_ylabel("Y", labelpad=10, fontsize=14)
    ax.set_zlabel("m(x,y)", labelpad=10, fontsize=14)
    plt.title("3D Surface Plot of Real m(x,y)", fontsize=14)
    fig.tight_layout()
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(X_np[:, 0], X_np[:, 1], m_values_np, c=m_values_np, cmap='viridis')
    # plt.colorbar(scatter)
    # ax.set_xlabel('X', fontsize=14)
    # ax.set_ylabel('Y', fontsize=14)
    # ax.set_zlabel('m(x, y)', fontsize=14)
    # plt.tight_layout()
    # plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(X_np[:, 0], X_np[:, 1], color='blue', label='Sample Points')
    ax2.scatter(observepts_np[:, 0], observepts_np[:, 1], color='orange', label='Observation Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title("2D Distribution of Sample and Observation Points for $m(x,y)$", fontsize=13.5)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('afig7Vfmatern.jpg', format='jpg', dpi=300)
    plt.show()





def plot_sampled_points_separate_torch1(X):
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(X[:, 0], X[:, 1], color='blue', label=r"Observation Points of $V^o(x,y)$")
    plt.title(r"2D Distribution of Observation Points for $V^o(x,y)$", fontsize=14)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('afig8Vfmatern.jpg', format='jpg', dpi=300)
    plt.show()