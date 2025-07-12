import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
torch.set_default_dtype(torch.float64)

def sampled_pts_grid_torch(N, domain):
    d = domain.size(0)
    N = int(N)
    axes = [torch.linspace(domain[i, 0], domain[i, 1], N) for i in range(d)]
    grid = torch.meshgrid(axes, indexing='ij')
    grid_points = torch.stack(grid, dim=-1).reshape(-1, d)
    print('shape',grid_points.shape)
    return grid_points

def sampled_pts_rdm_torch(N, domain):
    torch.manual_seed(45)
    d = domain.size(0)
    X_domain = torch.empty(N, d)
    for i in range(d):
        X_domain[:, i] = torch.rand(N) * (domain[i, 1] - domain[i, 0]) + domain[i, 0]
    return X_domain

def sampled_points_fun(N, domain=torch.tensor([[0.0, 1.0], [0.0, 1.0]])):
    def function(x, y):
        return -(torch.sin(2 * torch.pi * x) + torch.sin(2 * torch.pi * y)+torch.cos(4 * torch.pi * x))
    X = sampled_pts_rdm_torch(N, domain)
    function_values = function(X[:, 0], X[:, 1])
    print(function_values.shape)
    plot_sampled_points_separate_torch1(X)
    return X, function_values

def save_tensor(filename, data):
    torch.save(data, filename)

def load_tensor(filename):
    return torch.load(filename)

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
    plt.savefig('afig6.jpg', format='jpg', dpi=300)
    plt.show()


def sampled_points_fun_observe_torch(N, k, domain=torch.tensor([[0., 1.], [0., 1.]])):
    torch.manual_seed(45)
    observed = int(N * N / k / k)
    function = lambda x, y: -(torch.sin(2 * torch.pi * x) + torch.sin(2 * torch.pi * y) + torch.cos(4 * torch.pi * x))
    X = sampled_pts_grid_torch(N, domain)
    selection_mask = torch.zeros(len(X), dtype=torch.bool)
    indices = torch.randperm(len(X))[:observed]
    selection_mask[indices] = True
    observepts = X[selection_mask]
    function_values = function(X[:, 0], X[:, 1])
    filename = 'm_values.pt'
    if not os.path.exists(filename):
        from reference import main
        m_values = main(N)
        save_tensor(filename, m_values)
    else:
        m_values = load_tensor(filename)
    plot_sampled_points_separate_torch(X, observepts, function_values, m_values)
    plot_function_values(X, function_values)

    return X, function_values, m_values, selection_mask




def plot_sampled_points_separate_torch(X, observepts, function_values, m_values):
    X_np = X.numpy()
    observepts_np = observepts.numpy()
    m_values_np = m_values.numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_np[:, 0], X_np[:, 1], m_values_np, c=m_values_np, cmap='viridis')
    plt.colorbar(scatter)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('m(x, y)', fontsize=14)
    plt.tight_layout()
    plt.show()

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
    plt.savefig('afig7.jpg', format='jpg', dpi=300)
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
    plt.savefig('afig8.jpg', format='jpg', dpi=300)
    plt.show()