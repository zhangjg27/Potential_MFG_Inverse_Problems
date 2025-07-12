import numpy as onp
from numpy import random, sin, pi,exp
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_function_values(X, function_values):
    grid_x, grid_y = onp.mgrid[onp.min(X[:, 0]):onp.max(X[:, 0]):100j, onp.min(X[:, 1]):onp.max(X[:, 1]):100j]
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
    plt.savefig('afig6V.jpg', format='jpg', dpi=300)
    plt.show()



def plot_sampled_points_separate(X, observepts, function_values, m_values):
    X_np = X
    observepts_np = observepts
    grid_x, grid_y = onp.mgrid[onp.min(X[:, 0]):onp.max(X[:, 0]):100j, onp.min(X[:, 1]):onp.max(X[:, 1]):100j]
    grid_z = griddata(X[:, :2], m_values, (grid_x, grid_y), method='cubic')

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
    plt.savefig('afig7V.jpg', format='jpg', dpi=300)
    plt.show()

def plot_sampled_points_separate1(X):

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    ax3.scatter(X[:, 0], X[:, 1], color='blue', label=r"Observation Points of $V^o(x,y)$")
    plt.title(r"2D Distribution of Observation Points for $V^o(x,y)$", fontsize=14)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('afig8V.jpg', format='jpg', dpi=300)
    plt.show()



def sampled_pts_grid(N, domain):
    d = domain.shape[0]
    axes = [onp.linspace(domain[i, 0], domain[i, 1], N) for i in range(d)]
    grid = onp.meshgrid(*axes, indexing='ij')
    grid_points = onp.stack(grid, axis=-1).reshape(-1, d)
    return grid_points




def sampled_points_fun_observe(N,k, domain=onp.array([[0, 1], [0, 1]])):
    onp.random.seed(40)
    observed=int(N*N/k/k)
    function = lambda x, y: -sin(2 * pi * x) - sin(2 * pi * y)
    X = sampled_pts_grid(N, domain)
    selection_mask = onp.zeros(len(X), dtype=bool)
    indices = onp.random.choice(len(X), observed, replace=False)
    selection_mask[indices] = True
    observepts = X[selection_mask]
    function_values = function(X[:, 0], X[:, 1])
    lambda_val, _ = dblquad(lambda x, y: exp(sin(2 * pi * x) + sin(2 * pi * y)),
                            0, 1,
                            lambda x: 0,
                            lambda x: 1)
    m_values = exp(sin(2 * pi * X[:, 0]) + sin(2 * pi * X[:, 1]) - onp.log(lambda_val))
    #plot_sampled_points_separate(X, observepts,function_values, m_values)
    #plot_function_values(X, function_values)
    return X, function_values, m_values,selection_mask

def sampled_pts_rdm(N, domain):
    onp.random.seed(40)
    d = domain.shape[0]
    X_domain = random.uniform(domain[0, 0], domain[0, 1], (N, 1))
    for i in range(1, int(d)):
        X_domain = onp.concatenate((X_domain, random.uniform(domain[i, 0], domain[i, 1], (N, 1))),
                                   axis=1)
    return X_domain



def sampled_points_fun(N, domain=onp.array([[0, 1], [0, 1]])):
    function = lambda x, y: (-sin(2 * pi * x) - sin(2 * pi * y))
    X = sampled_pts_rdm(N, domain)
    function_values = function(X[:, 0], X[:, 1])
    #plot_sampled_points_separate1(X)
    return X, function_values