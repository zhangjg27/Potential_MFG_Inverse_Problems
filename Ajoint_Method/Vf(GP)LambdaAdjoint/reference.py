import numpy as np
from scipy.optimize import fsolve
import time
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch


# linear operator:div
def bmat1(N):
    Ntot=N**2
    h=1/N
    w1 = np.eye(Ntot)
    for i in range(Ntot):
        if i-N<0:
            w1[i][Ntot+i-N] = -1
        else:
            w1[i][i-N]=-1
    w2 = -np.eye(Ntot)
    for i in range(Ntot):
        if i+N>=Ntot:
            w2[i][i+N-Ntot] = 1
        else:
            w2[i][i+N] = 1

    w3 = np.eye(Ntot)
    counter = 1
    for i in range(Ntot):
        if counter==1:
            w3[i][i+N-1]=-1
        else:
            w3[i][i-1]=-1
        counter += 1
        if counter>N:
            counter=1
    w4 = -np.eye(Ntot)
    counter=1

    for i in range(Ntot):
        if counter==N:
            w4[i][i+1-N]=1
        else:
            w4[i][i+1]=1
        counter += 1
        if counter>N:
            counter=1

    return N*(np.block([w1,w2,w3,w4]))

def laplace_matrix(N):
    Ntot = N**2
    h = 1 / N
    L = 4*np.eye(Ntot)

    for i in range(Ntot):
        if i - N < 0:
            L[i][Ntot + i - N] = -1

        else:
            L[i][i - N] = -1


        if i + N >= Ntot:
            L[i][i + N - Ntot] = -1

        else:
            L[i][i + N] = -1


    counter = 1
    for i in range(Ntot):
        if counter == 1:
            L[i][i + N - 1] = -1

        else:
            L[i][i - 1] = -1

        counter += 1
        if counter > N:
            counter = 1

    counter = 1
    for i in range(Ntot):
        if counter == N:
            L[i][i + 1 - N] = -1
        else:
            L[i][i + 1] = -1

        counter += 1
        if counter > N:
            counter = 1

    return N**2*L

def prox1(tau, c, pichulitaold, m, w1, w2, w3, w4):
    k1 = np.max([0, w1])
    k2 = np.min([0, w2])
    k3 = np.max([0, w3])
    k4 = np.min([0, w4])
    f = lambda x: (x + tau * (x**3 - c) - m) * (x + tau) ** 2 - (tau / 2) * (
                k1 ** 2 + k2 ** 2 + k3 ** 2 + k4 ** 2)
    pichulitaold_new = fsolve(f, pichulitaold, xtol=1e-6)
    pichulitaold_n = pichulitaold_new
    msol = pichulitaold_new
    w1sol = pichulitaold_new * k1 / (pichulitaold_new + tau)
    w2sol = pichulitaold_new * k2 / (pichulitaold_new + tau)
    w3sol = pichulitaold_new * k3 / (pichulitaold_new + tau)
    w4sol = pichulitaold_new * k4 / (pichulitaold_new + tau)
    return [pichulitaold_n, msol, w1sol, w2sol, w3sol, w4sol]

# proximal operator of the primal function
def proxL2gen2dsdgomes1(tau,m,w,pichulitaold,N,c):
    w1 = w[:N]
    w2 = w[N:2*N]
    w3 = w[2*N:3*N]
    w4 = w[3*N:]
    t1 = time.time()
    re = [prox1(tau, c[i], pichulitaold[i], m[i], w1[i], w2[i], w3[i], w4[i]) for i in range(N)]
    t2 = time.time()
    print(t2-t1)
    pichulitaold_n = np.array([re[i][0] for i in range(N)]).flatten()
    msol = np.array([re[i][1] for i in range(N)]).flatten()
    w1sol = np.array([re[i][2] for i in range(N)]).flatten()
    w2sol = np.array([re[i][3] for i in range(N)]).flatten()
    w3sol = np.array([re[i][4] for i in range(N)]).flatten()
    w4sol = np.array([re[i][5] for i in range(N)]).flatten()

    wsol = np.block([w1sol, w2sol, w3sol, w4sol])
    return [msol,wsol,pichulitaold_n]

def main(N1):
    if isinstance(N1, torch.Tensor) and N1.numel() == 1:
        N1 = int(N1.item())
    else:
        raise ValueError("N1_tensor should be a single-element torch.Tensor")
    h = 1 / N1
    N = N1 * N1

    # CP algorithm parameter
    L = 1
    gamma = 0.05
    tau0 = 10 / L
    sigma0 = 0.1 / L
    errorac = []
    errorac2 = []
    tol = h ** 3
    error = 1
    X = np.linspace(0, 1, N1)
    Y = np.linspace(0, 1, N1)
    X, Y = np.meshgrid(X, Y, indexing='ij')
    sinX = np.sin(2 * np.pi * X)
    cosX = np.cos(4 * np.pi * X)
    sinY = np.sin(2 * np.pi * Y)
    c = (sinX + cosX + sinY)
    c = c.flatten()
    A = 0.1*laplace_matrix(N1)
    B = bmat1(N1)
    C = np.block([A,B])
    hm = np.block([(h**2)*np.ones([1,N]), np.zeros([1,4*N])])
    C = np.block([[C],[hm]])
    Q = np.matmul(C, np.transpose(C))
    n0 = np.zeros(N)
    v0 = np.zeros(4*N)
    m0 = np.ones(N)
    w0 = np.ones(4*N)
    m0b = m0
    w0b = w0
    pichulitaold = np.ones(N)
    counter = 0
    while(counter < 30 and error>tol):
        y = n0 + sigma0*m0b - sigma0*(np.ones(N))
        z = v0 + sigma0*w0b
        x1 = np.matmul(C, np.block([y,z]))
        x2 = np.linalg.solve(Q, x1)
        x3 = np.matmul(np.transpose(C), x2)
        n1 = x3[:N]
        v1 = x3[N:]
        print('update dual')

        #update prime variables
        ans = proxL2gen2dsdgomes1(tau0,m0-tau0*n1,w0-tau0*v1,pichulitaold,N,c)
        m1 = ans[0]
        w1 = ans[1]
        pichulitaold_1 = ans[2]


        #update step size
        theta = 1/np.sqrt(1+2*gamma*tau0)
        tau1 = tau0*theta
        sigma1 = sigma0/theta
        error = h*np.linalg.norm(m1-m0)

        #update
        m0b = m1 + theta*(m1 - m0)
        w0b = w1 + theta*(w1 - w0)
        n0 = n1
        v0 = v1
        m0 = m1
        w0 = w1
        tau0 = tau1
        sigma0 = sigma1
        pichulitaold = pichulitaold_1
        print('next iteration')
        counter += 1
        print('mshape',m1.shape)
        print([counter, error])

    g = np.linspace(0, 1, N1)
    X1, Y1 = np.meshgrid(g, g, indexing='ij')
    m0 = m0.flatten()
    X_flatten = X1.flatten()
    Y_flatten = Y1.flatten()

    grid_x, grid_y = np.mgrid[np.min(X_flatten):np.max(X_flatten):100j, np.min(Y_flatten):np.max(Y_flatten):100j]
    grid_z = griddata((X_flatten, Y_flatten), m0, (grid_x, grid_y), method='cubic')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
    plt.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('m(x, y)', fontsize=14)
    plt.title("3D Surface Plot of Real m(x, y)", fontsize=14)
    plt.tight_layout()
    plt.show()
    m0_tensor = torch.from_numpy(m0)

    return m0_tensor

# main(torch.tensor(50))