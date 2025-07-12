import numpy as np
from scipy.optimize import fsolve



# linear operator:div
def bmat(N):
    Ntot=N**2
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



def prox(tau,alpha_1, pichulitaold, m, w1, w2, w3, w4,v_1,m_0,selection_mask):
    k1 = np.max([0, w1])
    k2 = np.min([0, w2])
    k3 = np.max([0, w3])
    k4 = np.min([0, w4])
    # Adding noise to m_0
    noise = np.random.normal(0, 0.001, size=m_0.shape)
    m_0_noisy = m_0 + noise

    if selection_mask:
        f = lambda x: (x + tau * (v_1 + np.log(x) + 2 * alpha_1 * (x - m_0_noisy)) - m) * (x + tau) ** 2 - (tau / 2) * (
                k1 ** 2 + k2 ** 2 + k3 ** 2 + k4 ** 2)
    else:
        f = lambda x: (x + tau * (v_1 + np.log(x)) - m) * (x + tau) ** 2 - (tau / 2) * (
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
def proxL2gen2dsdgomes(tau,alpha_1,m,w,pichulitaold,v_1,m_0,N,selection_mask):
    w1 = w[:N]
    w2 = w[N:2*N]
    w3 = w[2*N:3*N]
    w4 = w[3*N:]


    re = [prox(tau,alpha_1,pichulitaold[i], m[i], w1[i], w2[i], w3[i], w4[i],v_1[i],m_0[i],selection_mask[i]) for i in range(N)]
    pichulitaold_n = np.array([re[i][0] for i in range(N)]).flatten()
    msol = np.array([re[i][1] for i in range(N)]).flatten()
    w1sol = np.array([re[i][2] for i in range(N)]).flatten()
    w2sol = np.array([re[i][3] for i in range(N)]).flatten()
    w3sol = np.array([re[i][4] for i in range(N)]).flatten()
    w4sol = np.array([re[i][5] for i in range(N)]).flatten()

    wsol = np.block([w1sol, w2sol, w3sol, w4sol])
    return [msol,wsol,pichulitaold_n]