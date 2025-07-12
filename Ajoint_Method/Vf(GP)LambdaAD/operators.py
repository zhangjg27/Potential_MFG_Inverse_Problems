import torch
import time
import gc

def bmat_torch(N):
    Ntot = N ** 2
    h = 1.0 / N
    w1 = torch.eye(Ntot)
    for i in range(Ntot):
        if i - N < 0:
            w1[i, Ntot + i - N] = -1
        else:
            w1[i, i - N] = -1
    w2 = -torch.eye(Ntot)
    for i in range(Ntot):
        if i + N >= Ntot:
            w2[i, i + N - Ntot] = 1
        else:
            w2[i, i + N] = 1

    w3 = torch.eye(Ntot)
    counter = 1
    for i in range(Ntot):
        if counter == 1:
            w3[i, i + N - 1] = -1
        else:
            w3[i, i - 1] = -1
        counter += 1
        if counter > N:
            counter = 1
    w4 = -torch.eye(Ntot)
    counter = 1
    for i in range(Ntot):
        if counter == N:
            w4[i, i + 1 - N] = 1
        else:
            w4[i, i + 1] = 1
        counter += 1
        if counter > N:
            counter = 1

    return N * torch.cat([w1, w2, w3, w4], dim=1)
def laplace_matrix_torch(N):
    Ntot = N**2
    h = 1 / N
    L = 4 * torch.eye(Ntot)

    for i in range(Ntot):
        if i - N < 0:
            L[i, Ntot + i - N] = -1
        else:
            L[i, i - N] = -1

        if i + N >= Ntot:
            L[i, i + N - Ntot] = -1
        else:
            L[i, i + N] = -1

    counter = 1
    for i in range(Ntot):
        if counter == 1:
            L[i, i + N - 1] = -1
        else:
            L[i, i - 1] = -1

        counter += 1
        if counter > N:
            counter = 1

    counter = 1
    for i in range(Ntot):
        if counter == N:
            L[i, i + 1 - N] = -1
        else:
            L[i, i + 1] = -1

        counter += 1
        if counter > N:
            counter = 1

    return N**2 * L


def prox_torch(tau,  pichulitaold, m, w1, w2, w3, w4, v_1,  z, F_prime,F_prime_prime, max_iter=30, tol=1e-4):
    k1, k2, k3, k4 = torch.clamp(w1, min=0), torch.clamp(w2, max=0), torch.clamp(w3, min=0), torch.clamp(w4, max=0)
    Q = (tau * v_1 - m) * tau ** 2 - tau / 2 * (k1 ** 2 + k2 ** 2 + k3 ** 2 + k4 ** 2)

    if m <= tau * v_1 and Q >= 0:
        if pichulitaold.dim() == 0:
            pichulitaold=pichulitaold.view(1, 1)
        elif pichulitaold.dim() == 1:
            pichulitaold=pichulitaold.unsqueeze(1)
        pichulitaold_n = 0. * pichulitaold
        w1sol = pichulitaold_n
        w2sol = pichulitaold_n
        w3sol = pichulitaold_n
        w4sol = pichulitaold_n
        return [pichulitaold_n, pichulitaold_n, w1sol, w2sol, w3sol, w4sol]

    else:
         x = pichulitaold
         lr = 1.0
         dynamic_max_iter = max_iter

         for _ in range(dynamic_max_iter):
             f =  (x + tau * (v_1 + F_prime(x,  z) + torch.log(torch.abs(x) + 1e-8) * 1e-8) - m) * (x + tau) ** 2 - (
                tau / 2) * (
                          k1 ** 2 + k2 ** 2 + k3 ** 2 + k4 ** 2)
             df =  (1 + tau * (F_prime_prime(x,  z) + 1 / (torch.abs(x) + 1e-8) * 1e-8)) * (x + tau) ** 2 + 2 * (
                x + tau) * (
                           x + tau * (v_1 + F_prime(x, z) + torch.log(torch.abs(x) + 1e-8) * 1e-8) - m)
             delta_x = -lr * f / (df + 1e-6)
             x_new = x + delta_x
             if torch.abs(delta_x) < tol:
                 break
             if x_new < 0:
                lr *= 0.5
                dynamic_max_iter += 3
                if dynamic_max_iter > 45:
                    break
                continue
             x = x_new

         if x.dim() == 0:
            x=x.view(1, 1)
         elif x.dim() == 1:
             x=x.unsqueeze(1)
         w1sol = x * k1 / (x + tau)
         w2sol = x * k2 / (x + tau)
         w3sol = x * k3 / (x + tau)
         w4sol = x * k4 / (x + tau)
         a=[x, x, w1sol, w2sol, w3sol, w4sol]

         return a





def proxL2gen2dsdgomes_torch(tau,  m, w, pichulitaold, v_1,  N,D,z,F_prime,F_prime_prime):
    w11 = w[:N]
    w22 = w[N:2 * N]
    w33 = w[2 * N:3 * N]
    w44 = w[3 * N:]
    W = torch.stack([w11, w22, w33, w44])
    DW = torch.mm(D, W)
    w1 = DW[0, :]
    w2 = DW[1, :]
    w3 = DW[2, :]
    w4 = DW[3, :]
    t1 = time.time()
    results = [prox_torch(tau,  pichulitaold[i], m[i], w1[i], w2[i], w3[i], w4[i], v_1[i],z,F_prime,F_prime_prime) for i in range(N)]
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")

    pichulitaold_n = torch.stack([r[0] for r in results]).view(-1)
    msol = torch.stack([r[1] for r in results]).view(-1)
    w1sol = torch.stack([r[2] for r in results]).view(-1)
    w2sol = torch.stack([r[3] for r in results]).view(-1)
    w3sol = torch.stack([r[4] for r in results]).view(-1)
    w4sol = torch.stack([r[5] for r in results]).view(-1)

    wsol = torch.cat([w1sol, w2sol, w3sol, w4sol], dim=0)
    del results, w1, w2, w3, w4, w1sol, w2sol, w3sol, w4sol
    gc.collect()
    return msol, wsol, pichulitaold_n