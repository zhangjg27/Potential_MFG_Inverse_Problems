import torch
import time
import gc

def bmat_torch(N):
    Ntot = N ** 2
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
def prox_torch(tau, pichulitaold, m, w1, w2, w3, w4, v_1,  z,F_prime,F_prime_prime,max_iter=15, tol=1e-3):
    k1, k2, k3, k4 = torch.clamp(w1, min=0), torch.clamp(w2, max=0), torch.clamp(w3, min=0), torch.clamp(w4, max=0)
    x = pichulitaold
    if x < 0:
        print('x is smaller than 0', x)
    lr = 1.0
    for _ in range(max_iter):
        f = (x + tau * (v_1 + F_prime(x, z)) + torch.log(torch.abs(x) + 1e-8) * (1e-8) - m) * (x + tau) ** 2 - (
                    tau / 2) * (
                    k1 ** 2 + k2 ** 2 + k3 ** 2 + k4 ** 2)
        df = (1 + tau * (F_prime_prime(x, z) + 1 / (torch.abs(x) + 1e-8) * (1e-8))) * (x + tau) ** 2 + 2 * (x + tau) * (
                x + tau * (v_1 + F_prime(x, z)) - m)
        delta_x = -lr * f / (df + 1e-8)
        x_new = x + delta_x
        if torch.abs(delta_x) < tol:
            break
        elif x_new < 0:
            lr *= 0.5
            continue
        x = x_new
    if x.dim() == 0:
        x = x.view(1, 1)
    elif x.dim() == 1:
        x = x.unsqueeze(1)
    w1sol = x * k1 / (x + tau)
    w2sol = x * k2 / (x + tau)
    w3sol = x * k3 / (x + tau)
    w4sol = x * k4 / (x + tau)
    a = [x, x, w1sol, w2sol, w3sol, w4sol]
    return a


def proxL2gen2dsdgomes_torch(tau, m, w, pichulitaold, v_1, N, z,  F_prime,F_prime_prime):
    w1 = w[:N]
    w2 = w[N:2*N]
    w3 = w[2*N:3*N]
    w4 = w[3*N:]
    t1 = time.time()

    results = [prox_torch(tau, pichulitaold[i], m[i], w1[i], w2[i], w3[i], w4[i], v_1[i], z, F_prime,F_prime_prime) for i in range(N)]
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

