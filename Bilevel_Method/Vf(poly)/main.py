import torch
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from kernels_gram import gram_matrix_assembly,PeriodicKernel
from operators import bmat_torch, prox_torch, proxL2gen2dsdgomes_torch
from samples import sampled_pts_grid_torch, sampled_points_fun_observe_torch,sampled_points_fun
import gc
import os
import matplotlib.ticker as ticker
torch.set_default_dtype(torch.float64)
fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))

class InverseProblemPotentialSolver(nn.Module):
    def __init__(self, domain=torch.tensor([[0.0, 1.0], [0.0, 1.0]]), kernel=PeriodicKernel, alpha1=10**6, alpha2=10**10, h=1/50):
        super(InverseProblemPotentialSolver, self).__init__()
        self.kernel = kernel()
        self.domain = domain
        self.alpha1 = torch.tensor(alpha1)
        self.alpha2 = torch.tensor(alpha2)
        self.h1=h
        self.h = torch.tensor(h)
        self.N1 = int((1 / self.h).floor())
        self.N = self.N1 * self.N1
        self.errorac = []
        self.errorac2 = []
        self.iter = 20
        self.lrv=0.03
        self.lrz = 0.00001
        self.k = 2
        self.V_1 = self.N
        self.V_2 = round(1 / h / self.k) * round(1 / h / self.k)
        self.initial_v = torch.ones(self.V_1, requires_grad=True)
        self.v = nn.Parameter(self.initial_v)
        self.num_nodes = 8
        initial_values = torch.ones(self.num_nodes)
        #initial_values = torch.tensor([0, 0, 0, 0.25, 0, 0, 0, 0], dtype=torch.float64)
        self.z = nn.Parameter(initial_values)

        self.exponents = torch.arange(1, len(self.z) + 1)
        self.alpha=3
        self.alpha3=0.01
        self.A = torch.zeros((self.N, self.N))
        self.B = bmat_torch(self.N1)
        self.C = torch.cat([self.A, self.B], dim=1)
        self.hm = torch.cat([(self.h ** 2) * torch.ones(1, self.N), torch.zeros(1, 4 * self.N)], dim=1)
        self.C = torch.cat([self.C, self.hm], dim=0)
        self.Q = torch.mm(self.C, self.C.T)
        self.n0 = torch.zeros(self.N)
        self.v0 = torch.zeros(4 * self.N)
        self.m0 = torch.ones(self.N)
        self.w0 = torch.zeros(4 * self.N)
        self.pichulitaold = torch.ones(self.N)
        self.Y =torch.linspace(0.5, 1.2, 500).reshape(-1, 1)
        torch.manual_seed(0)
    def sample_points(self, N):
        return sampled_pts_grid_torch(N, self.domain)

    def gram_matrix(self, X,Y, kernel):
        return gram_matrix_assembly(X,Y, kernel)

    def calculate_gram_V(self, X, kernel):
        V_2 = self.V_2
        v_0, V_0 = sampled_points_fun(V_2)
        V_0 = V_0.reshape(V_2, -1)
        v = torch.cat([X, v_0], dim=0)
        print('vshape', v.shape)
        gram_matrix_v = gram_matrix_assembly(v,v)
        print('gram_matrix_shape', gram_matrix_v.shape)
        eye = torch.eye(gram_matrix_v.shape[0], device=gram_matrix_v.device)
        gram_matrix_v += 1e-4 * eye
        L = torch.linalg.cholesky(gram_matrix_v)
        I = eye
        L_inv = torch.linalg.solve_triangular(L, I, upper=False)
        gram_matrix_inv = L_inv.T @ L_inv
        return gram_matrix_inv, X, v_0, V_0

    def Fx(self, x, z):
        powers = x.unsqueeze(-1).pow(self.exponents)
        result = torch.sum(z * powers, dim=-1)
        return result

    def F_prime(self, x, z):
        derivatives = self.exponents * z
        derivative_powers = x.unsqueeze(-1).pow(self.exponents - 1)
        result = torch.sum(derivatives * derivative_powers, dim=-1)
        return result

    def F_prime_prime(self, x, z):
        second_derivatives = self.exponents * (self.exponents - 1) * z
        second_derivative_powers = x.unsqueeze(-1).pow(self.exponents - 2)
        result = torch.sum(second_derivatives * second_derivative_powers, dim=-1)
        return result

    def calculate_Gaussian(self, gram_matrix_inv, v):
        v_inv = torch.matmul(gram_matrix_inv, v)
        gaussian_term = torch.dot(v.T, v_inv)
        return gaussian_term


    def calculate_Fdloss(self,Y, z):
        F_d = self.F_prime(Y, z)
        F_d_diag = torch.diag_embed(F_d.squeeze())
        R = torch.diag(Y[:, 0])
        # Create the L and U matrices
        N = len(F_d)
        L = -torch.tril(torch.ones(N, N))
        U = -L
        # Compute (F'L + UF') and (RL + UR)
        FL_plus_UF = F_d_diag @ L + U @ F_d_diag
        RL_plus_UR = R @ L + U @ R
        F_dloss = 2 * torch.sum(FL_plus_UF * RL_plus_UR)
        return F_dloss


    def prepare_data(self):
        domain = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        N1 = (1 / self.h).floor()
        lambda_val = torch.tensor(-1.0966796875)
        X, function_values, m_0, selection_mask = sampled_points_fun_observe_torch(N1, self.k,self.alpha,lambda_val, domain)
        #self.v = nn.Parameter(function_values)
        gram_matrix_inv, X, v_0, V_0 = self.calculate_gram_V(X, self.kernel)
        gamma = 0.001
        torch.manual_seed(0)
        V_0 = V_0.squeeze(1)
        noise_v0 = torch.normal(0, gamma, size=V_0.size())
        V_0 = V_0 + noise_v0
        Y=self.Y
        return X,function_values, m_0, selection_mask, gram_matrix_inv,  V_0,lambda_val,Y

    def upper_level_loss(self,X, m_0, selection_mask, gram_matrix_inv,  V_0,Y,step):
        full_v = torch.cat([self.v, V_0])
        gaussian_term = 1e4*self.calculate_Gaussian(gram_matrix_inv, full_v)
        F_dloss=self.alpha3*self.calculate_Fdloss(Y, self.z)
        F_loss = 4e4*torch.norm(self.z, p=2).pow(2)
        m0,error2 = self.proximal_algorithm_loss(m_0, selection_mask, self.v,Y)
        gamma = 0.001
        torch.manual_seed(0)
        noise = torch.normal(0, gamma, size=m_0.size())
        m_0_noisy = m_0 + noise
        errors = self.alpha1 * (m_0_noisy - m0) ** 2
        proximal_loss_term = errors[selection_mask].sum()
        if step % 1 == 0:
            print('gaussian_term', gaussian_term)
            print('F_dloss', F_dloss)
            print('F_loss', F_loss)
            print('proximal_loss_term',proximal_loss_term)
            print('z',self.z)
        total_loss =proximal_loss_term+gaussian_term+F_loss-F_dloss
        return total_loss,m0,error2



    def train_model(self, iterations):
        optimizer = optim.Adam([
            {'params': [self.v], 'lr': self.lrv},
            {'params': [self.z], 'lr': self.lrz}
        ])
        X,function_values, m_0, selection_mask, gram_matrix_inv, V_0,lambda_val,Y = self.prepare_data()
        Loss = []
        Iterations = []
        errormin = 1
        m0_min = torch.ones(self.N)
        v0_min = self.v.clone().detach()
        stepmin = 0
        toleran = 0
        errorac2min = []

        for step in range(iterations):
            optimizer.zero_grad()
            loss, m0, error2 = self.upper_level_loss(X, m_0, selection_mask, gram_matrix_inv, V_0, Y,step)
            if error2 < 0.423:
                optimizer.param_groups[0]['lr'] = 0.01
                optimizer.param_groups[1]['lr'] = 0.005
                
            if error2 < errormin:
                errormin = error2
                errorac2min = self.errorac2.copy()
                m0_min = m0.clone().detach()
                v0_min = self.v.clone().detach()
                toleran = 0
                stepmin = step
            else:
                toleran+=1

            if toleran>15:
                self.errorac2=errorac2min
                print(f"Stopping early at step {stepmin} due to low error2: {errormin} otherwise overfitting")
                self.plot()
                self.plot_sampled_recover(X, m0_min)
                self.plot_error_contour(m0_min, m_0)
                self.plot_learned_function_and_error(X,function_values,v0_min)
                break

            if torch.isnan(error2):
                print(f"Stopping early at step {step} due to NaN in error2")
                break
            if step % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Step {step}, Loss: {loss.item()}, Iter: {self.iter}, LR: {current_lr}")
            loss.backward()
            optimizer.step()
            Loss.append(loss.item())
            Iterations.append(step)
            del loss, m0, error2
            gc.collect()
        print("Optimized v:", self.v.data)
        print("Optimized z:", self.z)

    def proximal_algorithm_loss(self,m_0,selection_mask,v_T1,Y):
        self.errorac = []
        self.errorac2 = []
        L = torch.tensor(1.0, dtype=torch.float32)
        gamma = torch.tensor(0.05, dtype=torch.float32)
        tau0 = 10/ L
        sigma0 = 0.1 / L
        tol = self.h ** 3
        error = 1
        m0 = self.m0
        w0 = self.w0
        m0b = m0
        w0b = w0
        n0 = self.n0
        v0 = self.v0
        pichulitaold = self.pichulitaold
        exp_sol = m_0
        counter = 0
        while counter < self.iter:
        # while counter < self.iter and error > tol:
            print(counter)
            y = n0 + sigma0 * m0b - sigma0 * torch.ones(self.N)
            z = v0 + sigma0 * w0b
            x1 = torch.matmul(self.C, torch.cat([y, z]))
            x2 = torch.linalg.solve(self.Q, x1)
            x3 = torch.matmul(self.C.T, x2)
            n1 = x3[:self.N]
            v1 = x3[self.N:]
            print('update dual')

            ans = proxL2gen2dsdgomes_torch(tau0, m0 - tau0 * n1, w0 - tau0 * v1, pichulitaold, v_T1, self.N, self.z, self.F_prime,self.F_prime_prime)
            m1 = ans[0]
            w1 = ans[1]
            pichulitaold_1 = ans[2]
            print('update prime')

            theta = 1 / np.sqrt(1 + 2 * gamma * tau0)
            tau1 = tau0 * theta
            sigma1 = sigma0 / theta

            error = self.h * torch.linalg.norm(m1 - m0)
            self.errorac.append(error)
            error2 = self.h * torch.linalg.norm(m1 - exp_sol)
            self.errorac2.append(error2)

            m0b = m1 + theta * (m1 - m0)
            w0b = w1 + theta * (w1 - w0)
            n0 = n1
            v0 = v1
            m0 = m1
            w0 = w1

            tau0 = tau1
            sigma0 = sigma1
            pichulitaold = pichulitaold_1
            print('next iteration')

            counter += 1
            if counter % 1 == 0:
                print([counter, error.item(), error2.item()])
        print(self.h * torch.linalg.norm(m0 - exp_sol).item())
        return m0,error2
    def plotloss(self, Iterations, Losses):
        Losses = [float(loss) for loss in Losses]
        plt.figure(figsize=(10, 5))
        plt.plot(Iterations, Losses, label='Training Loss')
        plt.title('Loss vs. Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Logarithmic Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot(self):
        file_path = 'data1.pt'
        if not os.path.exists(file_path):
            torch.save(self.errorac2, file_path)
            print("Data saved to file.")
        else:
            # Load the saved data if the file exists
            self.errorac2 = torch.load(file_path)
            print("Data loaded from file.")
        plt.rcParams['text.usetex'] = False
        iterations = list(range(len(self.errorac2)))
        if isinstance(self.errorac2, torch.Tensor):
            errorac2 = self.errorac2.numpy()
        elif isinstance(self.errorac2, list):
            errorac2 = [float(e) for e in self.errorac2]
        plt.figure()
        plt.plot(iterations, errorac2, label='Error', marker='x', linestyle='--', color='r')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=14, loc='upper right')
        plt.title('Recovery Error of $m(x,y)$ Across Iterations', fontsize=14)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Error', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('afig1.jpg', format='jpg', dpi=300)
        plt.show()

    def plot_learned_function_and_error(self, X, exp_sol,v0_min):
        file_path = 'data3.pt'
        if not os.path.exists(file_path):
            torch.save(v0_min, file_path)
            print("Data saved to file.")
        else:
            v0_min = torch.load(file_path)
            print("Data loaded from file.")
        learned_v = v0_min.detach().numpy()
        X = X.numpy()
        grid_x, grid_y = np.mgrid[np.min(X[:, 0]):np.max(X[:, 0]):100j, np.min(X[:, 1]):np.max(X[:, 1]):100j]
        grid_z = griddata(X[:, :2], learned_v, (grid_x, grid_y), method='cubic')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
        plt.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel("X", labelpad=10, fontsize=12)
        ax.set_ylabel("Y", labelpad=10, fontsize=12)
        ax.set_zlabel("V(x,y)", labelpad=10, fontsize=12)

        plt.title("3D Surface Plot of Recovery V(x,y)", fontsize=14)
        fig.tight_layout()
        plt.tight_layout()
        plt.savefig('afig4.jpg', format='jpg', dpi=300)
        plt.show()

        exp_sol = exp_sol.detach().numpy()
        V0_grid = learned_v.reshape((self.N1, self.N1))
        exp_sol_grid = exp_sol.reshape((self.N1, self.N1))
        error = np.abs(V0_grid - exp_sol_grid)
        x = np.linspace(self.domain[0, 0], self.domain[0, 1], self.N1)
        y = np.linspace(self.domain[1, 0], self.domain[1, 1], self.N1)
        X, Y = np.meshgrid(x, y, indexing='ij')

        plt.figure()
        err_contourf = plt.contourf(X, Y, error)
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cbar = plt.colorbar(err_contourf, format=fmt)
        cbar.set_label('Error', size=12)
        cbar.ax.yaxis.get_offset_text().set_fontsize(12)
        cbar.ax.tick_params(labelsize=12)
        plt.title('Error Contour of V(x, y)', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.tight_layout()
        plt.savefig('afig5.jpg', format='jpg', dpi=300)
        plt.show()

    def plot_sampled_recover(self, X, m_values):
        file_path = 'data2.pt'
        if not os.path.exists(file_path):
            torch.save(m_values, file_path)
            print("Data saved to file.")
        else:
            m_values = torch.load(file_path)
            print("Data loaded from file.")
        X = X.numpy()
        m_values = m_values.detach().numpy()
        grid_x, grid_y = np.mgrid[np.min(X[:, 0]):np.max(X[:, 0]):100j, np.min(X[:, 1]):np.max(X[:, 1]):100j]
        grid_z = griddata(X[:, :2], m_values, (grid_x, grid_y), method='cubic')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.tick_params(axis='both', which='major', labelsize=12)
        surface = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
        ax.set_xlabel("X", labelpad=10, fontsize=14)
        ax.set_ylabel("Y", labelpad=10, fontsize=14)
        ax.set_zlabel("m(x,y)", labelpad=10, fontsize=14)
        plt.title("3D Surface Plot of Recovery m(x,y)", fontsize=14)
        plt.tight_layout()
        plt.savefig('afig2.jpg', format='jpg', dpi=300)
        plt.show()

    def run_and_plot_for_different_k(self, k_values):
        results = []
        v2_values = []
        for k in k_values:
            self.k = k
            self.V_2 = round(1 / self.h1 / self.k) * round(1 / self.h1 / (self.k))
            v2_values.append(self.V_2)
            self.train_model(1000)
            results.append(self.errorac2[-1])
            results = [float(e) for e in results]
        plt.figure(figsize=(10, 5))
        plt.loglog(v2_values, results, marker='o', linestyle='-')
        plt.xlabel('Number of Observed Points')
        plt.ylabel(r"$L_2$ Error")
        plt.title(r"log-log Plot of $L_2$ Error of m")
        plt.grid(True)
        plt.xticks(v2_values, [str(v) for v in v2_values])
        plt.show()

    def plot_error_contour(self, m0, exp_sol):
        file_path = 'data2.pt'
        if not os.path.exists(file_path):
            torch.save(m0, file_path)
            print("Data saved to file.")
        else:
            m0 = torch.load(file_path)
            print("Data loaded from file.")
        m0 = m0.detach().numpy()
        exp_sol = exp_sol.detach().numpy()
        m0_grid = m0.reshape((self.N1, self.N1))
        exp_sol_grid = exp_sol.reshape((self.N1, self.N1))
        error = np.abs(m0_grid - exp_sol_grid)
        x = np.linspace(self.domain[0, 0], self.domain[0, 1], self.N1)
        y = np.linspace(self.domain[1, 0], self.domain[1, 1], self.N1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        fig = plt.figure()
        err_contourf = plt.contourf(X, Y, error)
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cbar = plt.colorbar(err_contourf, format=fmt)
        cbar.set_label('Error', size=14)
        cbar.ax.yaxis.get_offset_text().set_fontsize(12)
        cbar.ax.tick_params(labelsize=12)
        plt.title('Error Contour of m(x, y)', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)
        plt.tight_layout()
        plt.savefig('afig3.jpg', format='jpg', dpi=300)
        plt.show()


solver = InverseProblemPotentialSolver()
k_values = [2]
solver.run_and_plot_for_different_k(k_values)
