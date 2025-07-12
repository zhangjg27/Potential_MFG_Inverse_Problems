import torch
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import psutil
from kernels_gram import gram_matrix_assembly
from operators import bmat_torch, prox_torch, proxL2gen2dsdgomes_torch
from samples import sampled_pts_grid_torch, sampled_points_fun_observe_torch,sampled_points_fun
from kernels_gram import PeriodicKernel
import gc
import pickle
import os
from matplotlib.ticker import ScalarFormatter
torch.set_default_dtype(torch.float64)


class InverseProblemPotentialSolver(nn.Module):
    def __init__(self, domain=torch.tensor([[0.0, 1.0], [0.0, 1.0]]), kernel=PeriodicKernel, alpha1=10**5, alpha2=10**10, h=1/40):
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
        self.errorV=0
        self.iter =30
        self.lr=0.1
        self.k = 4
        self.V_1 = self.N
        self.V_2 = round(1 / h / self.k) * round(1 / h / self.k)
        initial_v = torch.ones(self.V_1, requires_grad=True)
        self.v = nn.Parameter(initial_v)
        beta_initial = torch.tensor(2.0, requires_grad=True)
        self.beta=nn.Parameter(beta_initial)
        self.alpha=3
        self.alpha3=10
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

    def sample_points(self, N):
        return sampled_pts_grid_torch(N, self.domain)

    def gram_matrix(self, X, kernel):
        return gram_matrix_assembly(X, kernel)

    def calculate_gram_V(self, X, kernel):
        V_2 = self.V_2
        v_0, V_0 = sampled_points_fun(V_2)
        V_0 = V_0.reshape(V_2, -1)
        v = torch.cat([X, v_0], dim=0)
        gram_matrix_v = self.gram_matrix(v, kernel)
        eye = torch.eye(gram_matrix_v.shape[0], device=gram_matrix_v.device)
        gram_matrix_v += 1e-4 * eye
        L = torch.linalg.cholesky(gram_matrix_v)
        I = eye
        L_inv = torch.linalg.solve_triangular(L, I, upper=False)
        gram_matrix_inv = L_inv.T @ L_inv
        return gram_matrix_inv, X, v_0, V_0

    def calculate_Gaussian(self, gram_matrix_inv, v):
        v_inv = torch.matmul(gram_matrix_inv, v)
        gaussian_term = torch.dot(v.T, v_inv)
        return gaussian_term

    def prepare_data(self):
        domain = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        N1 = (1 / self.h).floor()
        lambda_val = torch.tensor(-1.0966796875)
        # Generate sample points
        X, function_values, m_0, selection_mask = sampled_points_fun_observe_torch(N1, self.k,self.alpha,lambda_val, domain)
        gram_matrix_inv, X, v_0, V_0 = self.calculate_gram_V(X, self.kernel)
        gamma = 0.001  # Standard deviation of the noise
        torch.manual_seed(0)
        V_0 = V_0.squeeze(1)
        # Generate Gaussian noise for V_0
        noise_v0 = torch.normal(0, gamma, size=V_0.size())
        V_0 = V_0 + noise_v0
        return X, m_0, selection_mask, gram_matrix_inv,  V_0,lambda_val, function_values

    def upper_level_loss(self,X, m_0, selection_mask, gram_matrix_inv,  V_0):
        full_v = torch.cat([self.v, V_0])
        gaussian_term = self.calculate_Gaussian(gram_matrix_inv, full_v)
        beta_loss=self.alpha3*self.beta**2
        m0,error2 = self.proximal_algorithm_loss(m_0, selection_mask, self.v,self.beta)
        gamma = 0.001  # Standard deviation of the noise
        torch.manual_seed(0)
        # Generate Gaussian noise
        noise = torch.normal(0, gamma, size=m_0.size())
        # Add the noise to your initial observations
        m_0_noisy = m_0 + noise
        errors = self.alpha1 * (m_0_noisy - m0) ** 2
        proximal_loss_term = errors[selection_mask].sum()
        total_loss =proximal_loss_term+gaussian_term+beta_loss

        return total_loss,m0,error2


    def train_model(self, iterations):

        optimizer = optim.Adam([self.v, self.beta], lr=self.lr)
        X, m_0, selection_mask, gram_matrix_inv, V_0,lambda_val, function_values = self.prepare_data()
        Loss = []
        Iterations = []
        errormin=1
        errorminV=1
        stepmin=0
        toleran=0
        betamin=self.beta
        signal=0
        for step in range(iterations):
            optimizer.zero_grad()
            loss,m0,error2 = self.upper_level_loss(X, m_0, selection_mask, gram_matrix_inv,  V_0)
            loss.backward()
            optimizer.step()
            if error2<errormin:
                errormin=error2
                errorminV=self.h * torch.linalg.norm(self.v - function_values)
                toleran=0
                betamin = self.beta
                stepmin=step
            else:
                toleran+=1
            if toleran>40:
                self.errorac2[-1]=errormin
                self.errorV=errorminV
                signal=1
                print(f"Stopping early at step {stepmin} due to low error2: {errormin} otherwise overfitting")
                print("Optimized beta:", betamin.item())
                break
            if step > 30:
                for g in optimizer.param_groups:
                    g['lr'] = 0.03
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 97:
                print(f"Stopping early at step {step} due to high memory usage: {memory_usage}%")
                break
            if step % 30 == 0:
                beta_grad = self.beta.grad if self.beta.grad is not None else 'No grad'
                current_lr = optimizer.param_groups[0]['lr']
                print('44grad', self.v.grad)
                print(f"Step {step}, Loss: {loss.item()}, Iter: {self.iter}, LR: {current_lr}, Beta: {self.beta.item()},Beta Grad: {beta_grad}")
            del loss,error2
            gc.collect()
        if signal==0:
            self.errorV = self.h * torch.linalg.norm(self.v - function_values)

        # self.plot()
        # self.plot_sampled_recover(X, m0)
        # self.plot_error_contour(m0, m_0)
        # self.plotloss(Iterations, Loss)
        # print("Optimized v:", self.v.data)
        # print("Optimized beta:", self.beta.item())

    def proximal_algorithm_loss(self,m_0,selection_mask,v_T1,beta):
        self.errorac = []
        self.errorac2 = []
        L = torch.tensor(1.0)
        gamma = torch.tensor(0.05)
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
        while counter < self.iter and error > tol:
            print(counter)
            y = n0 + sigma0 * m0b - sigma0 * torch.ones(self.N)
            z = v0 + sigma0 * w0b
            x1 = torch.matmul(self.C, torch.cat([y, z]))
            x2 = torch.linalg.solve(self.Q, x1)
            x3 = torch.matmul(self.C.T, x2)
            n1 = x3[:self.N]
            v1 = x3[self.N:]
            print('update dual')
            ans = proxL2gen2dsdgomes_torch(tau0, self.alpha1, m0 - tau0 * n1, w0 - tau0 * v1, pichulitaold, v_T1, m_0, self.N, selection_mask,beta)
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
        plt.rcParams['text.usetex'] = False
        iterations = list(range(len(self.errorac2)))
        if isinstance(self.errorac2, torch.Tensor):
            errorac2 = self.errorac2.numpy()
        elif isinstance(self.errorac2, list):
            errorac2 = [float(e) for e in self.errorac2]
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, errorac2, label='Error', marker='x', linestyle='--', color='r')
        plt.legend()
        plt.title('Recovery Error of Distribution Function m Across Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)



    def plot_sampled_recover(self, X, m_values):
        X = X.numpy()
        m_values = m_values.detach().numpy()
        grid_x, grid_y = np.mgrid[np.min(X[:, 0]):np.max(X[:, 0]):100j, np.min(X[:, 1]):np.max(X[:, 1]):100j]
        grid_z = griddata(X[:, :2], m_values, (grid_x, grid_y), method='cubic')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
        cax = fig.add_axes([0.15, 0.1, 0.02, 0.8])
        plt.colorbar(surface, cax=cax, orientation='vertical', label='Function Values')
        ax.set_xlabel("X", labelpad=10)
        ax.set_ylabel("Y", labelpad=10)
        ax.set_zlabel("m(x,y)", labelpad=10)
        plt.title("3D Surface Plot of Recovery m(x,y)", fontsize=14, position=(18, 0.95))


    def run_and_plot_for_different_k(self, k_values):
        results_filename = 'results23.pkl'
        resultsv_filename = 'resultsv23.pkl'
        v2_values_filename = 'v2_values23.pkl'

        if os.path.exists(results_filename) and os.path.exists(resultsv_filename) and os.path.exists(
                v2_values_filename):
            with open(results_filename, 'rb') as file:
                results = pickle.load(file)
            with open(resultsv_filename, 'rb') as file:
                resultsv = pickle.load(file)
            with open(v2_values_filename, 'rb') as file:
                v2_values = pickle.load(file)
        else:
            results = []
            resultsv = []
            v2_values = []
            for k in k_values:
                self.k = k
                self.V_2 = round(1 / self.h1 / self.k) * round(1 / self.h1 / self.k)
                v2_values.append(self.V_2)
                self.train_model(1500)
                results.append(self.errorac2[-1])
                resultsv.append(self.errorV)
            results = [float(e) for e in results]
            resultsv = [float(e) for e in resultsv]
            with open(results_filename, 'wb') as file:
                pickle.dump(results, file)
            with open(resultsv_filename, 'wb') as file:
                pickle.dump(resultsv, file)
            with open(v2_values_filename, 'wb') as file:
                pickle.dump(v2_values, file)
        plt.figure()
        ax = plt.gca()
        plt.loglog(v2_values, results, marker='o', linestyle='-')
        plt.xlabel('Number of Observed Points', fontsize=14)
        plt.ylabel(r"$L_2$ Error", fontsize=14)
        plt.title(r"log-log Plot of $L_2$ Error of m", fontsize=14)
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xticks(v2_values, labels=[str(int(v)) for v in v2_values])
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontsize(12)
        plt.tight_layout()
        plt.savefig('afig1loglogplot.jpg', format='jpg', dpi=300)
        plt.show()

        plt.figure()
        ax = plt.gca()
        plt.loglog(v2_values, resultsv, marker='o', linestyle='-')
        plt.xlabel('Number of Observed Points', fontsize=14)
        plt.ylabel(r"$L_2$ Error", fontsize=14)
        plt.title(r"log-log Plot of $L_2$ Error of V", fontsize=14)
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xticks(v2_values, labels=[str(int(v)) for v in v2_values])
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontsize(12)
        plt.tight_layout()
        plt.savefig('afig2loglogplot.jpg', format='jpg', dpi=300)
        plt.show()


    def plot_error_contour(self, m0, exp_sol):
        m0 = m0.detach().numpy()
        exp_sol = exp_sol.detach().numpy()
        m0_grid = m0.reshape((self.N1, self.N1))
        exp_sol_grid = exp_sol.reshape((self.N1, self.N1))
        error = np.abs(m0_grid - exp_sol_grid)
        x = np.linspace(self.domain[0, 0], self.domain[0, 1], self.N1)
        y = np.linspace(self.domain[1, 0], self.domain[1, 1], self.N1)
        X, Y = np.meshgrid(x, y)

        # Plot the contour plot of the error
        plt.figure()
        plt.contourf(X, Y, error)
        plt.colorbar(label='Error')
        plt.title('Error Contour of m(x, y)')
        plt.xlabel('X')
        plt.ylabel('Y')


solver = InverseProblemPotentialSolver()
k_values = [10,8,6,4,2]
solver.run_and_plot_for_different_k(k_values)