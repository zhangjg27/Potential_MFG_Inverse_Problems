import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from sample_observations import (
    sampled_pts_grid,
    sampled_points_fun_observe,
    sampled_points_fun
)
plt.rcParams["text.usetex"] = False
fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))


from gram_matrixs import Gram_matrix_assembly
from Operators import bmat, prox, proxL2gen2dsdgomes
from kernels import Periodic_kernel





class inverse_problem_potential_solver(object):
    def __init__(self,  domain=np.array([[0, 1], [0, 1]]),kernel=Periodic_kernel , alpha1=10**5, alpha2=10**5,h=1/50):
        self.kernel= kernel
        self.domain = domain
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.errorV = 0
        self.h=h
        self.N1 = round(1 / h)
        self.N = self.N1 * self.N1
        self.errorac = []
        self.errorac2 = []
        self.k=10
        self.V_2 = round(1 / h/self.k) * round(1 / h / (self.k))
    def sample_points(self, N):
        return sampled_pts_grid(N, self.domain)

    def gram_matrix(self, X, kernel):
        return Gram_matrix_assembly(X, kernel)

    def calculate_gram_V1(self,X, kernel):
        gram_matrix_v = self.gram_matrix(X, kernel)
        print(gram_matrix_v.shape)
        return gram_matrix_v, X

    def calculate_gram_V(self,X, kernel):

        V_2 = self.V_2
        v_0, V_0 = sampled_points_fun(V_2)
        V_0 = V_0.reshape(V_2)
        gamma = 0.001
        np.random.seed(0)
        noise_v0 = np.random.normal(0, gamma, V_0.shape)
        V_0 = V_0 + noise_v0
        v = np.concatenate([X, v_0], axis=0)
        gram_matrix_v = self.gram_matrix(v, kernel)
        return gram_matrix_v, X, v_0,V_0

    
    def calculate_v(self, gram_matrix_v, V_0, m, kernel):
        V_2 = self.V_2
        Sigma_2 = np.zeros((self.N + V_2, self.N + V_2))
        I = np.eye(V_2)
        Sigma_2[-V_2:, -V_2:] = I
        L = np.linalg.cholesky(gram_matrix_v + 1e-8 * np.eye(gram_matrix_v.shape[0]))
        inverse_term0 = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(L.shape[0])))
        matrix = 10 ** (0) / self.alpha2 * inverse_term0 + Sigma_2
        L1 = np.linalg.cholesky(matrix)
        v_m = np.concatenate([0.5 * m / self.alpha2, V_0]).reshape(-1, 1)
        inverse_term = np.linalg.solve(L1.T, np.linalg.solve(L1, v_m))
        v_T = inverse_term
        return v_T


    def proximal_algorithm(self):
           L = 1
           gamma = 0.05
           k=self.k
           kernel=self.kernel
           h=self.h
           N1 = round(1 / h)
           N = N1 * N1
           # CP algorithm parameter
           tau0 = 10 / L
           sigma0 = 0.1 / L
           errorac = []
           errorac2 = []
           error = 1
           domain = np.array([[h, 1], [h, 1]])

           X,function_values,m_0,selection_mask= sampled_points_fun_observe(N1,k,domain)
           exp_sol=m_0
           gram_matrix_v, X,v_0,V_0=self.calculate_gram_V(X, kernel)
           A = np.zeros((N, N))
           B = bmat(N1)
           C = np.block([A, B])
           hm = np.block([(h ** 2) * np.ones([1, N]), np.zeros([1, 4 * N])])
           C = np.block([[C], [hm]])
           Q = np.matmul(C, np.transpose(C))

           # initial point
           n0 = np.zeros(N)
           v0 = np.zeros(4 * N)
           m0 = np.ones(N)
           w0 = np.zeros(4 * N)
           m0b = m0
           w0b = w0
           pichulitaold = np.ones(N)
           # iteration algorithm
           counter = 0
           while  counter <= 30:
               print(counter)
               # update dual variables
               v_T=self.calculate_v(gram_matrix_v, V_0,m0b,kernel)
               v_T1 = v_T[:N]
               y = n0 + sigma0 * m0b - sigma0 * (np.ones(N))
               z = v0 + sigma0 * w0b
               x1 = np.matmul(C, np.block([y, z]))
               x2 = np.linalg.solve(Q, x1)
               x3 = np.matmul(np.transpose(C), x2)
               n1 = x3[:N]
               v1 = x3[N:]
               print('update dual')

               # update prime variables
               ans = proxL2gen2dsdgomes(tau0,self.alpha1,m0 - tau0 * n1, w0 - tau0 * v1, pichulitaold,v_T1,m_0,N,selection_mask)
               m1 = ans[0]
               w1 = ans[1]
               pichulitaold_1 = ans[2]
               print('update prime')

               # update step size
               theta = 1 / np.sqrt(1 + 2 * gamma * tau0)
               tau1 = tau0 * theta
               sigma1 = sigma0 / theta

               # error
               error = h * np.linalg.norm(m1 - m0)
               errorac.append(error)
               error2 = h*np.linalg.norm(m1 - exp_sol)
               errorac2.append(error2)

               # update
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
                   print([counter, error, error2])
           self.errorac = errorac
           self.errorac2 = errorac2
           e = h * np.linalg.norm(m0 - exp_sol)
           self.errorV = self.h * np.linalg.norm(np.squeeze(v_T1) - function_values)
           print(e)


    def plot(self):
        plt.rcParams['text.usetex'] = False
        errorac=self.errorac
        errorac2=self.errorac2
        iterations = list(range(len(errorac)))

        plt.figure()
        plt.plot(iterations, errorac2, label='Error', marker='x', linestyle='--', color='r')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=14, loc='upper right')
        plt.title('Recovery Error of $m(x,y)$ Across Iterations', fontsize=14)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Error', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('afig1V.jpg', format='jpg', dpi=300)
        plt.show()

    def plot_learned_function_and_error(self, X,learned_v, exp_sol):
        learned_v = np.squeeze(learned_v)
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
        plt.savefig('afig4V.jpg', format='jpg', dpi=300)
        plt.show()

        # Reshape m0 and exp_sol to 2D grid
        V0_grid = learned_v.reshape((self.N1, self.N1))
        exp_sol_grid = exp_sol.reshape((self.N1, self.N1))

        # Calculate the error between m0 and exp_sol
        error = np.abs(V0_grid - exp_sol_grid)
        # Create a grid of x and y values
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
        plt.savefig('afig5V.jpg', format='jpg', dpi=300)
        plt.show()



    def plot_sampled_recover(self, X, m_values):
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
        plt.savefig('afig2V.jpg', format='jpg', dpi=300)
        plt.show()

    def run_and_plot_for_different_k(self, k_values):
        results = []
        resultsv = []
        v2_values = []
        for k in k_values:
            self.k = k
            self.V_2 = round(1 / self.h / self.k) * round(1 / self.h / (self.k))  # 更新 self.V_2 的值
            v2_values.append(self.V_2)
            self.proximal_algorithm()
            results.append(self.errorac2[-1])
            resultsv.append(self.errorV)
        plt.figure()
        ax = plt.gca()
        plt.loglog(v2_values, results, marker='o', linestyle='-')  # 使用 loglog 函数
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
        plt.savefig('afigloglogplotmodifymanyVM.jpg', format='jpg', dpi=300)
        plt.show()

        plt.figure()
        ax = plt.gca()
        plt.loglog(v2_values, resultsv, marker='o', linestyle='-')  # 使用 loglog 函数
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
        plt.savefig('afigloglogplotmodifymanyVV.jpg', format='jpg', dpi=300)
        plt.show()

    def plot_error_contour(self, m0, exp_sol):
        m0_grid = m0.reshape((self.N1, self.N1))
        exp_sol_grid = exp_sol.reshape((self.N1, self.N1))

        # Calculate the error between m0 and exp_sol
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
        plt.savefig('afig3V.jpg', format='jpg', dpi=300)
        plt.show()
solver = inverse_problem_potential_solver()
k_values = [10, 8, 6,4,2]
solver.run_and_plot_for_different_k(k_values)
