import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
jax.config.update("jax_enable_x64", True)
from scipy import integrate
from functools import partial  # for jit to make codes faster




class Periodic_kernel(object):
    def __init__(self):
        pass

    @partial(jit, static_argnums=(0,))
    def kappa(self, x, y, sigma):
        d = 0
        for i in range(x.shape[0]):
            d += (-(x[i]-y[i])**2)/(2*sigma ** 2)
        return jnp.exp(d)

    def integral_x_kappa(self, y, domain, sigma):
        val = 1
        for i in range(y.shape[0]):
            f = lambda x: jnp.exp((-(x[i]-y[i])**2)/(2*sigma ** 2))
            val *= integrate.quad(f, domain[i, 0], domain[i, 1])[0]
        return val

    @partial(jit, static_argnums=(0,))
    def D_x_kappa(self, x, y, sigma, i):
        return grad(self.kappa, 0)(x, y, sigma)[i]

    @partial(jit, static_argnums=(0,))
    def D_y_kappa(self, x, y, sigma, i):
        return grad(self.kappa, 1)(x, y, sigma)[i]

    @partial(jit, static_argnums=(0,))
    def Div_x_kappa(self, x, y, sigma):
        return sum(grad(self.kappa, 0)(x, y, sigma))

    @partial(jit, static_argnums=(0,))
    def Div_y_kappa(self, x, y, sigma):
        return sum(grad(self.kappa, 1)(x, y, sigma))

    @partial(jit, static_argnums=(0,))
    def Delta_x_kappa(self, x, y, sigma):
        val = grad(self.D_x_kappa, 0)(x, y, sigma, 0)[0]
        for i in range(1, x.shape[0]):
            val += grad(self.D_x_kappa, 0)(x, y, sigma, i)[i]
        return val

    @partial(jit, static_argnums=(0,))
    def Delta_y_kappa(self, x, y, sigma):
        val = grad(self.D_y_kappa, 1)(x, y, sigma, 0)[0]
        for i in range(1, x.shape[0]):
            val += grad(self.D_y_kappa, 1)(x, y, sigma, i)[i]
        return val

    @partial(jit, static_argnums=(0,))
    def D_x_D_y_kappa(self, x, y, sigma, s, r):
        return grad(self.D_x_kappa, 1)(x, y, sigma, s)[r]

    @partial(jit, static_argnums=(0,))
    def D_y_D_y_kappa(self, x, y, sigma, s, r):
        return grad(self.D_y_kappa, 1)(x, y, sigma, s)[r]

    @partial(jit, static_argnums=(0,))
    def D_x_Delta_y_kappa(self, x, y, sigma, i):
        return grad(self.Delta_y_kappa, 0)(x, y, sigma)[i]

    @partial(jit, static_argnums=(0,))
    def D_y_Delta_x_kappa(self, x, y, sigma, i):
        return grad(self.Delta_x_kappa, 1)(x, y, sigma)[i]

    @partial(jit, static_argnums=(0,))
    def Delta_x_Delta_y_kappa(self, x, y, sigma):
        val = grad(self.D_x_Delta_y_kappa, 0)(x, y, sigma, 0)[0]
        for i in range(1, x.shape[0]):
            val += grad(self.D_x_Delta_y_kappa, 0)(x, y, sigma, i)[i]
        return val
