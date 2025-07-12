import jax.numpy as jnp
from jax import vmap
from kernels import Periodic_kernel
import numpy as np


def Gram_matrix_assembly(X, kernel='periodic', kernel_parameter=(0.6)):
    N = X.shape[0]
    XX = jnp.tile(X, (N, 1))
    XX0 = jnp.tile(jnp.array([X]), (N, 1, 1))
    XX0 = XX0.transpose([1, 0, 2])
    XX0 = XX0.reshape((N*N, X.shape[-1]))
    K = Periodic_kernel()
    val = vmap(lambda x, y: K.kappa(x, y, kernel_parameter))(XX, XX0)
    Theta = np.reshape(val, (N, N))
    return Theta




