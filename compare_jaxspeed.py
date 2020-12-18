import jax
from jax import numpy as np

dictopt = {0: {1: {2: 4.5}}}
arropt = np.ones((3, 3, 3)) * 4.5


@jax.jit
def h(x):
    """h function following CRP94 Eq. (AI15)."""
    return (6 - (6 + x * (6 + 3 * x + x ** 2)) * np.exp(-x)) / x ** 4
    

@jax.jit
def get_dict(dictopt):
    return h(dictopt[0][1][2])
    

@jax.jit
def get_arr(arropt):
    return h(arropt[0, 1, 2])


test_dict = get_dict(dictopt)
test_arr = get_arr(arropt)
