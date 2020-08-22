import jax
from jax import numpy as np

@jax.jit  # jit here speeds this function by 4x and its deriv by 2x
def ionic_strength(molalities, charges):
    """Ionic strength."""
    return 0.5 * np.sum(molalities * charges ** 2)

def ionic_z(molalities, charges):
    """Z function."""
    return np.sum(molalities * np.abs(charges))


molalities = np.array([1., 2., 3., 2., 1.])
charges = np.array([2, -1, 1, -1, -1])

# def excess_Gibbs(molalities, temperature=298.15, pressure=1.01325):
    
    
istr = ionic_strength(molalities, charges)
zstr = ionic_z(molalities, charges)


istr_grad = jax.jit(jax.grad(ionic_strength))
# ^ 1000 x faster with jit here, regardless of whether jit above is there
istr_g = istr_grad(molalities, charges)
