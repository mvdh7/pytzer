import jax
from jax import numpy as np

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
