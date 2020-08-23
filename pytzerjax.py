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


cats = jax.random.normal(jax.random.PRNGKey(1), shape=(1, 10))
combi = jax.random.normal(jax.random.PRNGKey(2), shape=(10, 8))
anis = jax.random.normal(jax.random.PRNGKey(3), shape=(8, 1))


@jax.jit
def sum_loop(cats, anis, combi):
    total = 0.0
    for c, cat in enumerate(cats.ravel()):
        for a, ani in enumerate(anis.ravel()):
            total = total + cat * ani * combi[c, a]
    return total
    
@jax.jit
def sum_matrix(cats, anis, combi):
    return cats @ combi @ anis


sl = sum_loop(cats, anis, combi)
sm = sum_matrix(cats, anis, combi)


grad_loop = jax.jit(jax.grad(sum_loop))
grad_matrix = jax.jit(jax.grad(sum_matrix))

# just use the loop approach with jit for equal or better performance than matrix
