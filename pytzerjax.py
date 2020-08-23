import jax
from jax import numpy as np


def M88_eq13(T, a):
    """M88 equation 13."""
    return (
        a[0]
        + a[1] * T
        + a[2] / T
        + a[3] * np.log(T)
        + a[4] / (T - 263)
        + a[5] * T ** 2
        + a[6] / (680 - T)
        + a[7] / (T - 227)
    )


def Aosm_M88(T, P):
    """From Moller (1988)."""
    Aosm = M88_eq13(
        T,
        np.array(
            [
                3.36901532e-1,
                -6.32100430e-4,
                9.14252359e00,
                -1.35143986e-2,
                2.26089488e-3,
                1.92118597e-6,
                4.52586464e1,
                0,
            ]
        ),
    )
    valid = (T >= 273.15) & (T <= 573.15) & (P == 10.1325)
    return Aosm, valid


class ParameterLibrary:
    def __init__(self, T):
        self.Aosm = Aosm_M88(T, 10.1325)


pl = ParameterLibrary(298.15)


@jax.jit  # jit here speeds this function by 4x and its deriv by 2x
def ionic_strength(molalities, charges):
    """Ionic strength."""
    return 0.5 * np.sum(molalities * charges ** 2)


def ionic_z(molalities, charges):
    """Z function."""
    return np.sum(molalities * np.abs(charges))


molalities = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
charges = np.array([2, -1, 1, -1, -1])

# def excess_Gibbs(molalities, temperature=298.15, pressure=1.01325):


@jax.jit
def Gibbs_DH(I, Aosm, b=1.2):  # from CRP94 Eq. (AI1)
    """Calculate the Debye-Hueckel component of the excess Gibbs energy."""
    return -4 * Aosm * I * np.log(1 + b * np.sqrt(I)) / b


@jax.jit
def Gibbs(molalities, charges, prmlib):
    I = ionic_strength(molalities, charges)
    return Gibbs_DH(I, prmlib["Aosm"])


@jax.jit
def activity_coefficients(molalities, charges, prmlib):
    return jax.grad(Gibbs)(molalities, charges, prmlib)


plibs = {"Aosm": 3.1}
istr = ionic_strength(molalities, charges)
zstr = ionic_z(molalities, charges)
g = Gibbs(molalities, charges, plibs)
acf = activity_coefficients(molalities, charges, plibs)

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
