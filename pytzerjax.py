import jax
from jax import numpy as jnp
import numpy as np

def ca_none(T, P):
    """cation:anion --- no interaction effect."""
    return {
        "beta0": 0,
        "beta1": 0,
        "alpha1": 1,
        "beta2": 0,
        "alpha2": 1,
        "C0": 0,
        "C1": 0,
        "omega": 1,
        "valid": (T > 0) & (P > 0),
    }

@jax.jit
def M88_eq13(T, a):
    """M88 equation 13."""
    return (
        a[0]
        + a[1] * T
        + a[2] / T
        + a[3] * jnp.log(T)
        + a[4] / (T - 263)
        + a[5] * T ** 2
        + a[6] / (680 - T)
        + a[7] / (T - 227)
    )


@jax.jit
def Aphi_M88(T, P):
    """From Moller (1988)."""
    Aphi = M88_eq13(
        T,
        [
            3.36901532e-1,
            -6.32100430e-4,
            9.14252359,
            -1.35143986e-2,
            2.26089488e-3,
            1.92118597e-6,
            4.52586464e1,
            0,
        ],
    )
    valid = (T >= 273.15) & (T <= 573.15) & (P == 10.1325)
    return Aphi, valid


ions2charges = {
    "Na^+": 1,
    "K^+": 1,
    "Ca^{2+}": 2,
    "Mg^{2+}": 2,
    "Cl^-": -1,
    "Br^-": -1,
    "HSO_4^-": -1,
}


def i2c(ions):
    return np.array([ions2charges[ion] for ion in ions])


class ParameterLibrary:
    def __init__(self):
        self.Aphi = Aphi_M88
        self.b = 1.2
        self.ca = {}
        
        
    def add_interaction(self, cation, anion, func=ca_none):
        self.ca[cation + ":" + anion] = func
        
        
    def get_parameters(self, ions, T=298.15, P=10.1325):
        # charges = i2c(ions)
        # cations = ions[charges > 0]
        # anions = ions[charges < 0]
        return {"Aphi": self.Aphi(T, P)[0], "b": self.b}#,
            # "ca": [self.ca[cation + ':' + anion](T, P)
            #        for cation in cations
            #        for anion in anions]}




@jax.jit  # jit here speeds this function by 4x and its deriv by 2x
def ionic_strength(molalities, charges):
    """Ionic strength."""
    return 0.5 * jnp.sum(molalities * charges ** 2)


@jax.jit
def ionic_z(molalities, charges):
    """Z function."""
    return jnp.sum(molalities * jnp.abs(charges))


@jax.jit
def Gibbs_DH(I, parameters):  # from CRP94 Eq. (AI1)
    """Calculate the Debye-Hueckel component of the excess Gibbs energy."""
    b = parameters["b"]
    return -4 * parameters["Aphi"] * I * jnp.log(1 + b * jnp.sqrt(I)) / b


@jax.jit
def Gibbs(molalities, charges, parameters):
    I = ionic_strength(molalities, charges)
    return Gibbs_DH(I, parameters)


@jax.jit
def activity_coefficients(molalities, charges, pdict):
    return jax.grad(Gibbs)(molalities, charges, pdict)


def solution2mz(solution):
    molalities = []
    ions = []
    for k, v in solution.items():
        molalities.append(v)
        ions.append(k)
    return jnp.array(molalities), np.array(ions)


pl = ParameterLibrary()
solution = {
    "Ca^{2+}": 1.0,
    "Cl^-": 2.0,
    "Na^+": 3.0,
    "HSO_4^-": 2.0,
    "Br^-": 1.0,
}
molalities, ions = solution2mz(solution)
charges = jnp.array(i2c(ions))
parameters = pl.get_parameters(ions)
g = Gibbs(molalities, charges, parameters)
acf = activity_coefficients(molalities, charges, parameters)

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
