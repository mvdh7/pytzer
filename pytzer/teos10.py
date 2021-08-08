# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Calculate properties of pure water."""
import jax
from jax import numpy as np

# Properties of pure water
# Source: http://www.teos-10.org/pubs/IAPWS-2009-Supplementary.pdf
# Validity: 100 < presPa < 1e8 Pa; (270.5 - presPa*7.43e-8) < tempK < 313.15 K
# Seawater available from http://www.teos-10.org/pubs/IAPWS-08.pdf


@jax.jit
def Gibbs(tempK, presPa):
    """Gibbs energy function."""
    # Coefficients of the Gibbs function as defined in Table 2:
    Gdict = {
        (0, 0): 0.101342743139674e3,
        (3, 2): 0.499360390819152e3,
        (0, 1): 0.100015695367145e6,
        (3, 3): -0.239545330654412e3,
        (0, 2): -0.254457654203630e4,
        (3, 4): 0.488012518593872e2,
        (0, 3): 0.284517778446287e3,
        (3, 5): -0.166307106208905e1,
        (0, 4): -0.333146754253611e2,
        (4, 0): -0.148185936433658e3,
        (0, 5): 0.420263108803084e1,
        (4, 1): 0.397968445406972e3,
        (0, 6): -0.546428511471039,
        (4, 2): -0.301815380621876e3,
        (1, 0): 0.590578347909402e1,
        (4, 3): 0.152196371733841e3,
        (1, 1): -0.270983805184062e3,
        (4, 4): -0.263748377232802e2,
        (1, 2): 0.776153611613101e3,
        (5, 0): 0.580259125842571e2,
        (1, 3): -0.196512550881220e3,
        (5, 1): -0.194618310617595e3,
        (1, 4): 0.289796526294175e2,
        (5, 2): 0.120520654902025e3,
        (1, 5): -0.213290083518327e1,
        (5, 3): -0.552723052340152e2,
        (2, 0): -0.123577859330390e5,
        (5, 4): 0.648190668077221e1,
        (2, 1): 0.145503645404680e4,
        (6, 0): -0.189843846514172e2,
        (2, 2): -0.756558385769359e3,
        (6, 1): 0.635113936641785e2,
        (2, 3): 0.273479662323528e3,
        (6, 2): -0.222897317140459e2,
        (2, 4): -0.555604063817218e2,
        (6, 3): 0.817060541818112e1,
        (2, 5): 0.434420671917197e1,
        (7, 0): 0.305081646487967e1,
        (3, 0): 0.736741204151612e3,
        (7, 1): -0.963108119393062e1,
        (3, 1): -0.672507783145070e3,
    }
    # Convert temperature and pressure:
    ctau = (tempK - 273.15) / 40
    cpi = (presPa - 101325) / 1e8
    # Initialise with zero and increment following Eq. (1):
    Gsum = 0
    for j in range(8):
        for k in range(7):
            if (j, k) in Gdict.keys():
                Gsum = Gsum + Gdict[(j, k)] * ctau ** j * cpi ** k
    return Gsum


# Get differentials
gt = jax.grad(Gibbs, argnums=0)
gp = jax.grad(Gibbs, argnums=1)
gtt = jax.grad(gt, argnums=0)
gtp = jax.grad(gt, argnums=1)
gpp = jax.grad(gp, argnums=1)

# Define functions for solution properties
def rho(tempK, presPa):
    """Density in kg/m**3."""
    # Table 3, Eq. (4)
    return 1 / gp(tempK, presPa)


def s(tempK, presPa):
    """Specific entropy in J/(kg*K)."""
    # Table 3, Eq. (5)
    return -gt(tempK, presPa)


def cp(tempK, presPa):
    """Specific isobaric heat capacity in J/(kg*K)."""
    # Table 3, Eq. (6)
    return -tempK * gtt(tempK, presPa)


def h(tempK, presPa):
    """Specific enthalpy in J/kg."""
    # Table 3, Eq. (7)
    return Gibbs(tempK, presPa) + tempK * s(tempK, presPa)


def u(tempK, presPa):
    """Specific internal energy in J/kg."""
    # Table 3, Eq. (8)
    return Gibbs(tempK, presPa) + tempK * s(tempK, presPa) - presPa * gp(tempK, presPa)


def f(tempK, presPa):
    """Specific Helmholtz energy in J/kg."""
    # Table 3, Eq. (9)
    return Gibbs(tempK, presPa) - presPa * gp(tempK, presPa)


def alpha(tempK, presPa):
    """Thermal expansion coefficient in 1/K."""
    # Table 3, Eq. (10)
    return gtp(tempK, presPa) / gp(tempK, presPa)


def bs(tempK, presPa):
    """Isentropic temp.-presPas. coefficient, adiabatic lapse rate in K/Pa."""
    # Table 3, Eq. (11)
    return -gtp(tempK, presPa) / gp(tempK, presPa)


def kt(tempK, presPa):
    """Isothermal compresPasibility in 1/Pa."""
    # Table 3, Eq. (12)
    return -gpp(tempK, presPa) / gp(tempK, presPa)


def ks(tempK, presPa):
    """Isentropic compresPasibility in 1/Pa."""
    # Table 3, Eq. (13)
    return (gtp(tempK, presPa) ** 2 - gtt(tempK, presPa) * gpp(tempK, presPa)) / (
        gp(tempK, presPa) * gtt(tempK, presPa)
    )


def w(tempK, presPa):
    """Speed of sound in m/s."""
    # Table 3, Eq. (14)
    return gp(tempK, presPa) * np.sqrt(
        gtt(tempK, presPa)
        / (gtp(tempK, presPa) ** 2 - gtt(tempK, presPa) * gpp(tempK, presPa))
    )
