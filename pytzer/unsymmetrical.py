# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Unsymmetrical mixing functions."""
import jax
from jax import numpy as np

# from scipy import integrate


def none(x):
    """Ignore unsymmetrical mixing."""
    return np.zeros_like(x)


# def numint(x):
#     """Evaluate unsymmetrical mixing function by numerical integration."""
#     # Cannot yet be automatically differentiated
#     # P91 Chapter 3 Eq. (B-12) [p123]
#     q = lambda x, y: -(x / y) * np.exp(-y)
#     J = np.full_like(x, np.nan)
#     for i, xi in enumerate(x):
#         # P91 Chapter 3 Eq. (B-13) [p123]
#         J[i] = (
#             integrate.quad(
#                 lambda y: (1 + q(xi, y) + q(xi, y) ** 2 / 2 - np.exp(q(xi, y)))
#                 * y ** 2,
#                 0,
#                 np.inf,
#             )[0]
#             / xi
#         )
#     return J


@jax.jit
def P75_eq46(x):
    """Evaluate unsymmetrical mixing function, following P75 eq. (46)."""
    # P75 Table III
    C = [
        4.118,
        7.247,
        -4.408,
        1.837,
        -0.251,
        0.0164,
    ]
    Jsum = np.zeros_like(x)
    for k in range(6):
        Jsum = Jsum + C[k] * x ** -(k + 1)
    return -(x ** 2) * np.log(x) * np.exp(-10 * x ** 2) / 6 + 1 / Jsum


@jax.jit
def P75_eq47(x):
    """Evaluate unsymmetrical mixing function, following P75 eq. (47)."""
    C = [
        4.0,
        4.581,
        0.7237,
        0.0120,
        0.528,
    ]
    J = x / (C[0] + C[1] * x ** -C[2] * np.exp(-C[3] * x ** C[4]))
    return J


@jax.jit
def _Harvie_raw(x):
    """Evaluate unsymmetrical mixing function and its derivative using
    Harvie's method, as described by P91 Ch. 3, pp. 124-125.
    """
    # Values from Table B-1, middle column (akI)
    akI = np.array(
        [
            -0.000000000010991,
            -0.000000000002563,
            0.000000000001943,
            0.000000000046333,
            -0.000000000050847,
            -0.000000000821969,
            0.000000001229405,
            0.000000013522610,
            -0.000000025267769,
            -0.000000202099617,
            0.000000396566462,
            0.000002937706971,
            -0.000004537895710,
            -0.000045036975204,
            0.000036583601823,
            0.000636874599598,
            0.000388260636404,
            -0.007299499690937,
            -0.029779077456514,
            -0.060076477753119,
            1.925154014814667,
        ]
    )
    # Values from Table B-1, final column (akII)
    akII = np.array(
        [
            0.000000000237816,
            -0.000000002849257,
            -0.000000006944757,
            0.000000004558555,
            0.000000080779570,
            0.000000216991779,
            -0.000000250453880,
            -0.000003548684306,
            -0.000004583768938,
            0.000034682122751,
            0.000087294451594,
            -0.000242107641309,
            -0.000887171310131,
            0.001130378079086,
            0.006519840398744,
            -0.001668087945272,
            -0.036552745910311,
            -0.028796057604906,
            0.150044637187895,
            0.462762985338493,
            0.628023320520852,
        ]
    )
    x_vec = np.full_like(akI, x)
    ak = np.where(x_vec < 1, akI, akII)
    z = np.where(
        x < 1, 4 * x ** 0.2 - 2, 40 / 9 * x ** -0.1 - 22 / 9  # Eq. (B-21)  # Eq. (B-25)
    )
    dz_dx = np.where(
        x < 1, 4 * x ** -0.8 / 5, -4 * x ** -1.1 / 9  # Eq. (B-22)  # Eq. (B-26)
    )
    b2, b1, b0 = 0.0, 0.0, 0.0
    d2, d1, d0 = 0.0, 0.0, 0.0
    for a in ak:
        b2 = b1 * 1
        b1 = b0 * 1
        d2 = d1 * 1
        d1 = d0 * 1
        b0 = z * b1 - b2 + a  # Eq. (B-23/27)
        d0 = b1 + z * d1 - d2  # Eq. (B-24/28)
    J = 0.25 * x - 1 + 0.5 * (b0 - b2)  # Eq. (B-29)
    Jp = 0.25 + 0.5 * dz_dx * (d0 - d2)  # Eq. (B-30)
    return J, Jp


@jax.custom_jvp
def Harvie(x):
    """Evaluate unsymmetrical mixing function using Harvie's method,
    as described by P91 Ch. 3, pp. 124-125.
    """
    return _Harvie_raw(x)[0]


@Harvie.defjvp
def _Harvie_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    ans, ans_dot = _Harvie_raw(x)
    return ans, ans_dot * x_dot
