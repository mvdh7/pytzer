# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
"""Unsymmetrical mixing functions."""
from jax import numpy as np
from scipy import integrate
from autograd.extend import primitive, defvjp


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


def P75_eq46(x):
    """Evaluate unsymmetrical mixing function following P75, eq. (46)."""
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


def P75_eq47(x):
    """Evaluate unsymmetrical mixing function following P75, eq. (47)."""
    C = [
        4.0,
        4.581,
        0.7237,
        0.0120,
        0.528,
    ]
    J = x / (C[0] + C[1] * x ** -C[2] * np.exp(-C[3] * x ** C[4]))
    return J


# ~~~~~~~ Harvie's method as described by Pitzer (1991) Ch. 3, pp. 124-125 ~~~~~
# Define the raw function - doesn't work in Pytzer (not autograd-able)
# Use Harvie() instead (code comes afterwards)
def Harvie(x):
    if x < 1.0:
        # Values from Table B-1, middle column (akI)
        ak = [
            1.925154014814667,
            -0.060076477753119,
            -0.029779077456514,
            -0.007299499690937,
            0.000388260636404,
            0.000636874599598,
            0.000036583601823,
            -0.000045036975204,
            -0.000004537895710,
            0.000002937706971,
            0.000000396566462,
            -0.000000202099617,
            -0.000000025267769,
            0.000000013522610,
            0.000000001229405,
            -0.000000000821969,
            -0.000000000050847,
            0.000000000046333,
            0.000000000001943,
            -0.000000000002563,
            -0.000000000010991,
        ]
        z = 4 * x ** 0.2 - 2  # Eq. (B-21)
        dz_dx = 4 * x ** -0.8 / 5  # Eq. (B-22)
    else:
        # Values from Table B-1, final column (akII)
        ak = [
            0.628023320520852,
            0.462762985338493,
            0.150044637187895,
            -0.028796057604906,
            -0.036552745910311,
            -0.001668087945272,
            0.006519840398744,
            0.001130378079086,
            -0.000887171310131,
            -0.000242107641309,
            0.000087294451594,
            0.000034682122751,
            -0.000004583768938,
            -0.000003548684306,
            -0.000000250453880,
            0.000000216991779,
            0.000000080779570,
            0.000000004558555,
            -0.000000006944757,
            -0.000000002849257,
            0.000000000237816,
        ]
        z = 40 / 9 * x ** -0.1 - 22 / 9  # Eq. (B-25)
        dz_dx = -4 * x ** -1.1 / 9  # Eq. (B-26)
    bk2 = 0.0
    bk1 = 0.0
    dk2 = 0.0
    dk1 = 0.0
    for i in reversed(range(21)):
        bk0 = z * bk1 - bk2 + ak[i]  # Eq. (B-23/27)
        dk0 = bk1 + z * dk1 - dk2  # Eq. (B-24/28)
        bk2 = bk1 * 1
        bk1 = bk0 * 1
        dk2 = dk1 * 1
        dk1 = dk0 * 1
    J = 0.25 * x - 1 + 0.5 * (bk0 - bk2)  # Eq. (B-29)
    Jp = 0.25 + 0.5 * dz_dx * (dk0 - dk2)  # Eq. (B-30)
    return J , Jp


# # Define the function to use in the model
# @primitive
# def Harvie(x):
#     """Evaluate unsymmetrical mixing function using Harvie's method."""
#     return _Harvie_raw(x)[0]


# # Set up its derivative for autograd
# def _Harvie_vjp(ans, x):
#     return lambda g: g * _Harvie_raw(x)[1]


# defvjp(Harvie, _Harvie_vjp)
