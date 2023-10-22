# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2023  Matthew P. Humphreys  (GNU GPLv3)
"""HM92: Holmes & Mesmer (1992) J. Chem. Thermodyn. 24(3), 317-328.
https://doi.org/10.1016/S0021-9614(05)80072-2
"""

from jax import numpy as np
from ..convert import solute_to_charge as i2c


def HM92_eq26(T, p):
    Tn = T - 298.15
    return p[0] + p[1] * Tn + p[2] * Tn**2 + p[3] * Tn**3


def bC_H_HSO4_HM92(T, P):
    """c-a: hydrogen bisulfate [HM92]."""
    b0 = HM92_eq26(
        T,
        [
            0.2118,
            -0.6157e-3,
            0.29193e-5,
            -1.4153e-8,
        ],
    )
    b1 = HM92_eq26(
        T,
        [
            0.4177,
            0,
            -0.178e-5,
            0,
        ],
    )
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 298.15) & (T <= 473.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_SO4_HM92(T, P):
    """c-a: hydrogen sulfate [HM92]."""
    b0 = HM92_eq26(
        T,
        [
            0.0819,
            -1.214e-3,
            0,
            7.63e-8,
        ],
    )
    b1 = 0
    b2 = 0
    Cphi = HM92_eq26(
        T,
        [
            0.0637,
            0.353e-3,
            -1.530e-5,
            5.27e-8,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["SO4"])))
    C1 = 0
    alph1 = -9
    alph2 = -9
    omega = -9
    valid = (T >= 298.15) & (T <= 473.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_HSO4_SO4_HM92(T, P):
    """a-a': bisulfate sulfate [HM92]."""
    theta = HM92_eq26(
        T,
        [
            -0.1756,
            3.146e-3,
            -2.862e-5,
            6.786e-8,
        ],
    )
    valid = (T >= 298.15) & (T <= 473.15)
    return theta, valid
