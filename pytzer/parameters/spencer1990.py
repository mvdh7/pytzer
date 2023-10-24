# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2023  Matthew P. Humphreys  (GNU GPLv3)
"""SMW90: Spencer et al. (1990) Geochim. Cosmochim. Acta 54(3), 575-590.
https://doi.org/10.1016/0016-7037(90)90354-N

Not all parameters in the paper have yet been added here!
"""

from jax import numpy as np
from ..convert import solute_to_charge as i2c


def SMW90_eq2(T, a):
    """a contains the coefficients [a1, a2, a6, a9, a3, a4]."""
    return a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3 + a[4] / T + a[5] * np.log(T)


def bC_Mg_Cl_SMW90(T, P):
    """c-a: magnesium chloride [SMW90]."""
    b0 = SMW90_eq2(
        T,
        [
            3.13852913e2,
            2.61769099e-1,
            -2.46268460e-4,
            1.15764787e-7,
            -5.53133381e3,
            -6.21616862e1,
        ],
    )
    b1 = SMW90_eq2(
        T,
        [
            -3.18432525e4,
            -2.86710358e1,
            2.78892838e-2,
            -1.3279705e-5,
            5.24032958e5,
            6.40770396e3,
        ],
    )
    b2 = 0
    Cphi = SMW90_eq2(
        T,
        [
            5.9532e-2,
            -2.49949e-4,
            2.41831e-7,
            0,
            0,
            0,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T > 273.15 - 54) & (T <= 298.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_SO4_SMW90(T, P):
    """c-a: magnesium sulfate [SMW90]."""
    b0 = SMW90_eq2(
        T,
        [
            5.40007849e3,
            4.90576884e0,
            -4.80489750e-3,
            2.31126994e-6,
            -8.80664146e4,
            -1.08839565e3,
        ],
    )
    b1 = SMW90_eq2(
        T,
        [
            2.78730869e0,
            4.30077440e-3,
            0,
            0,
            0,
            0,
        ],
    )
    b2 = 0
    Cphi = SMW90_eq2(
        T,
        [
            -5.88623653e2,
            -5.05522880e-1,
            4.8277657e-4,
            -2.3029838e-7,
            1.02002016e4,
            1.17303808e2,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["SO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T > 273.15 - 54) & (T <= 298.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def psi_Mg_Cl_SO4_SMW90(T, P):
    """c-a-a': magnesium chloride sulfate [SMW90]."""
    psi = SMW90_eq2(
        T,
        [
            -1.839158e-1,
            1.429444e-4,
            0,
            0,
            3.263e1,
            0,
        ],
    )
    valid = (T > 273.15 - 54) & (T <= 298.15)
    return psi, valid
