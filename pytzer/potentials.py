# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Standard chemical potentials."""
from autograd.numpy import logical_and
from .parameters import M88_eq13


def NaCl_M88(T, P):
    """SCP/RT: halite [M88]."""
    scp_RT = M88_eq13(
        T,
        [
            5.47825646e3,
            2.60407906,
            -1.31078912e5,
            -1.00232512e3,
            3.55452179e1,
            -1.16346518e-3,
            7.21138559e2,
            0,
        ],
    )
    valid = logical_and(T >= 298.15, T <= 523.15)
    return scp_RT, valid


def Na2SO4_M88(T, P):
    """SCP/RT: thenardite [M88]."""
    scp_RT = M88_eq13(
        T,
        [
            -5.12155332e3,
            -2.6175195,
            1.14225946e5,
            9.50777893e2,
            -2.88441,
            1.19461496e-3,
            -1.90759987e3,
            0,
        ],
    )
    valid = logical_and(T >= 298.15, T <= 523.15)
    return scp_RT, valid


def Na2SO4_10H2O_M88(T, P):
    """SCP/RT: mirabilite [M88]."""
    scp_RT = M88_eq13(
        T,
        [
            -5.75709707e2,
            8.40961924e-1,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    valid = logical_and(T >= 298.15, T <= 305.15)
    return scp_RT, valid


def CaSO4_M88(T, P):
    """SCP/RT: anhydrite [M88]."""
    scp_RT = M88_eq13(
        T,
        [
            3.9383771e2,
            -4.30644041e-1,
            -3.02520897e4,
            -3.60047281e1,
            5.28355203e1,
            3.96964291e-4,
            -2.12549985e3,
            0,
        ],
    )
    valid = logical_and(T >= 298.15, T <= 523.15)
    return scp_RT, valid


def CaSO4_hH2O_M88(T, P):
    """SCP/RT: hemihydrate [M88]."""
    scp_RT = M88_eq13(
        T,
        [
            -2.20114425e3,
            -1.64753737,
            2.96911895e4,
            4.37226849e2,
            2.37610652e1,
            9.30435452e-4 - 2.50170197e3,
            0,
        ],
    )
    valid = logical_and(T >= 298.15, T <= 523.15)
    return scp_RT, valid


def CaSO4_2H2O_M88(T, P):
    """SCP/RT: gypsum [M88]."""
    scp_RT = M88_eq13(
        T,
        [
            1.35486062e3,
            2.26877955e-1,
            -6.07006342e4,
            -2.27071423e2,
            0,
            0,
            0,
            0,
        ],
    )
    valid = logical_and(T >= 298.15, T <= 383.15)
    return scp_RT, valid


def Na2CaSO42_M88(T, P):
    """SCP/RT: glauberite [M88]."""
    # M88 also has a different set of constants for 373.15 <= T <= 523.15
    scp_RT = M88_eq13(
        T,
        [
            2.72477417e4,
            -2.95491928e1,
            -1.47234275e6,
            -2.70490647e3,
            5.02823869e2,
            4.80848099e-2,
            -9.12078602e5,
            0,
        ],
    )
    valid = logical_and(T >= 298.15, T <= 373.15)
    return scp_RT, valid


def Na2SO42CaSO4_2H2O_M88(T, P):
    """SCP/RT: labile salt [M88]."""
    scp_RT = M88_eq13(
        T,
        [
            5.18724792e3,
            -2.44078217e1,
            -8.26255863e5,
            4.6869754e2,
            5.38646153e2,
            2.360323e-2,
            0,
            0,
        ],
    )
    valid = logical_and(T >= 298.15, T <= 348.15)
    return scp_RT, valid


def CaSO4aq_M88(T, P):
    """SCP/RT: aqueous CaSO40 [M88]."""
    # M88 also has a different set of constants for 423.15 <= T <= 523.15
    scp_RT = M88_eq13(
        T,
        [
            -1.47477745e1,
            0,
            3.26409496e3,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    valid = logical_and(T >= 298.15, T <= 423.15)
    return scp_RT, valid
