# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Calculate Debye-Hueckel limiting slopes."""
from jax import numpy as np
from . import teos10
from .parameters import M88_eq13
from .constants import NA, dbar2Pa, dbar2MPa


def Aosm_M88(tempK, pres):
    """From Moller (1988)."""
    Aosm = M88_eq13(
        tempK,
        [
            3.36901532e-1,
            -6.32100430e-4,
            9.14252359e00,
            -1.35143986e-2,
            2.26089488e-3,
            1.92118597e-6,
            4.52586464e1,
            0,
        ],
    )
    valid = (tempK >= 273.15) & (tempK <= 573.15) & (pres == 10.1325)
    return Aosm, valid


def Aosm_CRP94(tempK, pres):
    """From Clegg et al. (1994) Appendix II."""
    # This function is long-winded to make it autograd-able w.r.t. temperature
    # Transform temperature:
    X = (2 * tempK - 607.3) / 139
    # Set parameters - CRP94 Table 11:
    a_Aosm = [
        0.797256081240 / 2,
        0.573389669896e-1,
        0.977632177788e-3,
        0.489973732417e-2,
        -0.313151784342e-2,
        0.179145971002e-2,
        -0.920584241844e-3,
        0.443862726879e-3,
        -0.203661129991e-3,
        0.900924147948e-4,
        -0.388189392385e-4,
        0.164245088592e-4,
        -0.686031972567e-5,
        0.283455806377e-5,
        -0.115641433004e-5,
        0.461489672579e-6,
        -0.177069754948e-6,
        0.612464488231e-7,
        -0.175689013085e-7,
    ]
    # Set up T "matrix" - CRP94 Eq. (AII2):
    Tmx00 = 1
    Tmx01 = X
    Tmx02 = 2 * X * Tmx01 - Tmx00
    Tmx03 = 2 * X * Tmx02 - Tmx01
    Tmx04 = 2 * X * Tmx03 - Tmx02
    Tmx05 = 2 * X * Tmx04 - Tmx03
    Tmx06 = 2 * X * Tmx05 - Tmx04
    Tmx07 = 2 * X * Tmx06 - Tmx05
    Tmx08 = 2 * X * Tmx07 - Tmx06
    Tmx09 = 2 * X * Tmx08 - Tmx07
    Tmx10 = 2 * X * Tmx09 - Tmx08
    Tmx11 = 2 * X * Tmx10 - Tmx09
    Tmx12 = 2 * X * Tmx11 - Tmx10
    Tmx13 = 2 * X * Tmx12 - Tmx11
    Tmx14 = 2 * X * Tmx13 - Tmx12
    Tmx15 = 2 * X * Tmx14 - Tmx13
    Tmx16 = 2 * X * Tmx15 - Tmx14
    Tmx17 = 2 * X * Tmx16 - Tmx15
    Tmx18 = 2 * X * Tmx17 - Tmx16
    # Solve for Aosm - CRP94 (E.AII1):
    Aosm = (
        Tmx00 * a_Aosm[0]
        + Tmx01 * a_Aosm[1]
        + Tmx02 * a_Aosm[2]
        + Tmx03 * a_Aosm[3]
        + Tmx04 * a_Aosm[4]
        + Tmx05 * a_Aosm[5]
        + Tmx06 * a_Aosm[6]
        + Tmx07 * a_Aosm[7]
        + Tmx08 * a_Aosm[8]
        + Tmx09 * a_Aosm[9]
        + Tmx10 * a_Aosm[10]
        + Tmx11 * a_Aosm[11]
        + Tmx12 * a_Aosm[12]
        + Tmx13 * a_Aosm[13]
        + Tmx14 * a_Aosm[14]
        + Tmx15 * a_Aosm[15]
        + Tmx16 * a_Aosm[16]
        + Tmx17 * a_Aosm[17]
        + Tmx18 * a_Aosm[18]
    )
    # Validity range:
    valid = (tempK >= 234.15) & (tempK <= 373.15) & (pres == 10.1325)
    return Aosm, valid


def Aosm_MarChemSpec25(tempK, pres):
    """For 298.15 K; value from Pitzer (1991) Chapter 3 Table 1 (page 99)."""
    Aosm = 0.3915
    valid = (tempK == 298.15) & (pres == 10.1325)
    return Aosm, valid


def Aosm_MarChemSpec05(tempK, pres):
    """For 278.15 K; value from FastPitz."""
    Aosm = 0.3792
    valid = (tempK == 278.15) & (pres == 10.1325)
    return Aosm, valid


def Aosm_MarChemSpec(tempK, pres):
    """Following CRP94, but with a correction to match AW90."""
    Aosm = Aosm_CRP94(tempK, pres)[0] + 2.99e-8
    valid = (tempK >= 234.15) & (tempK <= 373.15) & (pres == 10.1325)
    return Aosm, valid


def _gm1drho(tempK, presMPa):
    """AW90 Eq. (3): (g - 1) * rho."""
    # Produces values like in AW90 Fig. 1
    # tempK in K, pres in MPa
    # AW90 Table 2:
    b = [
        -4.044525e-2,
        103.6180,
        75.32165,
        -23.23778,
        -3.548184,
        -1246.311,
        263307.7,
        -6.928953e-1,
        -204.4473,
    ]
    # AW90 Eq. (3):
    gm1drho = (
        b[0] * presMPa / tempK
        + b[1] / np.sqrt(tempK)
        + b[2] / (tempK - 215)
        + b[3] / np.sqrt(tempK - 215)
        + b[4] / (tempK - 215) ** 0.25
        + np.exp(
            b[5] / tempK
            + b[6] / tempK ** 2
            + b[7] * presMPa / tempK
            + b[8] * presMPa / tempK ** 2
        )
    )
    return gm1drho


def _g(tempK, presMPa, rho):
    """Calculate g given density."""
    return _gm1drho(tempK, presMPa) * rho + 1


def _D(tempK, presMPa, rho):
    """Dielectric constant following Archer's DIEL()."""
    # Note that Archer's code uses different values from AW90 just in this
    # subroutine (so also different from in Aosm calculation below)
    Mw = 18.0153
    al = 1.444e-24
    k = 1.380658e-16
    mu = 1.84e-18
    A = (
        (al + _g(tempK, presMPa, rho) * mu ** 2 / (3 * k * tempK))
        * 4
        * np.pi
        * NA
        * rho
        / (3 * Mw)
    )
    return (1 + 9 * A + 3 * np.sqrt(9 * A ** 2 + 2 * A + 1)) / 4


def Aosm_AW90(tempK, pres):
    """D-H limiting slope for osmotic coefficient, following dhll.for."""
    presPa = pres * dbar2Pa
    presMPa = pres * dbar2MPa
    # Constants from Table 1 footnote:
    e = 1.6021773e-19  # charge on an electron in C
    E0 = 8.8541878e-12  # permittivity of vacuum in C**2/(J*m)
    # Constants from Table 2 footnote:
    k = 1.380658e-23  # Boltzmann constant in J/K
    rho = teos10.rho(tempK, presPa) * 1e-3
    Aosm = (
        np.sqrt(2e-3 * np.pi * rho * NA)
        * (100 * e ** 2 / (4 * np.pi * _D(tempK, presMPa, rho) * E0 * k * tempK)) ** 1.5
        / 3
    )
    valid = np.logical_and(
        (tempK >= 270.5 - presPa * 7.43e-8) & (tempK <= 313.15),
        (presPa >= 100) & (presPa <= 1e8),
    )
    return Aosm, valid
