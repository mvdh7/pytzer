# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
"""Calculate solution properties using the Pitzer model."""
import jax
from jax import numpy as np
from .constants import b, Mw
from .libraries import Seawater
from . import properties, unsymmetrical


def Gibbs_DH(Aosm, I):
    """The Debye-Hueckel component of the excess Gibbs energy,
    following CRP94 Eq. (AI1).
    """
    return -4 * Aosm * I * np.log(1 + b * np.sqrt(I)) / b


def g(x):
    """g function, following CRP94 Eq. (AI13)."""
    return 2 * (1 - (1 + x) * np.exp(-x)) / x ** 2


def h(x):
    """h function, following CRP94 Eq. (AI15)."""
    return (6 - (6 + x * (6 + 3 * x + x ** 2)) * np.exp(-x)) / x ** 4


def B(sqrt_I, b0, b1, b2, alph1, alph2):
    """B function, following CRP94 Eq. (AI7)."""
    return b0 + b1 * g(alph1 * sqrt_I) + b2 * g(alph2 * sqrt_I)


def CT(sqrt_I, C0, C1, omega):
    """CT function, following CRP94 Eq. (AI10)."""
    return C0 + 4 * C1 * h(omega * sqrt_I)


def xij(Aosm, I, z0, z1):
    """xij function for unsymmetrical mixing."""
    return 6 * z0 * z1 * Aosm * np.sqrt(I)


def etheta(Aosm, I, z0, z1, func_J):
    """etheta function for unsymmetrical mixing."""
    x00 = xij(Aosm, I, z0, z0)
    x01 = xij(Aosm, I, z0, z1)
    x11 = xij(Aosm, I, z1, z1)
    return z0 * z1 * (func_J(x01) - 0.5 * (func_J(x00) + func_J(x11))) / (4 * I)


@jax.jit
def ionic_strength(molalities, charges):
    """Ionic strength."""
    return 0.5 * np.sum(molalities * charges ** 2)


@jax.jit
def ionic_z(molalities, charges):
    """Z function."""
    return np.sum(molalities * np.abs(charges))


@jax.jit
def Gibbs_nRT(
    molalities,
    charges,
    Aosm=None,
    ca=None,
    cc=None,
    aa=None,
    cca=None,
    caa=None,
    nc=None,
    nn=None,
    na=None,
    nca=None,
    nnn=None,
    func_J=unsymmetrical.P75_eq47,
):
    """Calculate the excess Gibbs energy of a solution divided by n*R*T."""
    # Note that oceanographers record ocean pressure as only due to the water,
    # so at the sea surface pressure = 0 dbar, but the atmospheric pressure
    # should also be taken into account for this model.
    # Ionic strength etc.
    I = ionic_strength(molalities, charges)
    Z = ionic_z(molalities, charges)
    sqrt_I = np.sqrt(I)
    m_cats = molalities[charges > 0]
    m_anis = molalities[charges < 0]
    m_neus = molalities[charges == 0]
    # Split up charges
    z_cats = charges[charges > 0]
    z_anis = charges[charges < 0]
    # Begin with Debye-Hueckel component
    Gibbs_nRT = Gibbs_DH(Aosm, I)
    # Loop through cations
    for CX, m_cat_x in enumerate(m_cats):
        # Add c-a interactions
        for A, m_ani in enumerate(m_anis):
            Gibbs_nRT = Gibbs_nRT + m_cat_x * m_ani * (
                2 * B(sqrt_I, *ca[CX][A]) + Z * CT(sqrt_I, *ca[CX][A])
            )
        # Add c-c' interactions
        for _CY, m_cat_y in enumerate(m_cats[CX + 1 :]):
            CY = _CY + CX + 1
            Gibbs_nRT = Gibbs_nRT + m_cat_x * m_cat_y * 2 * cc[CX][CY]
            # Unsymmetrical mixing terms
            if z_cats[CX] != z_cats[CY]:
                Gibbs_nRT = Gibbs_nRT + m_cat_x * m_cat_y * 2 * etheta(
                    Aosm, I, z_cats[CX], z_cats[CY], func_J
                )
            # Add c-c'-a interactions
            for A, m_ani in enumerate(m_anis):
                Gibbs_nRT = Gibbs_nRT + m_cat_x * m_cat_y * m_ani * cca[CX][CY][A]
    # Loop through anions
    for AX, m_ani_x in enumerate(m_anis):
        # Add a-a' interactions
        for _AY, m_ani_y in enumerate(m_anis[AX + 1 :]):
            AY = _AY + AX + 1
            Gibbs_nRT = Gibbs_nRT + m_ani_x * m_ani_y * 2 * aa[AX][AY]
            # Unsymmetrical mixing terms
            if z_anis[AX] != z_anis[AY]:
                Gibbs_nRT = Gibbs_nRT + m_ani_x * m_ani_y * 2 * etheta(
                    Aosm, I, z_anis[AX], z_anis[AY], func_J
                )
            # Add c-a-a' interactions
            for C, m_cat in enumerate(m_cats):
                Gibbs_nRT = Gibbs_nRT + m_ani_x * m_ani_y * m_cat * caa[C][AX][AY]
    # Add neutral interactions
    for NX, m_neu_x in enumerate(m_neus):
        # Add n-c interactions
        for C, m_cat in enumerate(m_cats):
            Gibbs_nRT = Gibbs_nRT + m_neu_x * m_cat * 2 * nc[NX][C]
            # Add n-c-a interactions
            for A, m_ani in enumerate(m_anis):
                Gibbs_nRT = Gibbs_nRT + m_neu_x * m_cat * m_ani * nca[NX][C][A]
        # Add n-a interactions
        for A, m_ani in enumerate(m_anis):
            Gibbs_nRT = Gibbs_nRT + m_neu_x * m_ani * 2 * na[NX][A]
        # n-n' excluding n-n
        for _NY, m_neu_y in enumerate(m_neus[NX + 1 :]):
            NY = _NY + NX + 1
            Gibbs_nRT = Gibbs_nRT + m_neu_x * m_neu_y * 2 * nn[NX][NY]
        # n-n
        Gibbs_nRT = Gibbs_nRT + m_neu_x ** 2 * nn[NX][NX]
        # n-n-n
        Gibbs_nRT = Gibbs_nRT + m_neu_x ** 3 * nnn[NX]
    return Gibbs_nRT


@jax.jit
def log_activity_coefficients(molalities, charges, **parameters):
    """Calculate the natural log of the activity coefficient of all solutes."""
    return jax.grad(Gibbs_nRT)(molalities, charges, **parameters)


@jax.jit
def activity_coefficients(molalities, charges, **parameters):
    """Calculate the activity coefficient of all solutes."""
    return np.exp(log_activity_coefficients(molalities, charges, **parameters))


# def acfs(mols, ions, tempK, pres, prmlib=Seawater, Izero=False):
#     """Calculate the activity coefficients of all solutes."""
#     return np.exp(ln_acfs(mols, ions, tempK, pres, prmlib, Izero))


# def ln_acf2ln_acf_MX(ln_acfM, ln_acfX, nM, nX):
#     """Calculate the mean activity coefficient for an electrolyte."""
#     return (nM * ln_acfM + nX * ln_acfX) / (nM + nX)


# # Osmotic coefficient
# def osm(mols, ions, tempK, pres, prmlib=Seawater, Izero=False):
#     """Calculate the osmotic coefficient."""
#     ww = full_like(tempK, 1.0)
#     return 1 - egrad(
#         lambda ww: ww * Gibbs_nRT(mols / ww, ions, tempK, pres, prmlib, Izero)
#     )(ww) / np.sum(mols, axis=0)


# # Water activity
# def lnaw(mols, ions, tempK, pres, prmlib=Seawater, Izero=False):
#     """Calculate the natural log of the water activity."""
#     ww = full_like(tempK, 1.0)
#     return (
#         egrad(lambda ww: ww * Gibbs_nRT(mols / ww, ions, tempK, pres, prmlib, Izero))(
#             ww
#         )
#         - np.sum(mols, axis=0)
#     ) * Mw


# def aw(mols, ions, tempK, pres, prmlib=Seawater, Izero=False):
#     """Calculate the water activity."""
#     return np.exp(lnaw(mols, ions, tempK, pres, prmlib, Izero))


# # Conversions
# def osm2aw(mols, osm):
#     """Convert osmotic coefficient to water activity."""
#     return np.exp(-osm * Mw * np.sum(mols, axis=0))


# def aw2osm(mols, aw):
#     """Convert water activity to osmotic coefficient."""
#     return -np.log(aw) / (Mw * np.sum(mols, axis=0))
