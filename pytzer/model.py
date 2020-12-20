# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
"""Calculate solution properties using the Pitzer model."""
import jax
from jax import numpy as np
from .constants import b, Mw
from .libraries import Seawater
from . import properties, unsymmetrical


def Gibbs_DH(Aphi, I):
    """The Debye-Hueckel component of the excess Gibbs energy,
    following CRP94 Eq. (AI1).
    """
    return -4 * Aphi * I * np.log(1 + b * np.sqrt(I)) / b


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


def xij(Aphi, I, z0, z1):
    """xij function for unsymmetrical mixing."""
    return 6 * z0 * z1 * Aphi * np.sqrt(I)


def etheta(Aphi, I, z0, z1, func_J=unsymmetrical.Harvie):
    """etheta function for unsymmetrical mixing."""
    x00 = xij(Aphi, I, z0, z0)
    x01 = xij(Aphi, I, z0, z1)
    x11 = xij(Aphi, I, z1, z1)
    return z0 * z1 * (func_J(x01) - 0.5 * (func_J(x00) + func_J(x11))) / (4 * I)


def ionic_strength(molalities, charges):
    """Ionic strength."""
    return 0.5 * np.sum(molalities * charges ** 2)


def ionic_z(molalities, charges):
    """Z function."""
    return np.sum(molalities * np.abs(charges))


def split_molalities_charges(molalities, charges):
    """Split up molalities and charges inputs as required by other functions."""
    m_cats = np.compress(charges > 0, molalities)
    m_anis = np.compress(charges < 0, molalities)
    m_neus = np.compress(charges == 0, molalities)
    z_cats = np.compress(charges > 0, charges)
    z_anis = np.compress(charges < 0, charges)
    return m_cats, m_anis, m_neus, z_cats, z_anis


# Temporary way to allow adjustment of func_J (not very robust)
func_J = unsymmetrical.Harvie


@jax.jit
def Gibbs_nRT(
    m_cats,
    m_anis,
    m_neus,
    z_cats,
    z_anis,
    Aphi=None,
    ca=None,
    cc=None,
    aa=None,
    cca=None,
    caa=None,
    nc=None,
    na=None,
    nn=None,
    nca=None,
    nnn=None,
):
    """Calculate the excess Gibbs energy of a solution divided by n*R*T."""
    # Note that oceanographers record ocean pressure as only due to the water,
    # so at the sea surface pressure = 0 dbar, but the atmospheric pressure
    # should also be taken into account for this model.
    # Ionic strength etc.
    molalities = np.array([*m_cats, *m_anis, *m_neus])
    charges = np.array([*z_cats, *z_anis])
    I = ionic_strength(molalities, charges)
    Z = ionic_z(molalities, charges)
    sqrt_I = np.sqrt(I)
    # Begin with Debye-Hueckel component
    Gibbs_nRT = Gibbs_DH(Aphi, I)
    # Loop through cations
    for CX, m_cat_x in enumerate(m_cats):
        # Add c-a interactions
        for A, m_ani in enumerate(m_anis):
            v_ca = ca[CX][A]
            Gibbs_nRT = Gibbs_nRT + m_cat_x * m_ani * (
                2 * B(sqrt_I, v_ca[0], v_ca[1], v_ca[2], v_ca[5], v_ca[6])
                + Z * CT(sqrt_I, v_ca[3], v_ca[4], v_ca[7])
            )
        # Add c-c' interactions
        for _CY, m_cat_y in enumerate(m_cats[CX + 1 :]):
            CY = _CY + CX + 1
            Gibbs_nRT = Gibbs_nRT + m_cat_x * m_cat_y * 2 * cc[CX][CY]
            # # Unsymmetrical mixing terms
            Gibbs_nRT = Gibbs_nRT + m_cat_x * m_cat_y * 2 * etheta(
                Aphi, I, z_cats[CX], z_cats[CY], func_J=func_J
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
            Gibbs_nRT = Gibbs_nRT + m_ani_x * m_ani_y * 2 * etheta(
                Aphi, I, z_anis[AX], z_anis[AY], func_J=func_J
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
def log_activity_coefficients(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters):
    """Calculate the natural log of the activity coefficient of all solutes."""
    log_acf_cats = jax.grad(Gibbs_nRT, 0)(
        m_cats, m_anis, m_neus, z_cats, z_anis, **parameters
    )
    log_acf_anis = jax.grad(Gibbs_nRT, 1)(
        m_cats, m_anis, m_neus, z_cats, z_anis, **parameters
    )
    log_acf_neus = jax.grad(Gibbs_nRT, 2)(
        m_cats, m_anis, m_neus, z_cats, z_anis, **parameters
    )
    return log_acf_cats, log_acf_anis, log_acf_neus


@jax.jit
def activity_coefficients(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters):
    """Calculate the activity coefficient of all solutes."""
    log_acfs = log_activity_coefficients(
        m_cats, m_anis, m_neus, z_cats, z_anis, **parameters
    )
    return [np.exp(log_acf) for log_acf in log_acfs]


@jax.jit
def osmotic_coefficient(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters):
    """Calculate the osmotic coefficient."""
    return 1 - jax.grad(
        lambda ww: ww
        * Gibbs_nRT(m_cats / ww, m_anis / ww, m_neus / ww, z_cats, z_anis, **parameters)
    )(1.0) / (np.sum(m_cats) + np.sum(m_anis) + np.sum(m_neus))


@jax.jit
def log_activity_water(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters):
    """Calculate the natural log of the water activity."""
    return (
        jax.grad(
            lambda ww: ww
            * Gibbs_nRT(
                m_cats / ww, m_anis / ww, m_neus / ww, z_cats, z_anis, **parameters
            )
        )(1.0)
        - (np.sum(m_cats) + np.sum(m_anis) + np.sum(m_neus))
    ) * Mw


@jax.jit
def activity_water(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters):
    """Calculate the water activity."""
    return np.exp(
        log_activity_water(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters)
    )
