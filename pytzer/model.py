# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
"""Calculate solution properties using the Pitzer model."""
import itertools, jax
from jax import numpy as np, lax
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
    charges = np.array([*z_cats, *z_anis, *np.zeros_like(m_neus)])
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
def Gibbs_map(
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
    def add_ca(i_ca):
        ic, ia = i_ca
        v_ca = ca[ic][ia]
        return (
            m_cats[ic]
            * m_anis[ia]
            * (
                2 * B(sqrt_I, v_ca[0], v_ca[1], v_ca[2], v_ca[5], v_ca[6])
                + Z * CT(sqrt_I, v_ca[3], v_ca[4], v_ca[7])
            )
        )

    def add_cc(i_xy):
        ix, iy = i_xy
        return (
            m_cats[ix]
            * m_cats[iy]
            * (cc[ix][iy] + etheta(Aphi, I, z_cats[ix], z_cats[iy], func_J=func_J))
        )

    def add_aa(i_xy):
        ix, iy = i_xy
        return (
            m_anis[ix]
            * m_anis[iy]
            * (aa[ix][iy] + etheta(Aphi, I, z_anis[ix], z_anis[iy], func_J=func_J))
        )

    def add_cca(i_cca):
        ix, iy, ia = i_cca
        return 0.5 * m_cats[ix] * m_cats[iy] * m_anis[ia] * cca[ix][iy][ia]

    def add_caa(i_caa):
        ic, ix, iy = i_caa
        return 0.5 * m_cats[ic] * m_anis[ix] * m_anis[iy] * caa[ic][ix][iy]

    def add_nc(i_nc):
        il, ic = i_nc
        return 2 * m_neus[il] * m_cats[ic] * nc[il][ic]

    def add_na(i_na):
        il, ia = i_na
        return 2 * m_neus[il] * m_anis[ia] * na[il][ia]

    def add_nca(i_nca):
        il, ic, ia = i_nca
        return m_neus[il] * m_cats[ic] * m_anis[ia] * nca[il][ic][ia]

    def add_nn(i_nn):
        ix, iy = i_nn
        return m_neus[ix] * m_neus[iy] * nn[ix][iy]

    def add_nnn(i_nnn):
        return m_neus[i_nnn] ** 3 * nnn[i_nnn]

    n_cats = len(m_cats)
    n_anis = len(m_anis)
    n_neus = len(m_neus)
    i_ca = np.array(list(itertools.product(range(n_cats), range(n_anis))))
    i_cc = np.array(list(itertools.product(range(n_cats), range(n_cats))))
    i_aa = np.array(list(itertools.product(range(n_anis), range(n_anis))))
    i_cca = np.array(
        list(itertools.product(range(n_cats), range(n_cats), range(n_anis)))
    )
    i_caa = np.array(
        list(itertools.product(range(n_cats), range(n_anis), range(n_anis)))
    )
    i_nc = np.array(list(itertools.product(range(n_neus), range(n_cats))))
    i_na = np.array(list(itertools.product(range(n_neus), range(n_anis))))
    i_nca = np.array(
        list(itertools.product(range(n_neus), range(n_cats), range(n_anis)))
    )
    i_nn = np.array(list(itertools.product(range(n_neus), range(n_neus))))
    i_nnn = np.array(list(range(n_neus)))

    I = ionic_strength(m_cats, z_cats) + ionic_strength(m_anis, z_anis)
    Z = ionic_z(m_cats, z_cats) + ionic_z(m_anis, z_anis)
    sqrt_I = np.sqrt(I)
    return (
        Gibbs_DH(Aphi, I)
        + np.sum(lax.map(add_ca, i_ca))
        + np.sum(lax.map(add_cc, i_cc))
        + np.sum(lax.map(add_aa, i_aa))
        + np.sum(lax.map(add_cca, i_cca))
        + np.sum(lax.map(add_caa, i_caa))
        + np.sum(lax.map(add_nc, i_nc))
        + np.sum(lax.map(add_na, i_na))
        + np.sum(lax.map(add_nca, i_nca))
        + np.sum(lax.map(add_nn, i_nn))
        + np.sum(lax.map(add_nnn, i_nnn))
    )


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
def log_activity_coeffs_cations(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters):
    return jax.grad(Gibbs_map, 0)(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters)


@jax.jit
def log_activity_coeffs_anions(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters):
    return jax.grad(Gibbs_map, 1)(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters)


@jax.jit
def log_activity_coeffs_neutrals(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters):
    return jax.grad(Gibbs_map, 2)(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters)


@jax.jit
def log_activity_coefficients_map(m_cats, m_anis, m_neus, z_cats, z_anis, **parameters):
    """Calculate the natural log of the activity coefficient of all solutes."""
    log_acf_cats = log_activity_coeffs_cations(
        m_cats, m_anis, m_neus, z_cats, z_anis, **parameters
    )
    log_acf_anis = log_activity_coeffs_anions(
        m_cats, m_anis, m_neus, z_cats, z_anis, **parameters
    )
    log_acf_neus = log_activity_coeffs_neutrals(
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
    """Calculate the osmotic coefficient of the solution."""
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
