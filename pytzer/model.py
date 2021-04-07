# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Calculate solution properties using the Pitzer model."""
import itertools, jax
from collections import OrderedDict
from jax import numpy as np, lax
from .constants import b, Mw
from .libraries import Seawater
from . import convert, properties, unsymmetrical


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


func_J = unsymmetrical.Harvie


@jax.jit
def Gibbs_nRT(
    solutes,
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
    **parameters_extra
):
    """Calculate the excess Gibbs energy of a solution divided by n*R*T."""

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

    # Get the molalities of cations, anions and neutrals in separate arrays
    m_cats = np.array([m for s, m in solutes.items() if s in convert.all_cations])
    m_anis = np.array([m for s, m in solutes.items() if s in convert.all_anions])
    m_neus = np.array([m for s, m in solutes.items() if s in convert.all_neutrals])
    # Get the charges of cations and anions in separate arrays
    s2c = convert.solute_to_charge
    z_cats = np.array([s2c[s] for s in solutes.keys() if s in convert.all_cations])
    z_anis = np.array([s2c[s] for s in solutes.keys() if s in convert.all_anions])
    # Evaluate terms dependent on overall ionic strength
    I = (
        np.sum(
            np.array([m * convert.solute_to_charge[s] ** 2 for s, m in solutes.items()])
        )
        / 2
    )
    Z = np.sum(
        np.abs(np.array([m * convert.solute_to_charge[s] for s, m in solutes.items()]))
    )
    sqrt_I = np.sqrt(I)
    Gibbs = Gibbs_DH(Aphi, I)
    # Add Pitzer model interaction terms
    r_cats = range(len(m_cats))
    r_anis = range(len(m_anis))
    r_neus = range(len(m_neus))
    if len(m_cats) > 0 and len(m_anis) > 0:
        i_ca = np.array(list(itertools.product(r_cats, r_anis)))
        i_cc = np.array(list(itertools.product(r_cats, r_cats)))
        i_aa = np.array(list(itertools.product(r_anis, r_anis)))
        i_cca = np.array(list(itertools.product(r_cats, r_cats, r_anis)))
        i_caa = np.array(list(itertools.product(r_cats, r_anis, r_anis)))
        Gibbs = (
            Gibbs
            + np.sum(lax.map(add_ca, i_ca))
            + np.sum(lax.map(add_cc, i_cc))
            + np.sum(lax.map(add_aa, i_aa))
            + np.sum(lax.map(add_cca, i_cca))
            + np.sum(lax.map(add_caa, i_caa))
        )
        if len(m_neus) > 0:
            i_nca = np.array(list(itertools.product(r_neus, r_cats, r_anis)))
            Gibbs = Gibbs + np.sum(lax.map(add_nca, i_nca))
    if len(m_neus) > 0:
        i_nn = np.array(list(itertools.product(r_neus, r_neus)))
        i_nnn = np.array(list(r_neus))
        Gibbs = Gibbs + np.sum(lax.map(add_nn, i_nn)) + np.sum(lax.map(add_nnn, i_nnn))
        if len(m_cats) > 0:
            i_nc = np.array(list(itertools.product(r_neus, r_cats)))
            Gibbs = Gibbs + np.sum(lax.map(add_nc, i_nc))
        if len(m_anis) > 0:
            i_na = np.array(list(itertools.product(r_neus, r_anis)))
            Gibbs = Gibbs + np.sum(lax.map(add_na, i_na))
    return Gibbs


@jax.jit
def log_activity_coefficients(solutes, **parameters):
    """Calculate the natural log of the activity coefficient of all solutes."""
    return jax.grad(Gibbs_nRT)(solutes, **parameters)


@jax.jit
def activity_coefficients(solutes, **parameters):
    """Calculate the activity coefficient of all solutes."""
    log_acfs = log_activity_coefficients(solutes, **parameters)
    return {k: np.exp(v) for k, v in log_acfs.items()}


@jax.jit
def osmotic_coefficient(solutes, **parameters):
    """Calculate the osmotic coefficient of the solution."""
    return 1 - jax.grad(
        lambda ww: ww
        * Gibbs_nRT(OrderedDict({s: m / ww for s, m in solutes.items()}), **parameters)
    )(1.0) / np.sum(np.array([m for m in solutes.values()]))


@jax.jit
def log_activity_water(solutes, **parameters):
    """Calculate the natural log of the water activity."""
    return (
        jax.grad(
            lambda ww: ww
            * Gibbs_nRT(
                OrderedDict({s: m / ww for s, m in solutes.items()}), **parameters
            )
        )(1.0)
        - np.sum(np.array([m for m in solutes.values()]))
    ) * Mw


@jax.jit
def activity_water(solutes, **parameters):
    """Calculate the water activity."""
    return np.exp(log_activity_water(solutes, **parameters))
