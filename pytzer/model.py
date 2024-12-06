# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
"""Calculate solution properties using the Pitzer model."""

import jax
from jax import numpy as np

from . import libraries
from .constants import b_pitzer, mass_water

library = libraries.lib_CWTD23.library


def Gibbs_DH(Aphi, ionic_strength):
    """The Debye-Hueckel component of the excess Gibbs energy following CRP94 eq. (AI1).

    Parameters
    ----------
    Aphi : float
        Debye-Hueckel limiting slope for the osmotic coefficient.
    I : float
        Ionic strength of the solution in mol/kg.

    Returns
    -------
    float
        Debye-Hueckel component of the excess Gibbs energy.
    """
    return (
        -4
        * Aphi
        * ionic_strength
        * np.log(1 + b_pitzer * np.sqrt(ionic_strength))
        / b_pitzer
    )


def g(x):
    """g function, following CRP94 Eq. (AI13)."""
    return 2 * (1 - (1 + x) * np.exp(-x)) / x**2


def h(x):
    """h function, following CRP94 Eq. (AI15)."""
    return (6 - (6 + x * (6 + 3 * x + x**2)) * np.exp(-x)) / x**4


def B(sqrt_I, b0, b1, b2, alph1, alph2):
    """B function, following CRP94 Eq. (AI7)."""
    return b0 + b1 * g(alph1 * sqrt_I) + b2 * g(alph2 * sqrt_I)


def CT(sqrt_I, C0, C1, omega):
    """CT function, following CRP94 Eq. (AI10)."""
    return C0 + 4 * C1 * h(omega * sqrt_I)


def xij(Aphi, ionic_strength, z0, z1):
    """xij function for unsymmetrical mixing."""
    return 6 * z0 * z1 * Aphi * np.sqrt(ionic_strength)


def etheta(Aphi, ionic_strength, z0, z1):
    """etheta function for unsymmetrical mixing."""
    x00 = xij(Aphi, ionic_strength, z0, z0)
    x01 = xij(Aphi, ionic_strength, z0, z1)
    x11 = xij(Aphi, ionic_strength, z1, z1)
    return (
        z0
        * z1
        * (library.func_J(x01) - 0.5 * (library.func_J(x00) + library.func_J(x11)))
        / (4 * ionic_strength)
    )


@jax.jit
def _Gibbs_nRT_v3(solutes, temperature, pressure):
    def add_ca(combo):
        c, a = combo
        return (
            m_cats[c]
            * m_anis[a]
            * (
                2 * B(sqrt_I, *ca[c][a][:3], *ca[c][a][5:7])
                + Z * CT(sqrt_I, *ca[c][a][3:5], ca[c][a][7])
            )
        )

    def add_cc(combo):
        c0, c1 = combo
        return (
            2
            * m_cats[c0]
            * m_cats[c1]
            * (
                cc[c0][c1]
                + etheta(
                    Aphi,
                    ionic_strength,
                    library.charges_cat[c0],
                    library.charges_cat[c1],
                )
            )
        )

    def add_cca(combo):
        c0, c1, a = combo
        return m_cats[c0] * m_cats[c1] * m_anis[a] * cca[c0][c1][a]

    def add_aa(combo):
        a0, a1 = combo
        return (
            2
            * m_anis[a0]
            * m_anis[a1]
            * (
                aa[a0][a1]
                + etheta(
                    Aphi,
                    ionic_strength,
                    library.charges_ani[a0],
                    library.charges_ani[a1],
                )
            )
        )

    def add_caa(combo):
        c, a0, a1 = combo
        return m_cats[c] * m_anis[a0] * m_anis[a1] * caa[c][a0][a1]

    def add_nnn(n):
        return m_neus[n] ** 3 * nnn[n]

    def add_nn(combo):
        n0, n1 = combo
        return 2 * m_neus[n0] * m_neus[n1] * nn[n0][n1]

    def add_nc(combo):
        n, c = combo
        return 2 * m_neus[n] * m_cats[c] * nc[n][c]

    def add_na(combo):
        n, a = combo
        return 2 * m_neus[n] * m_anis[a] * na[n][a]

    def add_nca(combo):
        n, c, a = combo
        return m_neus[n] * m_cats[c] * m_anis[a] * nca[n][c][a]

    # Split up cations, anions and neutrals
    m_cats = np.array([solutes[c] for c in library.cations])
    m_anis = np.array([solutes[a] for a in library.anions])
    m_neus = np.array([solutes[n] for n in library.neutrals])
    # Calculate ionic-strength-dependent terms
    ionic_strength = 0.5 * (
        np.sum(m_cats * library.charges_cat**2)
        + np.sum(m_anis * library.charges_ani**2)
    )
    Z = np.sum(m_cats * library.charges_cat) - np.sum(m_anis * library.charges_ani)
    sqrt_I = np.sqrt(ionic_strength)
    tp = (temperature, pressure)
    Aphi = library.Aphi(*tp)[0]
    gibbs = Gibbs_DH(Aphi, ionic_strength)
    # Add specific interactions
    if len(library.ca_combos) > 0:
        ca = library.get_ca_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_ca, library.ca_combos))
    if len(library.cc_combos) > 0:
        cc = library.get_cc_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_cc, library.cc_combos))
    if len(library.cca_combos) > 0:
        cca = library.get_cca_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_cca, library.cca_combos))
    if len(library.aa_combos) > 0:
        aa = library.get_aa_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_aa, library.aa_combos))
    if len(library.caa_combos) > 0:
        caa = library.get_caa_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_caa, library.caa_combos))
    if len(library.nnn_combos) > 0:
        nnn = library.get_nnn_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_nnn, library.nnn_combos))
    if len(library.nn_combos) > 0:
        nn = library.get_nn_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_nn, library.nn_combos))
    if len(library.nc_combos) > 0:
        nc = library.get_nc_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_nc, library.nc_combos))
    if len(library.na_combos) > 0:
        na = library.get_na_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_na, library.na_combos))
    if len(library.nca_combos) > 0:
        nca = library.get_nca_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_nca, library.nca_combos))
    return gibbs


def Gibbs_nRT(solutes, temperature, pressure):
    """Calculate Gex/nRT.

    Parameters
    ----------
    solutes : dict
        Dissolved species (keys; str) and their molalities in mol/kg (values; float).
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in dbar.

    Returns
    -------
    float
        Gex/nRT.
    """
    return _Gibbs_nRT_v3(solutes, temperature, pressure)


@jax.jit
def log_activity_coefficients(solutes, temperature, pressure):
    """Calculate the natural logs of the activity coefficients of all solutes.

    Parameters
    ----------
    solutes : dict
        Dissolved species (keys; str) and their molalities in mol/kg (values; float).
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in dbar.

    Returns
    -------
    dict
        Natural logs of the activity coefficients.
    """
    return jax.grad(_Gibbs_nRT_v3)(solutes, temperature, pressure)


@jax.jit
def activity_coefficients(solutes, temperature, pressure):
    """Calculate the activity coefficient of all solutes.

    Parameters
    ----------
    solutes : dict
        Dissolved species (keys; str) and their molalities in mol/kg (values; float).
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in dbar.

    Returns
    -------
    dict
        Activity coefficients.
    """
    log_acfs = log_activity_coefficients(solutes, temperature, pressure)
    return {k: np.exp(v) for k, v in log_acfs.items()}


@jax.jit
def osmotic_coefficient(solutes, temperature, pressure):
    """Calculate the osmotic coefficient of the solution.

    Parameters
    ----------
    solutes : dict
        Dissolved species (keys; str) and their molalities in mol/kg (values; float).
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in dbar.

    Returns
    -------
    float
        Osmotic coefficient.
    """
    return 1 - jax.grad(
        lambda ww: ww
        * _Gibbs_nRT_v3({s: m / ww for s, m in solutes.items()}, temperature, pressure)
    )(1.0) / np.sum(np.array([m for m in solutes.values()]))


@jax.jit
def log_activity_water(solutes, temperature, pressure):
    """Calculate the natural log of the water activity.

    Parameters
    ----------
    solutes : dict
        Dissolved species (keys; str) and their molalities in mol/kg (values; float).
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in dbar.

    Returns
    -------
    float
        Natural log of the water activity.
    """
    return (
        jax.grad(
            lambda ww: ww
            * _Gibbs_nRT_v3(
                {s: m / ww for s, m in solutes.items()}, temperature, pressure
            )
        )(1.0)
        - np.sum(np.array([m for m in solutes.values()]))
    ) * mass_water


@jax.jit
def activity_water(solutes, temperature, pressure):
    """Calculate the water activity.

    Parameters
    ----------
    solutes : dict
        Dissolved species (keys; str) and their molalities in mol/kg (values; float).
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in dbar.

    Returns
    -------
    float
        Water activity.
    """
    return np.exp(log_activity_water(solutes, temperature, pressure))
