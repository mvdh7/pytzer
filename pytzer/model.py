# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
"""Calculate solution properties using the Pitzer model."""
import itertools, jax
from collections import OrderedDict
from jax import numpy as np, lax
from .constants import b, Mw
from . import convert, libraries, properties, unsymmetrical

library = libraries.Clegg23


def Gibbs_DH(Aphi, I):
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
    return -4 * Aphi * I * np.log(1 + b * np.sqrt(I)) / b


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


def xij(Aphi, I, z0, z1):
    """xij function for unsymmetrical mixing."""
    return 6 * z0 * z1 * Aphi * np.sqrt(I)


def etheta(Aphi, I, z0, z1, func_J):
    """etheta function for unsymmetrical mixing."""
    x00 = xij(Aphi, I, z0, z0)
    x01 = xij(Aphi, I, z0, z1)
    x11 = xij(Aphi, I, z1, z1)
    return z0 * z1 * (func_J(x01) - 0.5 * (func_J(x00) + func_J(x11))) / (4 * I)


@jax.jit
def _Gibbs_nRT(solutes, temperature, pressure):
    # === Separate the names, molalities and charges of cations, anions and neutrals ===
    n_solutes = list(solutes.keys())
    n_cations = [s for s in n_solutes if s in convert.all_cations]
    n_anions = [s for s in n_solutes if s in convert.all_anions]
    n_neutrals = [s for s in n_solutes if s in convert.all_neutrals]
    m_cations = []
    z_cations = []
    for s in n_cations:
        m_cations.append(solutes[s])
        z_cations.append(convert.solute_to_charge[s])
    m_anions = []
    z_anions = []
    for s in n_anions:
        m_anions.append(solutes[s])
        z_anions.append(convert.solute_to_charge[s])
    m_neutrals = []
    for s in n_neutrals:
        m_neutrals.append(solutes[s])
    m_cations = np.array(m_cations)
    z_cations = np.array(z_cations)
    m_anions = np.array(m_anions)
    z_anions = np.array(z_anions)
    # === Start with terms that depend only on ionic strength ==========================
    if len(n_cations) > 0 or len(n_anions) > 0:
        I = 0.5 * (np.sum(m_cations * z_cations**2) + np.sum(m_anions * z_anions**2))
    else:
        I = 0.0
    sqrt_I = np.sqrt(I)
    tp = (temperature, pressure)
    Aphi = library["Aphi"](*tp)[0]
    Gibbs = Gibbs_DH(Aphi, I)
    # === Add (neutral-)cation-anion interactions ======================================
    if len(n_cations) > 0 and len(n_anions) > 0:
        Z = np.sum(m_cations * z_cations) - np.sum(m_anions * z_anions)
        for c, cation in enumerate(n_cations):
            for a, anion in enumerate(n_anions):
                ca = library["ca"][cation][anion](*tp)
                Gibbs = Gibbs + m_cations[c] * m_anions[a] * (
                    2 * B(sqrt_I, *ca[:3], *ca[5:7]) + Z * CT(sqrt_I, *ca[3:5], ca[7])
                )
                if len(n_neutrals) > 0:
                    for n, neutral in enumerate(n_neutrals):
                        try:
                            Gibbs = (
                                Gibbs
                                + m_neutrals[n]
                                * m_cations[c]
                                * m_anions[a]
                                * library["nca"][neutral][cation][anion](*tp)[0]
                            )
                        except KeyError:
                            pass
    # === Add cation-cation(-anion) interactions =======================================
    if len(n_cations) > 1:
        for c0, cation0 in enumerate(n_cations):
            for _c1, cation1 in enumerate(n_cations[(c0 + 1) :]):
                c1 = c0 + _c1 + 1
                try:
                    Gibbs = Gibbs + 2 * m_cations[c0] * m_cations[c1] * (
                        library["cc"][cation0][cation1](*tp)[0]
                        + etheta(
                            Aphi,
                            I,
                            z_cations[c0],
                            z_cations[c1],
                            library["func_J"],
                        )
                    )
                except KeyError:
                    pass
                for a, anion in enumerate(n_anions):
                    try:
                        Gibbs = (
                            Gibbs
                            + m_cations[c0]
                            * m_cations[c1]
                            * m_anions[a]
                            * library["cca"][cation0][cation1][anion](*tp)[0]
                        )
                    except KeyError:
                        pass
    # === Add (cation-)anion-anion interactions ========================================
    if len(n_anions) > 1:
        for a0, anion0 in enumerate(n_anions):
            for _a1, anion1 in enumerate(n_anions[(a0 + 1) :]):
                a1 = a0 + _a1 + 1
                try:
                    Gibbs = Gibbs + 2 * m_anions[a0] * m_anions[a1] * (
                        library["aa"][anion0][anion1](*tp)[0]
                        + etheta(Aphi, I, z_anions[a0], z_anions[a1], library["func_J"])
                    )
                except KeyError:
                    pass
                for c, cation in enumerate(n_cations):
                    try:
                        Gibbs = (
                            Gibbs
                            + m_cations[c]
                            * m_anions[a0]
                            * m_anions[a1]
                            * library["caa"][cation][anion0][anion1](*tp)[0]
                        )
                    except KeyError:
                        pass
    # === Add other neutral interactions ===============================================
    if len(n_neutrals) > 0:
        for n0, neutral0 in enumerate(n_neutrals):
            # Neutral-neutral (can be the same or different neutrals)
            for n1, neutral1 in enumerate(n_neutrals):
                try:
                    Gibbs = (
                        Gibbs
                        + m_neutrals[n0]
                        * m_neutrals[n1]
                        * library["nn"][neutral0][neutral1](*tp)[0]
                    )
                except KeyError:
                    pass
            # Neutral-neutral-neutral (always the same neutral 3 times)
            try:
                Gibbs = Gibbs + m_neutrals[n0] ** 3 * library["nnn"][neutral0](*tp)[0]
            except KeyError:
                pass
            # Neutral-cation
            if len(n_cations) > 0:
                for c, cation in enumerate(n_cations):
                    try:
                        Gibbs = (
                            Gibbs
                            + 2
                            * m_neutrals[n0]
                            * m_cations[c]
                            * library["nc"][neutral0][cation](*tp)[0]
                        )
                    except KeyError:
                        pass
            # Neutral-anion
            if len(n_anions) > 0:
                for a, anion in enumerate(n_anions):
                    try:
                        Gibbs = (
                            Gibbs
                            + 2
                            * m_neutrals[n0]
                            * m_anions[a]
                            * library["na"][neutral0][anion](*tp)[0]
                        )
                    except KeyError:
                        pass
    return Gibbs


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
    return _Gibbs_nRT(solutes, temperature, pressure)


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
    return jax.grad(_Gibbs_nRT)(solutes, temperature, pressure)


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
        * _Gibbs_nRT({s: m / ww for s, m in solutes.items()}, temperature, pressure)
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
            * _Gibbs_nRT({s: m / ww for s, m in solutes.items()}, temperature, pressure)
        )(1.0)
        - np.sum(np.array([m for m in solutes.values()]))
    ) * Mw


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
