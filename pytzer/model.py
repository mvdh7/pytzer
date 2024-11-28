# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
"""Calculate solution properties using the Pitzer model."""
import itertools, jax
from collections import OrderedDict
from jax import numpy as np, lax
from .constants import b, Mw
from . import convert, libraries, properties, unsymmetrical

library_prev = libraries.Clegg23
library = libraries.lib_CWTD23.library


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


def etheta_prev(Aphi, I, z0, z1, func_J):
    """etheta function for unsymmetrical mixing."""
    x00 = xij(Aphi, I, z0, z0)
    x01 = xij(Aphi, I, z0, z1)
    x11 = xij(Aphi, I, z1, z1)
    return z0 * z1 * (func_J(x01) - 0.5 * (func_J(x00) + func_J(x11))) / (4 * I)


def etheta(Aphi, I, z0, z1):
    """etheta function for unsymmetrical mixing."""
    x00 = xij(Aphi, I, z0, z0)
    x01 = xij(Aphi, I, z0, z1)
    x11 = xij(Aphi, I, z1, z1)
    return (
        z0
        * z1
        * (library.func_J(x01) - 0.5 * (library.func_J(x00) + library.func_J(x11)))
        / (4 * I)
    )


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
    Aphi = library_prev["Aphi"](*tp)[0]
    Gibbs = Gibbs_DH(Aphi, I)
    # === Add (neutral-)cation-anion interactions ======================================
    if len(n_cations) > 0 and len(n_anions) > 0:
        Z = np.sum(m_cations * z_cations) - np.sum(m_anions * z_anions)
        for c, cation in enumerate(n_cations):
            for a, anion in enumerate(n_anions):
                try:
                    ca = library_prev["ca"][cation][anion](*tp)
                    Gibbs = Gibbs + m_cations[c] * m_anions[a] * (
                        2 * B(sqrt_I, *ca[:3], *ca[5:7])
                        + Z * CT(sqrt_I, *ca[3:5], ca[7])
                    )
                except KeyError:
                    pass
                if len(n_neutrals) > 0:
                    for n, neutral in enumerate(n_neutrals):
                        try:
                            Gibbs = (
                                Gibbs
                                + m_neutrals[n]
                                * m_cations[c]
                                * m_anions[a]
                                * library_prev["nca"][neutral][cation][anion](*tp)[0]
                            )
                        except KeyError:
                            pass
    # === Add cation-cation(-anion) interactions =======================================
    if len(n_cations) > 1:
        for c0, cation0 in enumerate(n_cations):
            for _c1, cation1 in enumerate(n_cations[(c0 + 1) :]):
                c1 = c0 + _c1 + 1
                try:
                    theta = library_prev["cc"][cation0][cation1](*tp)[0]
                except KeyError:
                    theta = 0.0
                Gibbs = Gibbs + 2 * m_cations[c0] * m_cations[c1] * (
                    theta
                    + etheta_prev(
                        Aphi,
                        I,
                        z_cations[c0],
                        z_cations[c1],
                        library_prev["func_J"],
                    )
                )
                for a, anion in enumerate(n_anions):
                    try:
                        Gibbs = (
                            Gibbs
                            + m_cations[c0]
                            * m_cations[c1]
                            * m_anions[a]
                            * library_prev["cca"][cation0][cation1][anion](*tp)[0]
                        )
                    except KeyError:
                        pass
    # === Add (cation-)anion-anion interactions ========================================
    if len(n_anions) > 1:
        for a0, anion0 in enumerate(n_anions):
            for _a1, anion1 in enumerate(n_anions[(a0 + 1) :]):
                a1 = a0 + _a1 + 1
                try:
                    theta = library_prev["aa"][anion0][anion1](*tp)[0]
                except KeyError:
                    theta = 0.0
                Gibbs = Gibbs + 2 * m_anions[a0] * m_anions[a1] * (
                    theta
                    + etheta_prev(
                        Aphi, I, z_anions[a0], z_anions[a1], library_prev["func_J"]
                    )
                )
                for c, cation in enumerate(n_cations):
                    try:
                        Gibbs = (
                            Gibbs
                            + m_cations[c]
                            * m_anions[a0]
                            * m_anions[a1]
                            * library_prev["caa"][cation][anion0][anion1](*tp)[0]
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
                        * library_prev["nn"][neutral0][neutral1](*tp)[0]
                    )
                except KeyError:
                    pass
            # Neutral-neutral-neutral (always the same neutral 3 times)
            try:
                Gibbs = (
                    Gibbs + m_neutrals[n0] ** 3 * library_prev["nnn"][neutral0](*tp)[0]
                )
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
                            * library_prev["nc"][neutral0][cation](*tp)[0]
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
                            * library_prev["na"][neutral0][anion](*tp)[0]
                        )
                    except KeyError:
                        pass
    return Gibbs


# This one aims to remove all the try/excepts by looping through the library_prev
# functions instead of all solutes.
# It did mean we had to add theta_zero functions in for everything that didn't have
# one assigned, because etheta still gets calculated if theta is zero for a c-c or a-a.
# Try to lax.map this to speed up compilation?  Some hybrid approach...
@jax.jit
def _Gibbs_nRT_wow(solutes, temperature, pressure):
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
    Aphi = library_prev["Aphi"](*tp)[0]
    Gibbs = Gibbs_DH(Aphi, I)
    charges = {s: convert.solute_to_charge[s] for s in solutes}
    # === Add (neutral-)cation-anion interactions ======================================
    if len(n_cations) > 0 and len(n_anions) > 0:
        Z = np.sum(m_cations * z_cations) - np.sum(m_anions * z_anions)
        for cation in library_prev["ca"]:
            for anion in library_prev["ca"][cation]:
                ca = library_prev["ca"][cation][anion](*tp)
                Gibbs = Gibbs + solutes[cation] * solutes[anion] * (
                    2 * B(sqrt_I, *ca[:3], *ca[5:7]) + Z * CT(sqrt_I, *ca[3:5], ca[7])
                )
        for neutral in library_prev["nca"]:
            for cation in library_prev["nca"][neutral]:
                for anion in library_prev["nca"][neutral][cation]:
                    Gibbs = (
                        Gibbs
                        + solutes[neutral]
                        * solutes[cation]
                        * solutes[anion]
                        * library_prev["nca"][neutral][cation][anion](*tp)[0]
                    )
    # === Add cation-cation(-anion) interactions =======================================
    for cation0 in library_prev["cc"]:
        for cation1 in library_prev["cc"][cation0]:
            theta = library_prev["cc"][cation0][cation1](*tp)[0]
            Gibbs = Gibbs + solutes[cation0] * solutes[cation1] * (
                theta
                + etheta_prev(
                    Aphi,
                    I,
                    charges[cation0],
                    charges[cation1],
                    library_prev["func_J"],
                )
            )
    for cation0 in library_prev["cca"]:
        for cation1 in library_prev["cca"][cation0]:
            for anion in library_prev["cca"][cation0][cation1]:
                Gibbs = (
                    Gibbs
                    + 0.5
                    * solutes[cation0]
                    * solutes[cation1]
                    * solutes[anion]
                    * library_prev["cca"][cation0][cation1][anion](*tp)[0]
                )
    # # === Add (cation-)anion-anion interactions ========================================
    for anion0 in library_prev["aa"]:
        for anion1 in library_prev["aa"][anion0]:
            theta = library_prev["aa"][anion0][anion1](*tp)[0]
            Gibbs = Gibbs + solutes[anion0] * solutes[anion1] * (
                theta
                + etheta_prev(
                    Aphi,
                    I,
                    charges[anion0],
                    charges[anion1],
                    library_prev["func_J"],
                )
            )
    for cation in library_prev["caa"]:
        for anion0 in library_prev["caa"][cation]:
            for anion1 in library_prev["caa"][cation][anion0]:
                Gibbs = (
                    Gibbs
                    + 0.5
                    * solutes[cation]
                    * solutes[anion0]
                    * solutes[anion1]
                    * library_prev["caa"][cation][anion0][anion1](*tp)[0]
                )
    # # === Add other neutral interactions ===============================================
    # Neutral-neutral (can be the same or different neutrals)
    for neutral0 in library_prev["nn"]:
        for neutral1 in library_prev["nn"][neutral0]:
            Gibbs = (
                Gibbs
                + solutes[neutral0]
                * solutes[neutral1]
                * library_prev["nn"][neutral0][neutral1](*tp)[0]
            )
    # Neutral-neutral-neutral (always the same neutral 3 times---for now)
    for neutral in library_prev["nnn"]:
        Gibbs = Gibbs + m_neutrals[neutral] ** 3 * library_prev["nnn"][neutral](*tp)[0]
    # Neutral-cation
    for neutral in library_prev["nc"]:
        for cation in library_prev["nc"][neutral]:
            Gibbs = (
                Gibbs
                + 2
                * solutes[neutral]
                * solutes[cation]
                * library_prev["nc"][neutral][cation](*tp)[0]
            )
    # Neutral-anion
    for neutral in library_prev["na"]:
        for anion in library_prev["na"][neutral]:
            Gibbs = (
                Gibbs
                + 2
                * solutes[neutral]
                * solutes[anion]
                * library_prev["na"][neutral][anion](*tp)[0]
            )
    return Gibbs


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
                + etheta(Aphi, I, library.charges_cat[c0], library.charges_cat[c1])
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
                + etheta(Aphi, I, library.charges_ani[a0], library.charges_ani[a1])
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
    I = 0.5 * (
        np.sum(m_cats * library.charges_cat**2)
        + np.sum(m_anis * library.charges_ani**2)
    )
    Z = np.sum(m_cats * library.charges_cat) - np.sum(m_anis * library.charges_ani)
    sqrt_I = np.sqrt(I)
    tp = (temperature, pressure)
    Aphi = library.Aphi(*tp)[0]
    gibbs = Gibbs_DH(Aphi, I)
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
