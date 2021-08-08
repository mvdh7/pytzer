# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Solve for thermodynamic equilibrium."""

import copy
from collections import OrderedDict
from scipy import optimize
import jax
from jax import numpy as np
from .. import dissociation, model
from . import components, stoichiometric


solutes_required = {
    "BOH3": ["BOH3", "BOH4", "H"],
    "MgOH": ["Mg", "OH", "MgOH"],
    "H2O": ["H", "OH"],
    "H2CO3": ["H", "HCO3", "CO2"],
    "HCO3": ["H", "CO3", "HCO3"],
    "HF": ["H", "F", "HF"],
    "HSO4": ["H", "HSO4", "SO4"],
    "trisH": ["tris", "H", "trisH"],
    "H3PO4": ["H3PO4", "H2PO4", "H"],
    "H2PO4": ["H2PO4", "HPO4", "H"],
    "HPO4": ["HPO4", "PO4", "H"],
    "H2S": ["H2S", "HS", "H"],
    "NH4": ["NH3", "NH4", "H"],
    "CaCO3": ["Ca", "CO3", "CaCO3"],
    "MgCO3": ["Mg", "CO3", "MgCO3"],
    "SrCO3": ["Sr", "CO3", "SrCO3"],
    "MgH2PO4": ["MgH2PO4", "H2PO4", "Mg"],
    "MgHPO4": ["MgHPO4", "HPO4", "Mg"],
    "MgPO4": ["MgPO4", "PO4", "Mg"],
    "CaH2PO4": ["CaH2PO4", "H2PO4", "Ca"],
    "CaHPO4": ["CaHPO4", "HPO4", "Ca"],
    "CaPO4": ["CaPO4", "PO4", "Ca"],
    "MgF": ["Mg", "F", "MgF"],
    "CaF": ["Ca", "F", "CaF"],
}


def Gibbs_H2O(log_kt_H2O, log_ks_H2O, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for water dissocation."""
    return log_acfs["H"] + log_acfs["OH"] - log_aH2O + log_ks_H2O - log_kt_H2O


def Gibbs_HSO4(log_kt_HSO4, log_ks_HSO4, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for the bisulfate-sulfate equilibrium."""
    return (
        log_acfs["H"] + log_acfs["SO4"] - log_acfs["HSO4"] + log_ks_HSO4 - log_kt_HSO4
    )


def Gibbs_HF(log_kt_HF, log_ks_HF, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy of hydrogen fluoride dissociation."""
    return log_acfs["H"] + log_acfs["F"] - log_acfs["HF"] + log_ks_HF - log_kt_HF


def Gibbs_MgOH(log_kt_MgOH, log_ks_MgOH, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for the magnesium-MgOH+ equilibrium."""
    return (
        log_acfs["Mg"] + log_acfs["OH"] - log_acfs["MgOH"] - log_ks_MgOH + log_kt_MgOH
    )


def Gibbs_trisH(log_kt_trisH, log_ks_trisH, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for the tris-trisH+ equilibrium."""
    return (
        log_acfs["tris"]
        + log_acfs["H"]
        - log_acfs["trisH"]
        + log_ks_trisH
        - log_kt_trisH
    )


def Gibbs_H2CO3(log_kt_H2CO3, log_ks_H2CO3, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for the H2CO3-bicarbonate equilibrium."""
    return (
        log_acfs["H"]
        + log_acfs["HCO3"]
        - log_acfs["CO2"]
        - log_aH2O
        + log_ks_H2CO3
        - log_kt_H2CO3
    )


def Gibbs_HCO3(log_kt_HCO3, log_ks_HCO3, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for bicarbonate dissociation."""
    return (
        log_acfs["H"] + log_acfs["CO3"] - log_acfs["HCO3"] + log_ks_HCO3 - log_kt_HCO3
    )


def Gibbs_BOH3(log_kt_BOH3, log_ks_BOH3, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for the boric acid equilibrium."""
    return (
        log_acfs["BOH4"]
        + log_acfs["H"]
        - log_acfs["BOH3"]
        - log_aH2O
        + log_ks_BOH3
        - log_kt_BOH3
    )


def Gibbs_CaCO3(log_kt_CaCO3, log_ks_CaCO3, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for CaCO3 formation."""
    return (
        log_acfs["CaCO3"]
        - log_acfs["Ca"]
        - log_acfs["CO3"]
        + log_ks_CaCO3
        - log_kt_CaCO3
    )


def Gibbs_MgCO3(log_kt_MgCO3, log_ks_MgCO3, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for MgCO3 formation."""
    return (
        log_acfs["MgCO3"]
        - log_acfs["Mg"]
        - log_acfs["CO3"]
        + log_ks_MgCO3
        - log_kt_MgCO3
    )


def Gibbs_SrCO3(log_kt_SrCO3, log_ks_SrCO3, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for SrCO3 formation."""
    return (
        log_acfs["SrCO3"]
        - log_acfs["Sr"]
        - log_acfs["CO3"]
        + log_ks_SrCO3
        - log_kt_SrCO3
    )


def Gibbs_H3PO4(log_kt_H3PO4, log_ks_H3PO4, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for H3PO4 dissociation."""
    return (
        log_acfs["H"]
        + log_acfs["H2PO4"]
        - log_acfs["H3PO4"]
        + log_ks_H3PO4
        - log_kt_H3PO4
    )


def Gibbs_H2PO4(log_kt_H2PO4, log_ks_H2PO4, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for H2PO4 dissociation."""
    return (
        log_acfs["H"]
        + log_acfs["HPO4"]
        - log_acfs["H2PO4"]
        + log_ks_H2PO4
        - log_kt_H2PO4
    )


def Gibbs_HPO4(log_kt_HPO4, log_ks_HPO4, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for HPO4 dissociation."""
    return (
        log_acfs["H"] + log_acfs["PO4"] - log_acfs["HPO4"] + log_ks_HPO4 - log_kt_HPO4
    )


def Gibbs_H2S(log_kt_H2S, log_ks_H2S, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for H2S dissociation."""
    return log_acfs["H"] + log_acfs["HS"] - log_acfs["H2S"] + log_ks_H2S - log_kt_H2S


def Gibbs_CaF(log_kt_CaF, log_ks_CaF, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for CaF formation."""
    return log_acfs["CaF"] - log_acfs["Ca"] - log_acfs["F"] + log_ks_CaF - log_kt_CaF


def Gibbs_MgF(log_kt_MgF, log_ks_MgF, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for MgF formation."""
    return log_acfs["MgF"] - log_acfs["Mg"] - log_acfs["F"] + log_ks_MgF - log_kt_MgF


def Gibbs_CaH2PO4(log_kt_CaH2PO4, log_ks_CaH2PO4, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for CaH2PO4 formation."""
    return (
        log_acfs["CaH2PO4"]
        - log_acfs["Ca"]
        - log_acfs["H2PO4"]
        + log_ks_CaH2PO4
        - log_kt_CaH2PO4
    )


def Gibbs_MgH2PO4(log_kt_MgH2PO4, log_ks_MgH2PO4, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for MgH2PO4 formation."""
    return (
        log_acfs["MgH2PO4"]
        - log_acfs["Mg"]
        - log_acfs["H2PO4"]
        + log_ks_MgH2PO4
        - log_kt_MgH2PO4
    )


def Gibbs_CaHPO4(log_kt_CaHPO4, log_ks_CaHPO4, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for CaHPO4 formation."""
    return (
        log_acfs["CaHPO4"]
        - log_acfs["Ca"]
        - log_acfs["HPO4"]
        + log_ks_CaHPO4
        - log_kt_CaHPO4
    )


def Gibbs_MgHPO4(log_kt_MgHPO4, log_ks_MgHPO4, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for MgHPO4 formation."""
    return (
        log_acfs["MgHPO4"]
        - log_acfs["Mg"]
        - log_acfs["HPO4"]
        + log_ks_MgHPO4
        - log_kt_MgHPO4
    )


def Gibbs_CaPO4(log_kt_CaPO4, log_ks_CaPO4, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for CaPO4 formation."""
    return (
        log_acfs["CaPO4"]
        - log_acfs["Ca"]
        - log_acfs["PO4"]
        + log_ks_CaPO4
        - log_kt_CaPO4
    )


def Gibbs_MgPO4(log_kt_MgPO4, log_ks_MgPO4, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for MgPO4 formation."""
    return (
        log_acfs["MgPO4"]
        - log_acfs["Mg"]
        - log_acfs["PO4"]
        + log_ks_MgPO4
        - log_kt_MgPO4
    )


def Gibbs_NH4(log_kt_NH4, log_ks_NH4, log_acfs, log_aH2O):
    """Evaluate the Gibbs energy for NH4 dissociation."""
    return log_acfs["H"] + log_acfs["NH4"] - log_acfs["NH4"] + log_ks_NH4 - log_kt_NH4


all_reactions = {
    "BOH3": Gibbs_BOH3,
    "H2CO3": Gibbs_H2CO3,
    "H2O": Gibbs_H2O,
    "HCO3": Gibbs_HCO3,
    "HF": Gibbs_HF,
    "HSO4": Gibbs_HSO4,
    "MgOH": Gibbs_MgOH,
    "trisH": Gibbs_trisH,
    "CaCO3": Gibbs_CaCO3,
    "MgCO3": Gibbs_MgCO3,
    "SrCO3": Gibbs_SrCO3,
    "H3PO4": Gibbs_H3PO4,
    "H2PO4": Gibbs_H2PO4,
    "HPO4": Gibbs_HPO4,
    "H2S": Gibbs_H2S,
    "MgF": Gibbs_MgF,
    "CaF": Gibbs_CaF,
    "MgH2PO4": Gibbs_MgH2PO4,
    "MgHPO4": Gibbs_MgHPO4,
    "MgPO4": Gibbs_MgPO4,
    "CaH2PO4": Gibbs_CaH2PO4,
    "CaHPO4": Gibbs_CaHPO4,
    "CaPO4": Gibbs_CaPO4,
    "NH4": Gibbs_NH4,
}


@jax.jit
def get_Gibbs_equilibria(
    pks_constants_to_solve,
    ptargets,
    totals,
    ks_constants,
    params,
    log_kt_constants,
):
    for i, rxn in enumerate(log_kt_constants.keys()):
        ks_constants[rxn] = 10.0 ** -pks_constants_to_solve[i]
    # Solve for pH
    ptargets = stoichiometric.solve(totals, ks_constants, ptargets=ptargets)
    solutes = components.get_solutes(totals, ks_constants, ptargets)
    log_aw = model.log_activity_water(solutes, **params)
    log_acfs = model.log_activity_coefficients(solutes, **params)
    # Get equilibria
    Gibbs_equilibria = np.array([])
    for rxn in log_kt_constants.keys():
        Gibbs_equilibria = np.append(
            Gibbs_equilibria,
            all_reactions[rxn](
                log_kt_constants[rxn], np.log(ks_constants[rxn]), log_acfs, log_aw
            ),
        )
    return Gibbs_equilibria


jac_Gibbs_equilibria = jax.jit(jax.jacfwd(get_Gibbs_equilibria))


def update_ks_constants(all_ks_constants, optresult_solve):
    ks_constants = copy.deepcopy(all_ks_constants)
    for i, rxn in enumerate(optresult_solve["equilibria"]):
        ks_constants[rxn] = 10.0 ** -optresult_solve["x"][i]
    return ks_constants


def solve(totals, ks_constants, params, log_kt_constants, ptargets=None):
    """Solve the reactions in log_kt_constants for thermodynamic equilibrium."""
    if ptargets is None:
        ptargets = stoichiometric.create_ptargets(totals, ks_constants)
    pks_constants_to_solve = np.array(
        [-np.log10(np.exp(log_kt)) for log_kt in log_kt_constants.values()]
    )
    optresult = optimize.root(
        get_Gibbs_equilibria,
        pks_constants_to_solve,
        args=(
            ptargets,
            totals,
            ks_constants,
            params,
            log_kt_constants,
        ),
        method="hybr",
        jac=jac_Gibbs_equilibria,
    )
    optresult["equilibria"] = log_kt_constants
    return optresult
