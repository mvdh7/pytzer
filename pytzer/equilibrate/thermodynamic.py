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
    """Evaluate the Gibbs energy for the bicarbonate-carbonate equilibrium."""
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


all_reactions = {
    "BOH3": Gibbs_BOH3,
    "H2CO3": Gibbs_H2CO3,
    "H2O": Gibbs_H2O,
    "HCO3": Gibbs_HCO3,
    "HF": Gibbs_HF,
    "HSO4": Gibbs_HSO4,
    "MgOH": Gibbs_MgOH,
    "trisH": Gibbs_trisH,
}


@jax.jit
def get_Gibbs_equilibria(
    pks_constants_to_solve,
    pfixed,
    totals,
    ks_constants,
    params,
    equilibria_to_solve,  # this is log_kt_constants
):
    for i, rxn in enumerate(equilibria_to_solve.keys()):
        ks_constants[rxn] = 10.0 ** -pks_constants_to_solve[i]
    # Solve for pH
    pfixed = stoichiometric.solve(totals, ks_constants, pfixed=pfixed)
    solutes = components.get_solutes(totals, ks_constants, pfixed)
    log_aw = model.log_activity_water(solutes, **params)
    log_acfs = model.log_activity_coefficients(solutes, **params)
    # Get equilibria
    Gibbs_equilibria = np.array([])
    for rxn in equilibria_to_solve.keys():
        Gibbs_equilibria = np.append(
            Gibbs_equilibria,
            all_reactions[rxn](
                equilibria_to_solve[rxn], np.log(ks_constants[rxn]), log_acfs, log_aw
            ),
        )
    return Gibbs_equilibria


jac_Gibbs_equilibria = jax.jit(jax.jacfwd(get_Gibbs_equilibria))


def update_ks_constants(all_ks_constants, optresult_solve):
    ks_constants = copy.deepcopy(all_ks_constants)
    for i, rxn in enumerate(optresult_solve["equilibria"]):
        ks_constants[rxn] = 10.0 ** -optresult_solve["x"][i]
    return ks_constants


def solve(equilibria_to_solve, totals, ks_constants, params, pfixed=None):
    """Solve for thermodynamic equilibrium."""
    if pfixed is None:
        pfixed = stoichiometric.create_pfixed(totals=totals)
    total_targets = stoichiometric.get_total_targets(totals, pfixed)
    pks_constants_to_solve = np.array(
        [-np.log10(np.exp(log_kt)) for log_kt in equilibria_to_solve.values()]
    )
    optresult = optimize.root(
        get_Gibbs_equilibria,
        pks_constants_to_solve,
        args=(
            pfixed,
            totals,
            ks_constants,
            params,
            equilibria_to_solve,
        ),
        method="hybr",
        jac=jac_Gibbs_equilibria,
    )
    optresult["equilibria"] = equilibria_to_solve
    return optresult
