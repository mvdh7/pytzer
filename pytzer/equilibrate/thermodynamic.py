# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
"""Solve for thermodynamic equilibrium."""
import copy
from collections import OrderedDict
from scipy import optimize
import jax
from jax import numpy as np
from .. import dissociation, model
from . import components, stoichiometric


def Gibbs_H2O(log_kt_H2O, solutes, log_activity_coefficients, log_activity_water):
    """Evaluate the Gibbs energy for water dissocation."""
    m, log_acf = solutes, log_activity_coefficients
    return (
        log_acf["H"]
        + np.log(m["H"])
        + log_acf["OH"]
        + np.log(m["OH"])
        - log_activity_water
        - log_kt_H2O
    )


def Gibbs_HSO4(log_kt_HSO4, solutes, log_activity_coefficients, *args):
    """Evaluate the Gibbs energy for the bisulfate-sulfate equilibrium."""
    m, log_acf = solutes, log_activity_coefficients
    return (
        log_acf["H"]
        + np.log(m["H"])
        + log_acf["SO4"]
        + np.log(m["SO4"])
        - log_acf["HSO4"]
        - np.log(m["HSO4"])
        - log_kt_HSO4
    )


def Gibbs_HF(log_kt_HF, solutes, log_activity_coefficients, *args):
    """Evaluate the Gibbs energy of the hydrogen fluoride equilibrium."""
    m, log_acf = solutes, log_activity_coefficients
    return (
        log_acf["H"]
        + np.log(m["H"])
        + log_acf["F"]
        + np.log(m["F"])
        - log_acf["HF"]
        - np.log(m["HF"])
        - log_kt_HF
    )


def Gibbs_MgOH(log_kt_MgOH, solutes, log_activity_coefficients, *args):
    """Evaluate the Gibbs energy for the magnesium-MgOH+ equilibrium."""
    m, log_acf = solutes, log_activity_coefficients
    return (
        log_acf["Mg"]
        + np.log(m["Mg"])
        + log_acf["OH"]
        + np.log(m["OH"])
        - log_acf["MgOH"]
        - np.log(m["MgOH"])
        + log_kt_MgOH
    )


def Gibbs_trisH(log_kt_trisH, solutes, log_activity_coefficients, *args):
    """Evaluate the Gibbs energy for the tris-trisH+ equilibrium."""
    m, log_acf = solutes, log_activity_coefficients
    return (
        log_acf["tris"]
        + np.log(m["tris"])
        - log_acf["trisH"]
        - np.log(m["trisH"])
        + log_acf["H"]
        + np.log(m["H"])
        - log_kt_trisH
    )


def Gibbs_H2CO3(log_kt_H2CO3, solutes, log_activity_coefficients, log_activity_water):
    """Evaluate the Gibbs energy for the H2CO3-bicarbonate equilibrium."""
    m, log_acf = solutes, log_activity_coefficients
    return (
        log_acf["H"]
        + np.log(m["H"])
        + log_acf["HCO3"]
        + np.log(m["HCO3"])
        - log_acf["CO2"]
        - np.log(m["CO2"])
        - log_activity_water
        - log_kt_H2CO3
    )


def Gibbs_HCO3(log_kt_HCO3, solutes, log_activity_coefficients, *args):
    """Evaluate the Gibbs energy for the bicarbonate-carbonate equilibrium."""
    m, log_acf = solutes, log_activity_coefficients
    return (
        log_acf["H"]
        + np.log(m["H"])
        + log_acf["CO3"]
        + np.log(m["CO3"])
        - log_acf["HCO3"]
        - np.log(m["HCO3"])
        - log_kt_HCO3
    )


def Gibbs_BOH3(log_kt_BOH3, solutes, log_activity_coefficients, log_activity_water):
    """Evaluate the Gibbs energy for the boric acid equilibrium."""
    m, log_acf = solutes, log_activity_coefficients
    return (
        log_acf["BOH4"]
        + np.log(m["BOH4"])
        + log_acf["H"]
        + np.log(m["H"])
        - log_acf["BOH3"]
        - np.log(m["BOH3"])
        - log_activity_water
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
    pm_initial,
    totals,
    all_ks_constants,
    total_targets,
    params,
    log_kt_constants,
):
    for i, rxn in enumerate(log_kt_constants.keys()):
        all_ks_constants[rxn] = 10.0 ** -pks_constants_to_solve[i]
    # Solve for pH
    pm_initial = stoichiometric.solve(
        pm_initial, totals, all_ks_constants, total_targets
    )
    molalities = 10.0 ** -pm_initial
    solutes = components.get_all(*molalities, totals, all_ks_constants)
    log_aw = model.log_activity_water(solutes, **params)
    log_acfs = model.log_activity_coefficients(solutes, **params)
    # Get equilibria
    Gibbs_equilibria = np.array([])
    for rxn in log_kt_constants.keys():
        Gibbs_equilibria = np.append(
            Gibbs_equilibria,
            all_reactions[rxn](log_kt_constants[rxn], solutes, log_acfs, log_aw),
        )
    return Gibbs_equilibria


jac_Gibbs_equilibria = jax.jit(jax.jacfwd(get_Gibbs_equilibria))


@jax.jit
def get_Gibbs_equilibria_v2(
    pks_constants_to_solve,
    pfixed_initial,
    totals,
    all_ks_constants,
    params,
    log_kt_constants,
):
    for i, rxn in enumerate(log_kt_constants.keys()):
        all_ks_constants[rxn] = 10.0 ** -pks_constants_to_solve[i]
    # Solve for pH
    pfixed_initial = stoichiometric.solve_v2(pfixed_initial, totals, all_ks_constants)
    fixed_initial = OrderedDict((k, 10.0 ** -v) for k, v in pfixed_initial.items())
    solutes = components.get_all_v2(fixed_initial, totals, all_ks_constants)
    log_aw = model.log_activity_water(solutes, **params)
    log_acfs = model.log_activity_coefficients(solutes, **params)
    # Get equilibria
    Gibbs_equilibria = np.array([])
    for rxn in log_kt_constants.keys():
        Gibbs_equilibria = np.append(
            Gibbs_equilibria,
            all_reactions[rxn](log_kt_constants[rxn], solutes, log_acfs, log_aw),
        )
    return Gibbs_equilibria


jac_Gibbs_equilibria_v2 = jax.jit(jax.jacfwd(get_Gibbs_equilibria_v2))


def update_ks_constants(all_ks_constants, optresult_solve):
    ks_constants = copy.deepcopy(all_ks_constants)
    for i, rxn in enumerate(optresult_solve["equilibria"]):
        ks_constants[rxn] = 10.0 ** -optresult_solve["x"][i]
    return ks_constants


def solve(
    equilibria_to_solve, pm_initial, totals, all_ks_constants, params, which_pms,
):
    total_targets = stoichiometric.get_total_targets(totals, which_pms)
    log_kt_constants = OrderedDict(
        (eq, dissociation.all_log_ks[eq](T=params["temperature"]))
        for eq in equilibria_to_solve
    )
    pks_constants_to_solve = np.array(
        [-np.log10(np.exp(log_kt)) for log_kt in log_kt_constants.values()]
    )
    optresult = optimize.root(
        get_Gibbs_equilibria,
        pks_constants_to_solve,
        args=(
            pm_initial,
            totals,
            all_ks_constants,
            total_targets,
            params,
            log_kt_constants,
        ),
        method="hybr",
        jac=jac_Gibbs_equilibria,
    )
    optresult["equilibria"] = equilibria_to_solve
    return optresult


def solve_v2(equilibria_to_solve, pfixed_initial, totals, all_ks_constants, params):
    total_targets = stoichiometric.get_total_targets(totals, pfixed_initial)
    log_kt_constants = OrderedDict(
        (eq, dissociation.all_log_ks[eq](T=params["temperature"]))
        for eq in equilibria_to_solve
    )
    pks_constants_to_solve = np.array(
        [-np.log10(np.exp(log_kt)) for log_kt in log_kt_constants.values()]
    )
    optresult = optimize.root(
        get_Gibbs_equilibria_v2,
        pks_constants_to_solve,
        args=(pfixed_initial, totals, all_ks_constants, params, log_kt_constants,),
        method="hybr",
        jac=jac_Gibbs_equilibria_v2,
    )
    optresult["equilibria"] = equilibria_to_solve
    return optresult
