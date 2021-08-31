# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
from collections import OrderedDict
from jax import numpy as np
from . import constants, properties


def osmotic_to_activity(molalities, osmotic_coefficient):
    """Convert osmotic coefficient to water activity."""
    return np.exp(-osmotic_coefficient * constants.Mw * np.sum(molalities))


def activity_to_osmotic(molalities, activity_water):
    """Convert water activity to osmotic coefficient."""
    return -np.log(activity_water) / (constants.Mw * np.sum(molalities))


def log_activities_to_mean(log_acf_M, log_acf_X, n_M, n_X):
    """Calculate the mean activity coefficient for an electrolyte."""
    return (n_M * log_acf_M + n_X * log_acf_X) / (n_M + n_X)


# Define dict of charges.
# Order: neutrals, cations, then anions, and alphabetical within each group.
solute_to_charge = {
    # Neutrals
    "BOH3": 0,
    "CaCO3": 0,
    "CaHPO4": 0,
    "CO2": 0,
    "H2S": 0,
    "H3PO4": 0,
    "H4SiO4": 0,
    "HF": 0,
    "glycerol": 0,
    "MgCO3": 0,
    "MgHPO4": 0,
    "NH3": 0,
    "SO2": 0,
    "SrCO3": 0,
    "sucrose": 0,
    "tris": 0,
    "urea": 0,
    # Cations
    "Ba": +2,
    "Ca": +2,
    "CaF": +1,
    "CaH2PO4": +1,
    "Cdjj": +2,
    "Cojj": +2,
    "Cs": +1,
    "Cujj": +2,
    "Eujjj": +3,
    "Fejj": +2,
    "Fejjj": +3,
    "H": +1,
    "K": +1,
    "La": +3,
    "Li": +1,
    "Mg": +2,
    "MgF": +1,
    "MgH2PO4": +1,
    "MgOH": +1,
    "Na": +1,
    "Ni": +2,
    "NH4": +1,
    "Rb": +1,
    "Sr": +2,
    "trisH": +1,
    "UO2": +2,
    "Znjj": +2,
    # Anions
    "acetate": -1,
    "AsO4": -2,
    "BOH4": -1,
    "Br": -1,
    "BrO3": -1,
    "CaPO4": -1,
    "Cl": -1,
    "ClO3": -1,
    "ClO4": -1,
    "CO3": -2,
    "F": -1,
    "H2AsO4": -1,
    "H2PO4": -1,
    "H3SiO4": -1,
    "HAsO4": -2,
    "HCO3": -1,
    "HPO4": -2,
    "HS": -1,
    "HSO3": -1,
    "HSO4": -1,
    "I": -1,
    "MgPO4": -1,
    "NO2": -1,
    "NO3": -1,
    "OH": -1,
    "PO4": -3,
    "S2O3": -2,
    "SCN": -1,
    "SO3": -2,
    "SO4": -2,
}


all_cations = set([s for s, c in solute_to_charge.items() if c > 0])
all_anions = set([s for s, c in solute_to_charge.items() if c < 0])
all_neutrals = set([s for s, c in solute_to_charge.items() if c == 0])


def solvent_to_solution(molalities, pks_constants=None):
    """Converts concentrations and equilibrium constants (pKs) from molality
    (mol/kg-solvent) to molinity (mol/kg-solution).
    molalities - ordered dict of the molalities of all seawater constituents
    (e.g. as returned by pytzer.solve).
    pks - ordered dict of pK values computed for the solution in molal
    (e.g. as returned by pytzer.solve).
    Returns molinities (mol/kg-solution), pKs in mol/kg-solution.
    """

    # Replace any NaNs with 0s
    for key in molalities.keys():
        if np.isnan(molalities[key]):
            molalities[key] = 0

    # Dict of weight concentrations (g/kg)
    gkg = OrderedDict(
        (key, mol * properties.ion_to_mass[key]) for key, mol in molalities.items()
    )

    # Get H2O fraction: 1 kg H2O / ((sum weights in 1 kg H2O) + 1 kg H2O)
    h2o = 1000 / (sum(gkg.values()) + 1000)

    # Convert molalities
    molinities = OrderedDict((key, mol * h2o) for key, mol in molalities.items())

    # Convert pKs
    eq_power = {
        "H2O": 2,
        "H2CO3": 1,
        "HCO3": 1,
        "HF": 1,
        "H2S": 1,
        "BOH3": 1,
        "HSO4": 1,
        "NH4": 1,
        "H3PO4": 1,
        "H2PO4": 1,
        "HPO4": 1,
        "MgOH": 1,
        "MgF": 1,
        "MgCO3": 1,
        "MgH2PO4": 1,
        "MgHPO4": 1,
        "MgPO4": 1,
        "CaF": 1,
        "CaCO3": 1,
        "CaH2PO4": 1,
        "CaHPO4": 1,
        "CaPO4": 1,
        "SrCO3": 1,
    }
    if pks_constants is not None:
        for key in pks_constants.keys():
            if np.isnan(pks_constants[key]):
                pks_constants[key] = 0
        ks = OrderedDict((key, 10 ** (-pk)) for key, pk in pks_constants.items())
        ks_out = OrderedDict((key, k * h2o ** eq_power[key]) for key, k in ks.items())
        pks_constants_out = OrderedDict(
            (key, -np.log10(k)) for key, k in ks_out.items()
        )
    else:
        pks_constants_out = None
    return molinities, pks_constants_out
