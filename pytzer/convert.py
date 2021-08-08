# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
from collections import OrderedDict
from jax import numpy as np
from . import constants


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


def solvent_to_solution(molalities, ks):
    """Calculates the fraction of H2O in a seawater solution of known composition and molalities. Converts concentrations and equilibrium constants (Ks) from molal to mol/kg-solution.
    molalities - ordered dict of the molalities of all seawater constituents (e.g. as returned by pytzer.solve).
    ks - ordered dict of K values computed for the solution in molal (e.g. as returned by pytzer.solve).
    Returns concentrations in mol/kg-solution, Ks in mol/kg-solution."""

    # Molacular weights (g/mol) of various ions, taken from PubChem (https://pubchem.ncbi.nlm.nih.gov/)
    MW = OrderedDict(
        {
            "Na": 22.9897693,  # Na(+)
            "Mg": 24.305,  # Mg(2+)
            "Ca": 40.08,  # Ca(2+)
            "K": 39.098,  # K(+)
            "Sr": 87.6,  # Sr(2+)
            "Cl": 35.45,  # Cl(-)
            "SO4": 96.07,  # SO4(2-)
            "CO2": 44.009,  # CO2
            "Br": 79.90,  # Br(-)
            "BOH3": 61.84,  # B(OH)3
            "F": 18.99840316,  # F(-)
            "NH3": 17.031,  # NH3
            "NO2": 46.006,  # NO2
            "H2S": 34.08,  # H2S
            "PO4": 94.971,  # PO4(3-)
            "H4SiO4": 96.11,  # H4SiO4
            "H": 1.008,  # H(+)
            "CO3": 60.009,  # CO3(2-)
            "OH": 17.007,  # OH(-)
            "HSO4": 97.07,  # HSO4(-)
            "HS": 33.08,  # HS(-)
            "BOH4": 78.84,  # B(OH)4(-)
            "NH4": 18.039,  # NH4(+)
            "H3SiO4": 95.11,  # H3SiO4(-)
            "CaF": 59.08,  # CaF(+)
            "MgF": 43.304,  # MgF(+)
            "HF": 20.006,  # HF
            "MgCO3": 84.31,  # MgCO3
            "CaCO3": 100.09,  # CaCO3
            "SrCO3": 147.6,  # SrCO3
            "HCO3": 61.017,  # HCO3(-)
            "HPO4": 95.979,  # HPO4(-)
            "H2PO4": 96.987,  # H2PO4(-)
            "H3PO4": 97.995,  # H3PO4
            "MgH2PO4": 121.3018,  # MgH2PO4(+) (this value from pwb.com)
            "MgHPO4": 120.28,  # MgHPO4
            "MgPO4": 119.28,  # MgPO4(-)
            "CaH2PO4": 137.07,  # CaH2PO4(+)
            "CaHPO4": 136.06,  # CaHPO4
            "CaPO4": 135.05,  # CaPO4(-)
            "MgOH": 41.313,  # MgOH(+)
        }
    )

    # Dict of weight concentrations (g/kg)
    gkg = OrderedDict((key, molalities[key] * MW[key]) for key in molalities.keys())

    # Get H2O fraction: 1 kg H2O / ((sum weights in 1 kg H2O) + 1 kg H2O)
    h2o = 1 / (sum(gkg.values()) + 1)

    # Convert concentrations
    concentrations = OrderedDict(
        (key, molalities[key] * h2o) for key in molalities.keys()
    )

    # Convert Ks
    ks_out = OrderedDict((key, ks[key] * h2o) for key in ks.keys())

    return concentrations, ks_out
