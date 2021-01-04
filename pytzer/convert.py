# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
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
