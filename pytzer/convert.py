# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
from collections import OrderedDict

from jax import numpy as np

from . import constants


def osmotic_to_activity(molalities, osmotic_coefficient):
    """Convert osmotic coefficient to water activity."""
    return np.exp(-osmotic_coefficient * constants.mass_water * np.sum(molalities))


def activity_to_osmotic(molalities, activity_water):
    """Convert water activity to osmotic coefficient."""
    return -np.log(activity_water) / (constants.mass_water * np.sum(molalities))


def log_activities_to_mean(log_acf_M, log_acf_X, n_M, n_X):
    """Calculate the mean activity coefficient for an electrolyte."""
    return (n_M * log_acf_M + n_X * log_acf_X) / (n_M + n_X)


# Salts to their solutes
salt_to_solute = {
    "BaBr2": dict(Ba=1, Br=2),
    "BaCl2": dict(Ba=1, Cl=2),
    "CaBr2": dict(Ca=1, Br=2),
    "CaCl2": dict(Ca=1, Cl=2),
    "CaI2": dict(Ca=1, I=1),
    "CsBr": dict(Cs=1, Br=1),
    "CsCl": dict(Cs=1, Cl=1),
    "CsI": dict(Cs=1, I=1),
    "H2SO4": dict(H=2, SO4=1),
    "KBr": dict(K=1, Br=1),
    "KCl": dict(K=1, Cl=1),
    "KI": dict(K=1, I=1),
    "KSCN": dict(K=1, SCN=1),
    "LiCl": dict(Li=1, Cl=1),
    "LiNO3": dict(Li=1, NO3=1),
    "MgCl2": dict(Mg=1, Cl=2),
    "NH4NO3": dict(NH4=1, NO3=1),
    "Na2SO4": dict(Na=2, SO4=1),
    "NaBr": dict(Na=1, Br=1),
    "NaCl": dict(Na=1, Cl=1),
    "NaI": dict(Na=1, I=1),
    "NaOH": dict(Na=1, OH=1),
    "RbCl": dict(Rb=1, Cl=1),
    "SrBr2": dict(Sr=1, Br=2),
    "SrCl2": dict(Sr=1, Cl=2),
    "SrI2": dict(Sr=1, I=2),
}


def from_salt_get_solutes(salt, molality):
    solutes = OrderedDict()
    if salt in salt_to_solute:
        for k, v in salt_to_solute[salt].items():
            solutes[k] = molality * v
    return solutes


salt_format = {
    "BaBr2": "BaBr$_2$",
    "BaCl2": "BaCl$_2$",
    "CaBr2": "CaBr$_2$",
    "CaCl2": "CaCl$_2$",
    "CaI2": "CaI$_2$",
    "CsBr": "CsBr",
    "CsCl": "CsCl",
    "CsI": "CsI",
    "H2SO4": "H$_2$SO$_4$",
    "KBr": "KBr",
    "KCl": "KCl",
    "KI": "KI",
    "LiCl": "LiCl",
    "LiNO3": "LiNO$_3$",
    "MgCl2": "MgCl$_2$",
    "NH4NO3": "NH4NO3",
    "Na2SO4": "Na$_2$SO$_4$",
    "NaBr": "NaBr",
    "NaCl": "NaCl",
    "NaI": "NaI",
    "NaOH": "NaOH",
    "RbCl": "RbCl",
    "KSCN": "KSCN",
    "SrBr2": "SrBr$_2$",
    "SrCl2": "SrCl$_2$",
    "SrI2": "SrI$_2$",
}

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
