# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
import numpy as np
from . import convert


def expand_solute_molalities(solute_molalities):
    """Get solutes, molalities and charges from the solute dict."""
    solutes = []
    molalities = []
    charges = []
    for solute, molality in solute_molalities.items():
        solutes.append(solute)
        molalities.append(molality)
        charges.append(convert.solute_to_charge[solute])
    return solutes, np.array(molalities), np.array(charges)


def split_solute_types(values, charges):
    """Split up input values into cations, anions and neutrals."""
    cations = np.compress(charges > 0, values)
    anions = np.compress(charges < 0, values)
    neutrals = np.compress(charges == 0, values)
    return cations, anions, neutrals


def get_pytzer_args(solute_molalities):
    solutes, molalities, charges = expand_solute_molalities(solute_molalities)
    m_cats, m_anis, m_neus = split_solute_types(molalities, charges)
    z_cats, z_anis = split_solute_types(charges, charges)[:2]
    pytzer_args = (m_cats, m_anis, m_neus, z_cats, z_anis)
    solutes_split = {}
    (
        solutes_split["cations"],
        solutes_split["anions"],
        solutes_split["neutrals"],
    ) = split_solute_types(solutes, charges)
    return pytzer_args, solutes_split


def salinity_to_molalities_MZF93(salinity, MgOH=False):
    """Convert salinity (g/kg-sw) to molality for typical seawater, simplified
    for the WM13 tris buffer model, following MZF93.
    """
    if MgOH:
        mols = np.array([0.44516, 0.01077, 0.01058, 0.56912]) * salinity / 43.189
        ions = np.array(["Na", "Ca", "K", "Cl"])
        tots = np.array([0.02926, 0.05518]) * salinity / 43.189
        eles = np.array(["t_HSO4", "t_Mg"])
    else:
        mols = (
            np.array([0.44516, 0.05518, 0.01077, 0.01058, 0.56912]) * salinity / 43.189
        )
        ions = np.array(["Na", "Mg", "Ca", "K", "Cl"])
        tots = np.array([0.02926]) * salinity / 43.189
        eles = np.array(["t_HSO4"])
    solute_molalities = {ion: mol for ion, mol in zip(ions, mols)}
    solute_molalities.update({ele: tot for ele, tot in zip(eles, tots)})
    return solute_molalities


def salinity_to_molalities_MFWM08(salinity=35):
    """Convert salinity (g/kg-sw) to molality for standard seawater following MFWM08."""
    solute_molalities = {
        "Na": 0.4860597,
        "Mg": 0.0547421,
        "Ca": 0.0106568,
        "K": 0.0105797,
        "Sr": 0.0000940,
        "Cl": 0.5657647,
        "SO4": 0.0292643,
        "HCO3": 0.0017803,
        "Br": 0.0008728,
        "CO3": 0.0002477,
        "BOH4": 0.0001045,
        "F": 0.0000708,
        "OH": 0.0000082,
        "BOH3": 0.0003258,
        "CO2": 0.0000100,
    }
    solute_molalities = {
        solute: molality * salinity / 35
        for solute, molality in solute_molalities.items()
    }
    return solute_molalities


def salinity_to_totals_MFWM08(salinity=35):
    """Convert salinity (g/kg-sw) to total molality for standard seawater following MFWM08."""
    total_molalities = {
        "Na": 0.4860597,
        "Mg": 0.0547421,
        "Ca": 0.0106568,
        "K": 0.0105797,
        "Sr": 0.0000940,
        "Cl": 0.5657647,
        "SO4": 0.0292643,
        "CO2": 0.0017803 + 0.0002477 + 0.0000100,
        "Br": 0.0008728,
        "B": 0.0001045 + 0.0003258,
        "F": 0.0000708,
    }
    total_molalities = {
        total: molality * salinity / 35
        for total, molality in total_molalities.items()
    }
    return total_molalities
