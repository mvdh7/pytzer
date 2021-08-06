# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
from collections import OrderedDict
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
    solute_molalities = OrderedDict(ion_mol for ion_mol in zip(ions, mols))
    solute_molalities.update({ele: tot for ele, tot in zip(eles, tots)})
    return solute_molalities


def salinity_to_molalities_MFWM08(salinity=35):
    """Convert salinity (g/kg-sw) to molality for standard seawater following MFWM08."""
    solute_molalities = OrderedDict()
    solute_molalities.update(
        {
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
    )
    solute_molalities = OrderedDict(
        (solute, molality * salinity / 35)
        for solute, molality in solute_molalities.items()
    )
    return solute_molalities


def salinity_to_totals_MFWM08(salinity=35):
    """Convert salinity (g/kg-sw) to total molality for standard seawater following MFWM08."""
    total_molalities = OrderedDict()
    total_molalities.update(
        {
            "Na": 0.4860597,
            "Mg": 0.0547421,
            "Ca": 0.0106568,
            "K": 0.0105797,
            "Sr": 0.0000940,
            "Cl": 0.5657647,
            "SO4": 0.0292643,
            "CO2": 0.0017803 + 0.0002477 + 0.0000100,
            "Br": 0.0008728,
            "BOH3": 0.0001045 + 0.0003258,
            "F": 0.0000708,
        }
    )
    total_molalities = OrderedDict(
        (total, molality * salinity / 35)
        for total, molality in total_molalities.items()
    )
    return total_molalities


salt_to_solutes = {  # used for conversions in subsequent functions
    "BaCl2": dict(Ba=1, Cl=2),
    "CaCl2": dict(Ca=1, Cl=2),
    "CsCl": dict(Cs=1, Cl=1),
    "CuCl2": dict(Cujj=1, Cl=2),
    "CuSO4": dict(Cujj=1, SO4=1),
    "Eu(NO3)3": dict(Eujjj=1, NO3=3),
    "H2SO4": dict(H=2, SO4=1),  # need to enable equilibration!
    "H3BO3": dict(H=3, BO3=1),  # need to enable equilibration?
    "K2B4O7": dict(K=2, B4O7=1),
    "KBr": dict(K=1, Br=1),
    "K2CO3": dict(K=2, CO2=1),
    "KCl": dict(K=1, Cl=1),
    "LiCl": dict(Li=1, Cl=1),
    "MgCl2": dict(Mg=1, Cl=2),
    "MgSO4": dict(Mg=1, SO4=1),
    "Na2Mg(SO4)2": dict(Na=2, Mg=1, SO4=2),
    "Na2SO4": dict(Na=2, SO4=1),
    "NaCl": dict(Na=1, Cl=1),
    "Na2CO3": dict(Na=2, CO2=1),
    "NaF": dict(Na=1, F=1),
    "NaHCO3": dict(Na=1, H=1, CO2=1),
    "NaNO3": dict(Na=1, NO3=1),
    "NiSO4": dict(Ni=1, SO4=1),
    "SrCl2": dict(Sr=1, Cl=2),
    "glycerol": dict(glycerol=1),
    "sucrose": dict(sucrose=1),
    "urea": dict(urea=1),
    "(trisH)2SO4": dict(trisH=2, SO4=1),
    "tris": dict(tris=1),
    "trisHCl": dict(trisH=1, Cl=1),
}


def salinity_to_totals_RRV93(salinity=35):
    """Convert salinity (g/kg-sw) to total molality for synthetic seawater following RRV93."""
    global salt_to_solutes
    salt_molalities = OrderedDict()
    salt_molalities.update(
        {
            "NaCl": 0.42464,
            "Na2SO4": 0.02927,
            "KCl": 0.01058,
            "MgCl2": 0.05474,
            "CaCl2": 0.01075,
            "Na2CO3": 0.001,
            "NaHCO3": 0.001,
        }
    )

    # Convert salts to ions
    total_molalities = OrderedDict()
    for salt, mol in salt_molalities.items():
        for k, v in salt_to_solutes[salt].items():
            if k not in total_molalities.keys():
                total_molalities[k] = 0
            total_molalities[k] += v * mol

    # Convert to required salinity
    total_molalities = OrderedDict(
        (total, molality * salinity / 35)
        for total, molality in total_molalities.items()
    )
    return total_molalities


def salinity_to_totals_GP89(salinity=35):
    """Convert salinity (g/kg-sw) to total molality for synthetic seawater following GP89."""
    global salt_to_solutes
    gravimetric_salts = (
        OrderedDict()
    )  # Concentrations given in g/kg solution in the paper
    gravimetric_salts.update(
        {
            "NaCl": 23.926,
            "Na2SO4": 4.008,
            "KCl": 0.677,
            "KBr": 0.098,
            "NaF": 0.003,
            "Na2CO3": 0.106,
        }
    )

    volumetric_salts = (
        OrderedDict()
    )  # Concentrations given in mol/kg solution in the paper
    volumetric_salts.update({"MgCl2": 0.05327, "CaCl2": 0.01033, "SrCl2": 9e-05})

    # Formula weights are provided in the paper for the gravimetric salts, the rest are taken from PubChem
    formula_weights = OrderedDict()
    formula_weights.update(
        {
            "NaCl": 58.44,
            "Na2SO4": 142.04,
            "KCl": 74.56,
            "KBr": 119.01,
            "NaF": 41.99,
            "Na2CO3": 105.99,
            "MgCl2": 95.21,
            "CaCl2": 110.98,
            "SrCl2": 158.5,
        }
    )

    # Get mol/kg solution for gravimetric salts
    volumetric_salts.update(
        {
            key: gravimetric_salts[key] / formula_weights[key]
            for key in gravimetric_salts.keys()
        }
    )

    # Get g/kg solution for volumetric salts
    gravimetric_salts.update(
        {
            key: formula_weights[key] * volumetric_salts[key]
            for key in ["MgCl2", "CaCl2", "SrCl2"]
        }
    )

    h2o = 1 - sum(gravimetric_salts.values()) / 1000  # fraction of H2O in solution

    # Convert mol/kg-solution to mol/kg-solvent (molal)
    salt_molalities = OrderedDict()
    salt_molalities.update(
        {key: volumetric_salts[key] / h2o for key in volumetric_salts.keys()}
    )

    # Convert salts to ions
    total_molalities = OrderedDict()
    for salt, mol in salt_molalities.items():
        for k, v in salt_to_solutes[salt].items():
            if k not in total_molalities.keys():
                total_molalities[k] = 0
            total_molalities[k] += v * mol

    # Convert to required salinity
    total_molalities = OrderedDict(
        (total, molality * salinity / 35)
        for total, molality in total_molalities.items()
    )
    return total_molalities


def salinity_to_totals_H73a(salinity=35):
    """Convert salinity (g/kg-sw) to total molality for synthetic seawater following H73a."""
    ion_concentrations = OrderedDict(  # given in mMol/kg of solution in the paper
        {"Na": 478, "Mg": 54, "Ca": 10, "Cl": 550, "SO4": 28, "HCO3": 1.3, "CO3": 0.7}
    )

    # Convert to mol/kg of solution
    ion_concentrations.update(
        (key, value / 1000) for key, value in ion_concentrations.items()
    )

    # Formula weights
    formula_weights = OrderedDict()
    formula_weights.update(
        {
            "Na": 22.98976928,
            "Mg": 24.3039,
            "Ca": 40.077,
            "Cl": 35.454,
            "SO4": 96.064,
            "HCO3": 61.0168,
            "CO3": 60.009,
        }
    )

    # Get g/kg of solution
    gkg = OrderedDict(
        (key, ion_concentrations[key] * formula_weights[key])
        for key in ion_concentrations.keys()
    )

    h2o = 1 - sum(gkg.values()) / 1000  # fraction of H2O in solution

    # Convert mol/kg-solution to mol/kg-solvent (molal)
    total_molalities = OrderedDict()
    total_molalities.update(
        {key: ion_concentrations[key] / h2o for key in ion_concentrations.keys()}
    )

    # Replace individual carbonates with total CO2
    total_molalities["CO2"] = total_molalities["CO3"] + total_molalities["HCO3"]
    del total_molalities["CO3"]
    del total_molalities["HCO3"]

    # Convert to required salinity
    total_molalities = OrderedDict(
        (total, molality * salinity / 35)
        for total, molality in total_molalities.items()
    )
    return total_molalities


def salinity_to_totals_D90a(salinity=35):
    """Convert salinity (g/kg-sw) to total molality for synthetic seawater following D90a."""
    global salt_to_solutes
    salt_molalities = OrderedDict()
    salt_molalities.update(
        {
            "NaCl": 0.42764,
            "Na2SO4": 0.02927,
            "KCl": 0.01058,
            "MgCl2": 0.05474,
            "CaCl2": 0.01075,
        }
    )

    # Convert salts to ions
    total_molalities = OrderedDict()
    for salt, mol in salt_molalities.items():
        for k, v in salt_to_solutes[salt].items():
            if k not in total_molalities.keys():
                total_molalities[k] = 0
            total_molalities[k] += v * mol

    # Convert to required salinity
    total_molalities = OrderedDict(
        (total, molality * salinity / 35)
        for total, molality in total_molalities.items()
    )
    return total_molalities


def salinity_to_totals_KRCB77(salinity=35):
    """Convert salinity (g/kg-sw) to total molality for synthetic seawater following KRCB77."""
    global salt_to_solutes
    salt_molalities = OrderedDict()
    salt_molalities.update(
        {
            "NaCl": 0.42664,
            "Na2SO4": 0.02926,
            "KCl": 0.01058,
            "MgCl2": 0.05518,
            "CaCl2": 0.01077,
        }
    )

    # Convert salts to ions
    total_molalities = OrderedDict()
    for salt, mol in salt_molalities.items():
        for k, v in salt_to_solutes[salt].items():
            if k not in total_molalities.keys():
                total_molalities[k] = 0
            total_molalities[k] += v * mol

    # Convert to required salinity
    total_molalities = OrderedDict(
        (total, molality * salinity / 35)
        for total, molality in total_molalities.items()
    )
    return total_molalities


def salinity_to_totals_DR79(salinity=35):
    """Convert salinity (g/kg-sw) to total molality for synthetic seawater following DR79."""

    total_molalities = OrderedDict()
    total_molalities.update(
        {
            "Na": 0.67284,
            "Mg": 0.07577,
            "Ca": 0.01474,
            "K": 0.01464,
            "Sr": 0.00012,
            "Cl": 0.78642,
            "SO4": 0.04051,
            "Br": 0.0012,
            "F": 0.0001,
        }
    )

    # Convert to required salinity
    total_molalities = OrderedDict(
        (total, molality * salinity / 35)
        for total, molality in total_molalities.items()
    )
    return total_molalities
