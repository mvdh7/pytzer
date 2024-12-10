# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
"""Solve for thermodynamic equilibrium."""


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
        log_acfs["Mg"] + log_acfs["OH"] - log_acfs["MgOH"] + log_ks_MgOH - log_kt_MgOH
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


reactions_all = {
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
