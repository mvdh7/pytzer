# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Evaluate thermodynamic equilibrium constants."""
from collections import OrderedDict
from jax import numpy as np

ln10 = np.log(10)


def HSO4_CRP94_extra(T=298.15):
    """Bisulfate dissociation: CRP94 Eq. (21) with extra digits on constant
    term (S.L. Clegg, pers. comm, 7 Feb 2019).
    """
    log10kHSO4 = (
        562.694864456
        - 102.5154 * np.log(T)
        - 1.117033e-4 * T ** 2
        + 0.2477538 * T
        - 13273.75 / T
    )
    lnkHSO4 = log10kHSO4 * ln10
    return lnkHSO4


def HSO4_CRP94(T=298.15):
    """CRP94 Eq. (21) without extra digits on constant term."""
    # Matches Clegg's model [2019-07-02]
    log10kHSO4 = (
        562.69486
        - 102.5154 * np.log(T)
        - 1.117033e-4 * T ** 2
        + 0.2477538 * T
        - 13273.75 / T
    )
    lnkHSO4 = log10kHSO4 * ln10
    return lnkHSO4


def trisH_BH64(T=298.15):
    """TrisH+ dissociation following BH64 Eq. (3)."""
    # Matches Clegg's model [2019-07-02]
    log10ktrisH = -(2981.4 / T - 3.5888 + 0.005571 * T)
    lnktrisH = log10ktrisH * ln10
    return lnktrisH


def rhow_K75(T=298.15):
    """Water density in g/cm**3 following Kell (1975) J. Chem. Eng. Data 20(1),
    97-105.
    """
    # Matches Clegg's model [2019-07-02]
    tempC = T - 273.15
    return (
        0.99983952
        + 16.945176e-3 * tempC
        - 7.9870401e-6 * tempC ** 2
        - 46.170461e-9 * tempC ** 3
        + 105.56302e-12 * tempC ** 4
        - 280.54253e-15 * tempC ** 5
    ) / (1 + 16.879850e-3 * tempC)


def H2O_M79(T=298.15):
    """Water dissociation following M79."""
    # MP98 says this is HO58 refitted by M79
    return 148.9802 - 13847.26 / T - 23.6521 * np.log(T)


def H2O_M88(T=298.15):
    """Water dissociation following M88."""
    return (
        1.04031130e3
        + 4.86092851e-1 * T
        - 3.26224352e4 / T
        - 1.90877133e2 * np.log(T)
        - 5.35204850e-1 / (T - 263)
        - 2.32009393e-4 * T ** 2
        + 5.20549183e1 / (680 - T)
    )


def H2O_MF(T=298.15):
    """Marshall and Frank, J. Phys. Chem. Ref. Data 10, 295-304."""
    # Matches Clegg's model [2019-07-02]
    log10kH2O = (
        -4.098
        - 3.2452e3 / T
        + 2.2362e5 / T ** 2
        - 3.984e7 / T ** 3
        + (1.3957e1 - 1.2623e3 / T + 8.5641e5 / T ** 2) * np.log10(rhow_K75(T))
    )
    lnkH2O = log10kH2O * ln10
    return lnkH2O


def MgOH_CW91_ln(T=298.15):
    """MgOH+ formation following CW91 Eq. (244) [p392]."""
    return -(8.9108 - 1155 / T)


def MgOH_CW91(T=298.15):
    """MgOH+ formation following CW91 in log10 and then converted."""
    # Matches Clegg's model [2019-07-02]
    log10kMg = -(3.87 - 501.5 / T)
    lnkMg = log10kMg * ln10
    return lnkMg


def MgOH_MP98(T=298.15):
    """MgOH+ formation following MP98."""
    log10kMg = -(3.87 - 501.6 / T)
    lnkMg = log10kMg * ln10
    return lnkMg


def _MP98_eq23(T, A=0, B=0, C=0, D=0):
    """Equation (23) of MP98.  Returns ln(K)."""
    return A + B / T + C * np.log(T) + D * T


def HF_MP98(T=298.15):
    """Hydrogen fluoride dissociation [MP98 following DR79a]."""
    return _MP98_eq23(T, A=-12.641, B=1590.2)


def H2S_MP98(T=298.15):
    """Hydrogen sulfide dissociation [MP98 following M88]."""
    return _MP98_eq23(T, A=225.8375, B=-13275.324, C=-34.64354)


def BOH3_M79(T=298.15):
    """Boric acid dissociation [MP98 following M79]."""
    return _MP98_eq23(T, A=148.0248, B=-8966.901, C=-24.4344)


def NH4_MP98(T=298.15):
    """Ammonium dissociation [MP98 following BP49]."""
    return _MP98_eq23(T, A=-0.25444, B=-6285.33, D=0.0001635)


def H2CO3_MP98(T=298.15):
    """H2CO3 dissociation [MP98 following M79]."""
    return _MP98_eq23(T, A=290.9097, B=-14554.21, C=-45.0575)


def HCO3_MP98(T=298.15):
    """HCO3 dissociation [MP98 following M79]."""
    return _MP98_eq23(T, A=207.6548, B=-11843.79, C=-33.6485)


def H3PO4_MP98(T=298.15):
    """H3PO4 dissociation [MP98 following B51]."""
    return _MP98_eq23(T, A=115.54, B=-4576.7518, C=-18.453)


def H2PO4_MP98(T=298.15):
    """H2PO4 dissociation [MP98 following BA43]."""
    return _MP98_eq23(T, A=172.1033, B=-8814.715, C=-27.927)


def HPO4_MP98(T=298.15):
    """HPO4 dissociation [MP98 following SM64]."""
    return _MP98_eq23(T, A=-18.126, B=-3070.75)


def _MP98_eq24(T, A=0, B=0, C=0):
    """Equation (24) of MP98.  Returns pK."""
    return A + B / T + C * T


def MgF_MP98_MR97(T=298.15):
    """MgF+ formation [MP98 following MR97]."""
    return -_MP98_eq24(T, A=3.504, B=-501.6) * ln10


def CaF_MP98_MR97(T=298.15):
    """CaF+ formation [MP98 following MR97]."""
    return -_MP98_eq24(T, A=3.014, B=-501.6) * ln10


def MgCO3_MP98_MR97(T=298.15):
    """MgCO3 formation [MP98 following MR97]."""
    return -_MP98_eq24(T, A=1.028, C=0.0066154) * ln10


def CaCO3_MP98_MR97(T=298.15):
    """CaCO3 formation [MP98 following MR97]."""
    return -_MP98_eq24(T, A=1.178, C=0.0066154) * ln10


def SrCO3_MP98_MR97(T=298.15):
    """SrCO3 formation [MP98 following MR97]."""
    return -_MP98_eq24(T, A=1.028, C=0.0066154) * ln10


def MgH2PO4_MP98_MR97(T=298.15):
    """MgH2PO4+ formation [MP98 following MR97]."""
    return -1.13 * ln10


def CaH2PO4_MP98_MR97(T=298.15):
    """CaH2PO4+ formation [MP98 following MR97]."""
    return -1.0 * ln10


def MgHPO4_MP98_MR97(T=298.15):
    """MgHPO4 formation [MP98 following MR97]."""
    return -2.7 * ln10


def CaHPO4_MP98_MR97(T=298.15):
    """CaHPO4 formation [MP98 following MR97]."""
    return -2.74 * ln10


def MgPO4_MP98_MR97(T=298.15):
    """MgPO4- formation [MP98 following MR97]."""
    return -5.63 * ln10


def CaPO4_MP98_MR97(T=298.15):
    """CaPO4- formation [MP98 following MR97]."""
    return -7.1 * ln10


def pK_MgOH(T=298.15):
    """MgOH+ formation [MP98 following MR97]."""
    return _MP98_eq24(T, A=3.87, B=-501.6)


def pK_MgF(T=298.15):
    """MgF+ formation [MP98 following MR97]."""
    return _MP98_eq24(T, A=3.504, B=-501.6)


def pK_CaF(T=298.15):
    """CaF+ formation [MP98 following MR97]."""
    return _MP98_eq24(T, A=3.014, B=-501.6)


def pK_MgCO3(T=298.15):
    """MgCO3 formation [MP98 following MR97]."""
    return _MP98_eq24(T, A=1.028, C=0.0066154)


def pK_CaCO3(T=298.15):
    """CaCO3 formation [MP98 following MR97]."""
    return _MP98_eq24(T, A=1.178, C=0.0066154)


def pK_SrCO3(T=298.15):
    """SrCO3 formation [MP98 following MR97]."""
    return _MP98_eq24(T, A=1.028, C=0.0066154)


def pK_MgH2PO4(T=298.15):
    """MgH2PO4+ formation [MP98 following MR97]."""
    return 1.13


def pK_CaH2PO4(T=298.15):
    """CaH2PO4+ formation [MP98 following MR97]."""
    return 1.0


def pK_MgHPO4(T=298.15):
    """MgHPO4 formation [MP98 following MR97]."""
    return 2.7


def pK_CaHPO4(T=298.15):
    """CaHPO4 formation [MP98 following MR97]."""
    return 2.74


def pK_MgPO4(T=298.15):
    """MgPO4- formation [MP98 following MR97]."""
    return 5.63


def pK_CaPO4(T=298.15):
    """CaPO4- formation [MP98 following MR97]."""
    return 7.1


all_log_ks = {
    "BOH3": BOH3_M79,
    "H2CO3": H2CO3_MP98,
    "H2O": H2O_M88,
    "HCO3": HCO3_MP98,
    "HF": HF_MP98,
    "HSO4": HSO4_CRP94,
    # "MgOH": lambda T=298.15: np.log(10.0 ** -pK_MgOH(T)),
    "MgOH": MgOH_MP98,
    "trisH": trisH_BH64,
}


def assemble(temperature=298.15, exclude_equilibria=None, totals=None):
    """Evaluate all thermodynamic equilibrium constants."""
    kt_constants = OrderedDict()
    kt_constants["H2O"] = np.exp(H2O_M88(T=temperature))
    if "F" in totals:
        kt_constants["HF"] = np.exp(HF_MP98(T=temperature))
    if "H2S" in totals:
        kt_constants["H2S"] = np.exp(H2S_MP98(T=temperature))
    if "BOH3" in totals:
        kt_constants["BOH3"] = np.exp(BOH3_M79(T=temperature))
    if "SO4" in totals:
        kt_constants["HSO4"] = np.exp(HSO4_CRP94(T=temperature))
    if "NH3" in totals:
        kt_constants["NH4"] = np.exp(NH4_MP98(T=temperature))
    if "CO2" in totals:
        kt_constants["H2CO3"] = np.exp(H2CO3_MP98(T=temperature))
        kt_constants["HCO3"] = np.exp(HCO3_MP98(T=temperature))
    if "PO4" in totals:
        kt_constants["H3PO4"] = np.exp(H3PO4_MP98(T=temperature))
        kt_constants["H2PO4"] = np.exp(H2PO4_MP98(T=temperature))
        kt_constants["HPO4"] = np.exp(HPO4_MP98(T=temperature))
    if "Mg" in totals:
        kt_constants["MgOH"] = np.exp(MgOH_MP98(T=temperature))
        if "F" in totals:
            kt_constants["MgF"] = np.exp(MgF_MP98_MR97(T=temperature))
        if "CO2" in totals:
            kt_constants["MgCO3"] = np.exp(MgCO3_MP98_MR97(T=temperature))
        if "PO4" in totals:
            kt_constants["MgH2PO4"] = np.exp(MgH2PO4_MP98_MR97(T=temperature))
            kt_constants["MgHPO4"] = np.exp(MgHPO4_MP98_MR97(T=temperature))
            kt_constants["MgPO4"] = np.exp(MgPO4_MP98_MR97(T=temperature))
    if "Ca" in totals:
        if "F" in totals:
            kt_constants["CaF"] = np.exp(CaF_MP98_MR97(T=temperature))
        if "CO2" in totals:
            kt_constants["CaCO3"] = np.exp(CaCO3_MP98_MR97(T=temperature))
        if "PO4" in totals:
            kt_constants["CaH2PO4"] = np.exp(CaH2PO4_MP98_MR97(T=temperature))
            kt_constants["CaHPO4"] = np.exp(CaHPO4_MP98_MR97(T=temperature))
            kt_constants["CaPO4"] = np.exp(CaPO4_MP98_MR97(T=temperature))
    if "Sr" in totals and "CO2" in totals:
        kt_constants["SrCO3"] = np.exp(SrCO3_MP98_MR97(T=temperature))
    if exclude_equilibria is not None:
        for eq in exclude_equilibria:
            if eq in kt_constants:
                kt_constants.pop(eq)
    return kt_constants
