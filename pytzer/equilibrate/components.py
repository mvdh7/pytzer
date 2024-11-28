# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
"""Calculate molality of each solution component."""


def get_OH(h, ks_constants):
    k = ks_constants
    return k["H2O"] / h


def get_F(h, totals, ks_constants):
    t, k = totals, ks_constants
    return k["HF"] * t["F"] / (k["HF"] + h)


def get_HF(h, f, ks_constants):
    k = ks_constants
    return f * h / k["HF"]


def get_CO3(h, totals, ks_constants):
    t, k = totals, ks_constants
    return (
        t["CO2"]
        * k["H2CO3"]
        * k["HCO3"]
        / (h**2 + k["H2CO3"] * h + k["H2CO3"] * k["HCO3"])
    )


def get_HCO3(h, co3, ks_constants):
    k = ks_constants
    return co3 * h / k["HCO3"]


def get_CO2(h, co3, ks_constants):
    k = ks_constants
    return co3 * h**2 / (k["H2CO3"] * k["HCO3"])


def get_PO4(h, totals, ks_constants):
    t, k = totals, ks_constants
    return (
        t["PO4"]
        * k["H3PO4"]
        * k["H2PO4"]
        * k["HPO4"]
        / (
            h**3
            + k["H3PO4"] * h**2
            + k["H3PO4"] * k["H2PO4"] * h
            + k["H3PO4"] * k["H2PO4"] * k["HPO4"]
        )
    )


def get_HPO4(h, po4, ks_constants):
    k = ks_constants
    return po4 * h / k["HPO4"]


def get_H2PO4(h, po4, ks_constants):
    k = ks_constants
    return po4 * h**2 / (k["H2PO4"] * k["HPO4"])


def get_H3PO4(h, po4, ks_constants):
    k = ks_constants
    return po4 * h**3 / (k["H3PO4"] * k["H2PO4"] * k["HPO4"])


def get_SO4(h, totals, ks_constants):
    t, k = totals, ks_constants
    return k["HSO4"] * t["SO4"] / (h + k["HSO4"])


def get_HSO4(h, totals, ks_constants):
    t, k = totals, ks_constants
    return h * t["SO4"] / (h + k["HSO4"])


def get_NO2(h, totals, ks_constants):
    t, k = totals, ks_constants
    return k["HNO2"] * t["NO2"] / (h + k["HNO2"])


def get_HNO2(h, totals, ks_constants):
    t, k = totals, ks_constants
    return h * t["NO2"] / (h + k["HNO2"])


def get_NH3(h, totals, ks_constants):
    t, k = totals, ks_constants
    return k["NH4"] * t["NH3"] / (h + k["NH4"])


def get_NH4(h, totals, ks_constants):
    t, k = totals, ks_constants
    return h * t["NH3"] / (h + k["NH4"])


def get_HS(h, totals, ks_constants):
    t, k = totals, ks_constants
    return k["H2S"] * t["H2S"] / (h + k["H2S"])


def get_H2S(h, totals, ks_constants):
    t, k = totals, ks_constants
    return h * t["H2S"] / (h + k["H2S"])


def get_H3SiO4(h, totals, ks_constants):
    t, k = totals, ks_constants
    return k["H4SiO4"] * t["H4SiO4"] / (h + k["H4SiO4"])


def get_H4SiO4(h, totals, ks_constants):
    t, k = totals, ks_constants
    return h * t["H4SiO4"] / (h + k["H4SiO4"])


def get_BOH4(h, totals, ks_constants):
    t, k = totals, ks_constants
    return k["BOH3"] * t["BOH3"] / (h + k["BOH3"])


def get_BOH3(h, totals, ks_constants):
    t, k = totals, ks_constants
    return h * t["BOH3"] / (h + k["BOH3"])


def get_tris(h, totals, ks_constants):
    t, k = totals, ks_constants
    return k["trisH"] * t["tris"] / (h + k["trisH"])


def get_trisH(h, totals, ks_constants):
    t, k = totals, ks_constants
    return h * t["tris"] / (h + k["trisH"])


def get_Ca(h, f, co3, po4, totals, ks_constants):
    t, k = totals, ks_constants
    denom = 1.0
    if "CaF" in k:
        denom = denom + k["CaF"] * f
    if "CaCO3" in k:
        denom = denom + k["CaCO3"] * co3
    if "CaH2PO4" in k:
        H2PO4 = get_H2PO4(h, po4, ks_constants)
        denom = denom + k["CaH2PO4"] * H2PO4
    if "CaHPO4" in k:
        HPO4 = get_HPO4(h, po4, ks_constants)
        denom = denom + k["CaHPO4"] * HPO4
    if "CaPO4" in k:
        denom = denom + k["CaPO4"] * po4
    return t["Ca"] / denom


def get_CaF(h, f, co3, po4, totals, ks_constants):
    Ca = get_Ca(h, f, co3, po4, totals, ks_constants)
    k = ks_constants
    return k["CaF"] * Ca * f


def get_CaCO3(h, f, co3, po4, totals, ks_constants):
    Ca = get_Ca(h, f, co3, po4, totals, ks_constants)
    k = ks_constants
    return k["CaCO3"] * Ca * co3


def get_CaH2PO4(h, f, co3, po4, totals, ks_constants):
    Ca = get_Ca(h, f, co3, po4, totals, ks_constants)
    H2PO4 = get_H2PO4(h, po4, ks_constants)
    k = ks_constants
    return k["CaH2PO4"] * Ca * H2PO4


def get_CaHPO4(h, f, co3, po4, totals, ks_constants):
    Ca = get_Ca(h, f, co3, po4, totals, ks_constants)
    HPO4 = get_HPO4(h, po4, ks_constants)
    k = ks_constants
    return k["CaHPO4"] * Ca * HPO4


def get_CaPO4(h, f, co3, po4, totals, ks_constants):
    Ca = get_Ca(h, f, co3, po4, totals, ks_constants)
    k = ks_constants
    return k["CaPO4"] * Ca * po4


def get_Mg(h, f, co3, po4, totals, ks_constants):
    t, k = totals, ks_constants
    denom = 1.0
    if "MgOH" in k:
        OH = get_OH(h, ks_constants)
        denom = denom + OH / k["MgOH"]
    if "MgF" in k:
        denom = denom + k["MgF"] * f
    if "MgCO3" in k:
        denom = denom + k["MgCO3"] * co3
    if "MgH2PO4" in k:
        H2PO4 = get_H2PO4(h, po4, ks_constants)
        denom = denom + k["MgH2PO4"] * H2PO4
    if "MgHPO4" in k:
        HPO4 = get_HPO4(h, po4, ks_constants)
        denom = denom + k["MgHPO4"] * HPO4
    if "MgPO4" in k:
        denom = denom + k["MgPO4"] * po4
    return t["Mg"] / denom


def get_MgOH(h, f, co3, po4, totals, ks_constants):
    Mg = get_Mg(h, f, co3, po4, totals, ks_constants)
    OH = get_OH(h, ks_constants)
    k = ks_constants
    return Mg * OH / k["MgOH"]


def get_MgF(h, f, co3, po4, totals, ks_constants):
    Mg = get_Mg(h, f, co3, po4, totals, ks_constants)
    k = ks_constants
    return k["MgF"] * Mg * f


def get_MgCO3(h, f, co3, po4, totals, ks_constants):
    Mg = get_Mg(h, f, co3, po4, totals, ks_constants)
    k = ks_constants
    return k["MgCO3"] * Mg * co3


def get_MgH2PO4(h, f, co3, po4, totals, ks_constants):
    Mg = get_Mg(h, f, co3, po4, totals, ks_constants)
    H2PO4 = get_H2PO4(h, po4, ks_constants)
    k = ks_constants
    return k["MgH2PO4"] * Mg * H2PO4


def get_MgHPO4(h, f, co3, po4, totals, ks_constants):
    Mg = get_Mg(h, f, co3, po4, totals, ks_constants)
    HPO4 = get_HPO4(h, po4, ks_constants)
    k = ks_constants
    return k["MgHPO4"] * Mg * HPO4


def get_MgPO4(h, f, co3, po4, totals, ks_constants):
    Mg = get_Mg(h, f, co3, po4, totals, ks_constants)
    k = ks_constants
    return k["MgPO4"] * Mg * po4


def get_Sr(co3, totals, ks_constants):
    t, k = totals, ks_constants
    return t["Sr"] / (1 + k["SrCO3"] * co3)


def get_SrCO3(co3, totals, ks_constants):
    t, k = totals, ks_constants
    return t["Sr"] * k["SrCO3"] * co3 / (1 + k["SrCO3"] * co3)
