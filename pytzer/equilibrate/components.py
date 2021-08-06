# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Calculate molality of each solution component."""

from collections import OrderedDict
import jax
from . import stoichiometric


def get_wrap(get_func):
    """Wrapper to return 0.0 in case of KeyError."""

    def get_wrapped(*args):
        try:
            get_out = get_func(*args)
        except KeyError:
            get_out = 0.0
        return get_out

    return get_wrapped


@get_wrap
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
        / (h ** 2 + k["H2CO3"] * h + k["H2CO3"] * k["HCO3"])
    )


def get_HCO3(h, co3, ks_constants):
    k = ks_constants
    return co3 * h / k["HCO3"]


def get_CO2(h, co3, ks_constants):
    k = ks_constants
    return co3 * h ** 2 / (k["H2CO3"] * k["HCO3"])


def get_PO4(h, totals, ks_constants):
    t, k = totals, ks_constants
    return (
        t["PO4"]
        * k["H3PO4"]
        * k["H2PO4"]
        * k["HPO4"]
        / (
            h ** 3
            + k["H3PO4"] * h ** 2
            + k["H3PO4"] * k["H2PO4"] * h
            + k["H3PO4"] * k["H2PO4"] * k["HPO4"]
        )
    )


@get_wrap
def get_HPO4(h, po4, ks_constants):
    k = ks_constants
    return po4 * h / k["HPO4"]


@get_wrap
def get_H2PO4(h, po4, ks_constants):
    k = ks_constants
    return po4 * h ** 2 / (k["H2PO4"] * k["HPO4"])


def get_H3PO4(h, po4, ks_constants):
    k = ks_constants
    return po4 * h ** 3 / (k["H3PO4"] * k["H2PO4"] * k["HPO4"])


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


def get_Ca(h, f, co3, po4, totals, ks_constants):
    H2PO4 = get_H2PO4(h, po4, ks_constants)
    HPO4 = get_HPO4(h, po4, ks_constants)
    t, k = totals, ks_constants
    denom = 1.0
    if "CaF" in k:
        denom = denom + k["CaF"] * f
    if "CaCO3" in k:
        denom = denom + k["CaCO3"] * co3
    if "CaH2PO4" in k:
        denom = denom + k["CaH2PO4"] * H2PO4
    if "CaHPO4" in k:
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
    OH = get_OH(h, ks_constants)
    H2PO4 = get_H2PO4(h, po4, ks_constants)
    HPO4 = get_HPO4(h, po4, ks_constants)
    t, k = totals, ks_constants
    denom = 1.0
    if "MgOH" in k:
        denom = denom + k["MgOH"] * OH
    if "MgF" in k:
        denom = denom + k["MgF"] * f
    if "MgCO3" in k:
        denom = denom + k["MgCO3"] * co3
    if "MgH2PO4" in k:
        denom = denom + k["MgH2PO4"] * H2PO4
    if "MgHPO4" in k:
        denom = denom + k["MgHPO4"] * HPO4
    if "MgPO4" in k:
        denom = denom + k["MgPO4"] * po4
    return t["Mg"] / denom


def get_MgOH(h, f, co3, po4, totals, ks_constants):
    Mg = get_Mg(h, f, co3, po4, totals, ks_constants)
    OH = get_OH(h, ks_constants)
    k = ks_constants
    return k["MgOH"] * Mg * OH


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


def find_solutes(totals, ks_constants, ptargets=None):
    """Get set of solutes in final solution given totals and ks_constants."""
    # Solutes MUST be added in EXACTLY the same order as in get_solutes()!
    if ptargets is None:
        ptargets = stoichiometric.create_ptargets(totals, ks_constants)
    targets = ptargets
    solutes = []
    solutes = solutes + list(totals.keys())
    solutes.append("H")  # H must always be a target (for now)
    if "F" in targets:
        solutes.append("F")
    else:
        if "F" in totals:
            solutes.append("F")
    if "CO3" in targets:
        solutes.append("CO3")
    else:
        if "CO2" in totals:
            solutes.append("CO3")
    if "PO4" in targets:
        solutes.append("PO4")
    else:
        if "PO4" in totals:
            solutes.append("PO4")
    if "H2O" in ks_constants:
        solutes.append("OH")
    if (
        "H2SO4" in totals or "HSO4" in totals or "SO4" in totals
    ) and "HSO4" in ks_constants:
        solutes.append("HSO4")
        solutes.append("SO4")
    if "H2S" in totals and "H2S" in ks_constants:
        solutes.append("H2S")
        solutes.append("HS")
    if "BOH3" in totals and "BOH3" in ks_constants:
        solutes.append("BOH3")
        solutes.append("BOH4")
    if "NH3" in totals and "NH4" in ks_constants:
        solutes.append("NH3")
        solutes.append("NH4")
    if "H4SiO4" in totals and "H4SiO4" in ks_constants:
        solutes.append("H3SiO4")
        solutes.append("H4SiO4")
    if "NO2" in totals and "HNO2" in ks_constants:
        solutes.append("HNO2")
        solutes.append("NO2")
    if "F" in targets and "F" in totals:
        if "Ca" in totals and "CaF" in ks_constants:
            solutes.append("CaF")
        if "Mg" in totals and "MgF" in ks_constants:
            solutes.append("MgF")
    if "F" in totals and "HF" in ks_constants:
        solutes.append("HF")
    if "CO3" in targets:
        if "CO2" in totals:
            if "Mg" in totals and "MgCO3" in ks_constants:
                solutes.append("MgCO3")
            if "Ca" in totals and "CaCO3" in ks_constants:
                solutes.append("CaCO3")
            if "Sr" in totals and "SrCO3" in ks_constants:
                solutes.append("Sr")
                solutes.append("SrCO3")
    if "CO2" in totals and "HCO3" in ks_constants:
        solutes.append("HCO3")
        if "H2CO3" in ks_constants:
            solutes.append("CO2")
    if "PO4" in totals and "HPO4" in ks_constants:
        solutes.append("HPO4")
        if "H2PO4" in ks_constants:
            solutes.append("H2PO4")
            if "H3PO4" in ks_constants:
                solutes.append("H3PO4")
    if "PO4" in targets and "PO4" in totals:
        if "Mg" in totals:
            if "MgH2PO4" in ks_constants:
                solutes.append("MgH2PO4")
            if "MgHPO4" in ks_constants:
                solutes.append("MgHPO4")
            if "MgPO4" in ks_constants:
                solutes.append("MgPO4")
        if "Ca" in totals:
            if "CaH2PO4" in ks_constants:
                solutes.append("CaH2PO4")
            if "CaHPO4" in ks_constants:
                solutes.append("CaHPO4")
            if "CaPO4" in ks_constants:
                solutes.append("CaPO4")
    if "Mg" in totals:
        solutes.append("Mg")
        if "MgOH" in ks_constants:
            solutes.append("MgOH")
    if "Ca" in totals:
        solutes.append("Ca")
    solutes_unique = []
    for solute in solutes:
        if solute not in solutes_unique:
            solutes_unique.append(solute)
    return solutes_unique


@jax.jit
def get_solutes(totals, ks_constants, ptargets):
    targets = OrderedDict((k, 10.0 ** -v) for k, v in ptargets.items())
    solutes = OrderedDict()
    totals = totals.copy()
    solutes.update(totals)
    solutes["H"] = h = targets["H"]  # H must always be a target (for now)
    if "F" in targets:
        solutes["F"] = f = targets["F"]
    else:
        if "F" in totals:
            solutes["F"] = f = get_F(h, totals, ks_constants)
        else:
            f = 0.0
    if "CO3" in targets:
        solutes["CO3"] = co3 = targets["CO3"]
    else:
        if "CO2" in totals:
            solutes["CO3"] = co3 = get_CO3(h, totals, ks_constants)
        else:
            co3 = 0.0
    if "PO4" in targets:
        solutes["PO4"] = po4 = targets["PO4"]
    else:
        if "PO4" in totals:
            solutes["PO4"] = po4 = get_PO4(h, totals, ks_constants)
        else:
            po4 = 0.0
    if "H2O" in ks_constants:
        solutes["OH"] = get_OH(h, ks_constants)
    if "SO4" in totals and "HSO4" in ks_constants:
        solutes["HSO4"] = get_HSO4(h, totals, ks_constants)
        solutes["SO4"] = get_SO4(h, totals, ks_constants)
    if "H2S" in totals and "H2S" in ks_constants:
        solutes["H2S"] = get_H2S(h, totals, ks_constants)
        solutes["HS"] = get_HS(h, totals, ks_constants)
    if "BOH3" in totals and "BOH3" in ks_constants:
        solutes["BOH3"] = get_BOH3(h, totals, ks_constants)
        solutes["BOH4"] = get_BOH4(h, totals, ks_constants)
    if "NH3" in totals and "NH4" in ks_constants:
        solutes["NH3"] = get_NH3(h, totals, ks_constants)
        solutes["NH4"] = get_NH4(h, totals, ks_constants)
    if "H4SiO4" in totals and "H4SiO4" in ks_constants:
        solutes["H3SiO4"] = get_H3SiO4(h, totals, ks_constants)
        solutes["H4SiO4"] = get_H4SiO4(h, totals, ks_constants)
    if "NO2" in totals and "HNO2" in ks_constants:
        solutes["HNO2"] = get_HNO2(h, totals, ks_constants)
        solutes["NO2"] = get_NO2(h, totals, ks_constants)
    if "F" in targets and "F" in totals:
        if "Ca" in totals and "CaF" in ks_constants:
            solutes["CaF"] = get_CaF(h, f, co3, po4, totals, ks_constants)
        if "Mg" in totals and "MgF" in ks_constants:
            solutes["MgF"] = get_MgF(h, f, co3, po4, totals, ks_constants)
    if "F" in totals and "HF" in ks_constants:
        solutes["HF"] = get_HF(h, f, ks_constants)
    if "CO3" in targets:
        if "CO2" in totals:
            if "Mg" in totals and "MgCO3" in ks_constants:
                solutes["MgCO3"] = get_MgCO3(h, f, co3, po4, totals, ks_constants)
            if "Ca" in totals and "CaCO3" in ks_constants:
                solutes["CaCO3"] = get_CaCO3(h, f, co3, po4, totals, ks_constants)
            if "Sr" in totals and "SrCO3" in ks_constants:
                solutes["Sr"] = get_Sr(co3, totals, ks_constants)
                solutes["SrCO3"] = get_SrCO3(co3, totals, ks_constants)
    if "CO2" in totals and "HCO3" in ks_constants:
        solutes["HCO3"] = get_HCO3(h, co3, ks_constants)
        if "H2CO3" in ks_constants:
            solutes["CO2"] = get_CO2(h, co3, ks_constants)
    if "PO4" in totals and "HPO4" in ks_constants:
        solutes["HPO4"] = get_HPO4(h, po4, ks_constants)
        if "H2PO4" in ks_constants:
            solutes["H2PO4"] = get_H2PO4(h, po4, ks_constants)
            if "H3PO4" in ks_constants:
                solutes["H3PO4"] = get_H3PO4(h, po4, ks_constants)
    if "PO4" in targets and "PO4" in totals:
        if "Mg" in totals:
            if "MgH2PO4" in ks_constants:
                solutes["MgH2PO4"] = get_MgH2PO4(h, f, co3, po4, totals, ks_constants)
            if "MgHPO4" in ks_constants:
                solutes["MgHPO4"] = get_MgHPO4(h, f, co3, po4, totals, ks_constants)
            if "MgPO4" in ks_constants:
                solutes["MgPO4"] = get_MgPO4(h, f, co3, po4, totals, ks_constants)
        if "Ca" in totals:
            if "CaH2PO4" in ks_constants:
                solutes["CaH2PO4"] = get_CaH2PO4(h, f, co3, po4, totals, ks_constants)
            if "CaHPO4" in ks_constants:
                solutes["CaHPO4"] = get_CaHPO4(h, f, co3, po4, totals, ks_constants)
            if "CaPO4" in ks_constants:
                solutes["CaPO4"] = get_CaPO4(h, f, co3, po4, totals, ks_constants)
    if "Mg" in totals:
        solutes["Mg"] = get_Mg(h, f, co3, po4, totals, ks_constants)
        if "MgOH" in ks_constants:
            solutes["MgOH"] = get_MgOH(h, f, co3, po4, totals, ks_constants)
    if "Ca" in totals:
        solutes["Ca"] = get_Ca(h, f, co3, po4, totals, ks_constants)
    return solutes
