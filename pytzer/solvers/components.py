import copy


def get_OH(h, k_constants):
    k = k_constants
    return k["H2O"] / h


def get_HF(h, f, k_constants):
    k = k_constants
    return f * h / k["HF"]


def get_HCO3(h, co3, k_constants):
    k = k_constants
    return co3 * h / k["HCO3"]


def get_CO2(h, co3, k_constants):
    k = k_constants
    return co3 * h ** 2 / (k["H2CO3"] * k["HCO3"])


def get_HPO4(h, po4, k_constants):
    k = k_constants
    return po4 * h / k["HPO4"]


def get_H2PO4(h, po4, k_constants):
    k = k_constants
    return po4 * h ** 2 / (k["H2PO4"] * k["HPO4"])


def get_H3PO4(h, po4, k_constants):
    k = k_constants
    return po4 * h ** 3 / (k["H3PO4"] * k["H2PO4"] * k["HPO4"])


def get_SO4(h, totals, k_constants):
    t, k = totals, k_constants
    return k["HSO4"] * t["SO4"] / (h + k["HSO4"])


def get_HSO4(h, totals, k_constants):
    t, k = totals, k_constants
    return h * t["SO4"] / (h + k["HSO4"])


def get_NO2(h, totals, k_constants):
    t, k = totals, k_constants
    return k["HNO2"] * t["NO2"] / (h + k["HNO2"])


def get_HNO2(h, totals, k_constants):
    t, k = totals, k_constants
    return h * t["NO2"] / (h + k["HNO2"])


def get_NH3(h, totals, k_constants):
    t, k = totals, k_constants
    return k["NH4"] * t["NH3"] / (h + k["NH4"])


def get_NH4(h, totals, k_constants):
    t, k = totals, k_constants
    return h * t["NH3"] / (h + k["NH4"])


def get_HS(h, totals, k_constants):
    t, k = totals, k_constants
    return k["H2S"] * t["H2S"] / (h + k["H2S"])


def get_H2S(h, totals, k_constants):
    t, k = totals, k_constants
    return h * t["H2S"] / (h + k["H2S"])


def get_H3SiO4(h, totals, k_constants):
    t, k = totals, k_constants
    return k["H4SiO4"] * t["SiO4"] / (h + k["H4SiO4"])


def get_H4SiO4(h, totals, k_constants):
    t, k = totals, k_constants
    return h * t["SiO4"] / (h + k["H4SiO4"])


def get_BOH4(h, totals, k_constants):
    t, k = totals, k_constants
    return k["BOH3"] * t["B"] / (h + k["BOH3"])


def get_BOH3(h, totals, k_constants):
    t, k = totals, k_constants
    return h * t["B"] / (h + k["BOH3"])


def get_Ca(h, f, co3, po4, totals, k_constants):
    H2PO4 = get_H2PO4(h, po4, k_constants)
    HPO4 = get_HPO4(h, po4, k_constants)
    t, k = totals, k_constants
    return t["Ca"] / (
        1
        + k["CaF"] * f
        + k["CaCO3"] * co3
        + k["CaH2PO4"] * H2PO4
        + k["CaHPO4"] * HPO4
        + k["CaPO4"] * po4
    )


def get_CaF(h, f, co3, po4, totals, k_constants):
    Ca = get_Ca(h, f, co3, po4, totals, k_constants)
    k = k_constants
    return k["CaF"] * Ca * f


def get_CaCO3(h, f, co3, po4, totals, k_constants):
    Ca = get_Ca(h, f, co3, po4, totals, k_constants)
    k = k_constants
    return k["CaCO3"] * Ca * co3


def get_CaH2PO4(h, f, co3, po4, totals, k_constants):
    Ca = get_Ca(h, f, co3, po4, totals, k_constants)
    H2PO4 = get_H2PO4(h, po4, k_constants)
    k = k_constants
    return k["CaH2PO4"] * Ca * H2PO4


def get_CaHPO4(h, f, co3, po4, totals, k_constants):
    Ca = get_Ca(h, f, co3, po4, totals, k_constants)
    HPO4 = get_HPO4(h, po4, k_constants)
    k = k_constants
    return k["CaHPO4"] * Ca * HPO4


def get_CaPO4(h, f, co3, po4, totals, k_constants):
    Ca = get_Ca(h, f, co3, po4, totals, k_constants)
    k = k_constants
    return k["CaPO4"] * Ca * po4


def get_Mg(h, f, co3, po4, totals, k_constants):
    OH = get_OH(h, k_constants)
    H2PO4 = get_H2PO4(h, po4, k_constants)
    HPO4 = get_HPO4(h, po4, k_constants)
    t, k = totals, k_constants
    return t["Mg"] / (
        1
        + k["MgOH"] * OH
        + k["MgF"] * f
        + k["MgCO3"] * co3
        + k["MgH2PO4"] * H2PO4
        + k["MgHPO4"] * HPO4
        + k["MgPO4"] * po4
    )


def get_MgOH(h, f, co3, po4, totals, k_constants):
    Mg = get_Mg(h, f, co3, po4, totals, k_constants)
    OH = get_OH(h, k_constants)
    k = k_constants
    return k["MgOH"] * Mg * OH


def get_MgF(h, f, co3, po4, totals, k_constants):
    Mg = get_Mg(h, f, co3, po4, totals, k_constants)
    k = k_constants
    return k["MgF"] * Mg * f


def get_MgCO3(h, f, co3, po4, totals, k_constants):
    Mg = get_Mg(h, f, co3, po4, totals, k_constants)
    k = k_constants
    return k["MgCO3"] * Mg * co3


def get_MgH2PO4(h, f, co3, po4, totals, k_constants):
    Mg = get_Mg(h, f, co3, po4, totals, k_constants)
    H2PO4 = get_H2PO4(h, po4, k_constants)
    k = k_constants
    return k["MgH2PO4"] * Mg * H2PO4


def get_MgHPO4(h, f, co3, po4, totals, k_constants):
    Mg = get_Mg(h, f, co3, po4, totals, k_constants)
    HPO4 = get_HPO4(h, po4, k_constants)
    k = k_constants
    return k["MgHPO4"] * Mg * HPO4


def get_MgPO4(h, f, co3, po4, totals, k_constants):
    Mg = get_Mg(h, f, co3, po4, totals, k_constants)
    k = k_constants
    return k["MgPO4"] * Mg * po4


def get_Sr(co3, totals, k_constants):
    t, k = totals, k_constants
    return t["Sr"] / (1 + k["SrCO3"] * co3)


def get_SrCO3(co3, totals, k_constants):
    t, k = totals, k_constants
    return t["Sr"] * k["SrCO3"] * co3 / (1 + k["SrCO3"] * co3)


def get_all(h, f, co3, po4, totals, k_constants):
    solutes = copy.deepcopy(totals)
    solutes["H"] = h
    if "H2O" in k_constants:
        solutes["OH"] = get_OH(h, k_constants)
    if "HSO4" in k_constants:
        solutes["HSO4"] = get_HSO4(h, totals, k_constants)
        solutes["SO4"] = get_HSO4(h, totals, k_constants)
    if "H2S" in k_constants:
        solutes["H2S"] = get_H2S(h, totals, k_constants)
        solutes["HS"] = get_HS(h, totals, k_constants)
    if "BOH3" in k_constants:
        solutes["BOH3"] = get_BOH3(h, totals, k_constants)
        solutes["BOH4"] = get_BOH4(h, totals, k_constants)
    if "NH4" in k_constants:
        solutes["NH3"] = get_NH3(h, totals, k_constants)
        solutes["NH4"] = get_NH4(h, totals, k_constants)
    if "H4SiO4" in k_constants:
        solutes["H3SiO4"] = get_H3SiO4(h, totals, k_constants)
        solutes["H4SiO4"] = get_H4SiO4(h, totals, k_constants)
    if "HNO2" in k_constants:
        solutes["HNO2"] = get_HNO2(h, totals, k_constants)
        solutes["NO2"] = get_NO2(h, totals, k_constants)
    solutes["F"] = f
    if "HF" in k_constants:
        solutes["HF"] = get_HF(h, f, k_constants)
    solutes["CO3"] = co3
    if "H2CO3" in k_constants and "HCO3" in k_constants:
        solutes["CO2"] = get_CO2(h, co3, k_constants)
        solutes["HCO3"] = get_HCO3(h, co3, k_constants)
    solutes["PO4"] = po4
    if "H3PO4" in k_constants and "H2PO4" in k_constants and "HPO4" in k_constants:
        solutes["H3PO4"] = get_H3PO4(h, po4, k_constants)
        solutes["H2PO4"] = get_H2PO4(h, po4, k_constants)
        solutes["HPO4"] = get_HPO4(h, po4, k_constants)
    solutes["Mg"] = get_Mg(h, f, co3, po4, totals, k_constants)
    solutes["MgOH"] = get_MgOH(h, f, co3, po4, totals, k_constants)
    solutes["MgF"] = get_MgF(h, f, co3, po4, totals, k_constants)
    solutes["MgCO3"] = get_MgCO3(h, f, co3, po4, totals, k_constants)
    solutes["MgH2PO4"] = get_MgH2PO4(h, f, co3, po4, totals, k_constants)
    solutes["MgHPO4"] = get_MgHPO4(h, f, co3, po4, totals, k_constants)
    solutes["MgPO4"] = get_MgPO4(h, f, co3, po4, totals, k_constants)
    solutes["Ca"] = get_Ca(h, f, co3, po4, totals, k_constants)
    solutes["CaF"] = get_CaF(h, f, co3, po4, totals, k_constants)
    solutes["CaCO3"] = get_CaCO3(h, f, co3, po4, totals, k_constants)
    solutes["CaH2PO4"] = get_CaH2PO4(h, f, co3, po4, totals, k_constants)
    solutes["CaHPO4"] = get_CaHPO4(h, f, co3, po4, totals, k_constants)
    solutes["CaPO4"] = get_CaPO4(h, f, co3, po4, totals, k_constants)
    solutes["Sr"] = get_Sr(co3, totals, k_constants)
    solutes["SrCO3"] = get_SrCO3(co3, totals, k_constants)
    return solutes
