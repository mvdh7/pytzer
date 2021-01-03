import copy


def get_OH(h, k_constants):
    k = k_constants
    return k["H2O"] / h


def get_HF(h, f, k_constants):
    k = k_constants
    return f * h / k["HF"]


def get_HCO3(h, co3, k_constants):
    k = k_constants
    return co3 * h / k["C2"]


def get_CO2(h, co3, k_constants):
    k = k_constants
    return co3 * h ** 2 / (k["C1"] * k["C2"])


def get_HPO4(h, po4, k_constants):
    k = k_constants
    return po4 * h / k["P3"]


def get_H2PO4(h, po4, k_constants):
    k = k_constants
    return po4 * h ** 2 / (k["P2"] * k["P3"])


def get_H3PO4(h, po4, k_constants):
    k = k_constants
    return po4 * h ** 3 / (k["P1"] * k["P2"] * k["P3"])


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
    return k["B"] * t["B"] / (h + k["B"])


def get_BOH3(h, totals, k_constants):
    t, k = totals, k_constants
    return h * t["B"] / (h + k["B"])


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
    solutes["OH"] = get_OH(h, k_constants)
    solutes["HSO4"] = get_HSO4(h, totals, k_constants)
    solutes["SO4"] = get_HSO4(h, totals, k_constants)
    solutes["H2S"] = get_H2S(h, totals, k_constants)
    solutes["HS"] = get_HS(h, totals, k_constants)
    solutes["BOH3"] = get_BOH3(h, totals, k_constants)
    solutes["BOH4"] = get_BOH4(h, totals, k_constants)
    solutes["NH3"] = get_NH3(h, totals, k_constants)
    solutes["NH4"] = get_NH4(h, totals, k_constants)
    solutes["H3SiO4"] = get_H3SiO4(h, totals, k_constants)
    solutes["H4SiO4"] = get_H4SiO4(h, totals, k_constants)
    solutes["HNO2"] = get_HNO2(h, totals, k_constants)
    solutes["NO2"] = get_NO2(h, totals, k_constants)
    solutes["HF"] = get_HF(h, f, k_constants)
    solutes["F"] = f
    solutes["CO2"] = get_CO2(h, co3, k_constants)
    solutes["HCO3"] = get_HCO3(h, co3, k_constants)
    solutes["CO3"] = co3
    solutes["H3PO4"] = get_H3PO4(h, po4, k_constants)
    solutes["H2PO4"] = get_H2PO4(h, po4, k_constants)
    solutes["HPO4"] = get_HPO4(h, po4, k_constants)
    solutes["PO4"] = po4
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


def get_alkalinity(solutes):
    return (
        solutes["OH"]
        - solutes["H"]
        + solutes["MgOH"]
        - solutes["HF"]
        + solutes["HCO3"]
        + solutes["CO3"] * 2
        + solutes["HPO4"]
        + solutes["po4"] * 2
        - solutes["H3PO4"]
        + solutes["MgCO3"] * 2
        + solutes["CaCO3"] * 2
        + solutes["SrCO3"] * 2
        + solutes["MgHPO4"]
        + solutes["MgPO4"] * 2
        + solutes["CaHPO4"]
        + solutes["CaPO4"] * 2
        - solutes["HSO4"]
        + solutes["HS"]
        + solutes["BOH4"]
        + solutes["NH3"]
        + solutes["H3SiO4"]
        - solutes["HNO2"]
    )


def get_explicit_alkalinity(totals):
    return (
        t["Na"]
        + t["K"]
        - t["Cl"]
        - t["Br"]
        + t["Mg"] * 2
        + t["Ca"] * 2
        + t["Sr"] * 2
        - t["F"]
        - t["PO4"]
        - t["SO4"] * 2
        + t["NH3"]
        - t["NO2"]
    )


def get_total_F(solutes):
    return solutes["F"] + solutes["HF"] + solutes["MgF"] + solutes["CaF"]


def get_total_CO2(solutes):
    return (
        solutes["CO2"]
        + solutes["HCO3"]
        + solutes["CO3"]
        + solutes["CaCO3"]
        + solutes["MgCO3"]
        + solutes["SrCO3"]
    )


def get_total_PO4(solutes):
    return (
        solutes["PO4"]
        + solutes["HPO4"]
        + solutes["H2PO4"]
        + solutes["H3PO4"]
        + solutes["MgPO4"]
        + solutes["MgHPO4"]
        + solutes["MgH2PO4"]
        + solutes["CaPO4"]
        + solutes["CaHPO4"]
        + solutes["CaH2PO4"]
    )


def get_targets(solutes):
    return (
        get_alkalinity(solutes),
        get_total_F(solutes),
        get_total_CO2(solutes),
        get_total_PO4(solutes),
    )
