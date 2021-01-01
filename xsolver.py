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


def get_Sr(co3, totals, k_constants):
    t, k = totals, k_constants
    return t["Sr"] / (1 + k["SrCO3"] * co3)


def get_SrCO3(co3, totals, k_constants):
    t, k = totals, k_constants
    return t["Sr"] * k["SrCO3"] * co3 / (1 + k["SrCO3"] * co3)
