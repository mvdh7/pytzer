from collections import OrderedDict
import jax


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


def get_HPO4(h, po4, ks_constants):
    k = ks_constants
    return po4 * h / k["HPO4"]


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
    return t["Ca"] / (
        1
        + k["CaF"] * f
        + k["CaCO3"] * co3
        + k["CaH2PO4"] * H2PO4
        + k["CaHPO4"] * HPO4
        + k["CaPO4"] * po4
    )


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
    return t["Mg"] / (
        1
        + k["MgOH"] * OH
        + k["MgF"] * f
        + k["MgCO3"] * co3
        + k["MgH2PO4"] * H2PO4
        + k["MgHPO4"] * HPO4
        + k["MgPO4"] * po4
    )


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


@jax.jit
def get_solutes__H(totals, ks_constants, pfixed):
    """Get solutes when pfixed contains only 'H'."""
    # Initial preparations
    fixed = OrderedDict((k, 10.0 ** -v) for k, v in pfixed.items())
    solutes = OrderedDict()
    solutes.update(totals)
    solutes["H"] = h = fixed["H"]
    # Deal with other absent pfixed variables
    if "F" in totals:
        assert "HF" in ks_constants
        solutes["F"] = f = get_F(h, totals, ks_constants)
        solutes["HF"] = get_HF(h, f, ks_constants)
    if "CO2" in totals:
        assert "H2CO3" in ks_constants
        assert "HCO3" in ks_constants
        solutes["CO3"] = co3 = get_CO3(h, totals, ks_constants)
        solutes["HCO3"] = get_HCO3(h, co3, ks_constants)
        solutes["CO2"] = get_CO2(h, co3, ks_constants)
    if "PO4" in totals:
        assert "H3PO4" in ks_constants
        assert "H2PO4" in ks_constants
        assert "HPO4" in ks_constants
        solutes["PO4"] = po4 = get_PO4(h, totals, ks_constants)
        solutes["HPO4"] = get_HPO4(h, po4, ks_constants)
        solutes["H2PO4"] = get_H2PO4(h, po4, ks_constants)
        solutes["H3PO4"] = get_H3PO4(h, po4, ks_constants)
    # Add other standard H-equilibria
    if "H2O" in ks_constants:
        solutes["OH"] = get_OH(h, ks_constants)
    if "SO4" in totals:
        assert "HSO4" in ks_constants
        solutes["HSO4"] = get_HSO4(h, totals, ks_constants)
        solutes["SO4"] = get_HSO4(h, totals, ks_constants)
    if "H2S" in totals:
        assert "H2S" in ks_constants
        solutes["H2S"] = get_H2S(h, totals, ks_constants)
        solutes["HS"] = get_HS(h, totals, ks_constants)
    if "BOH3" in totals:
        assert "BOH3" in ks_constants
        solutes["BOH3"] = get_BOH3(h, totals, ks_constants)
        solutes["BOH4"] = get_BOH4(h, totals, ks_constants)
    if "NH3" in totals:
        assert "NH4" in ks_constants
        solutes["NH3"] = get_NH3(h, totals, ks_constants)
        solutes["NH4"] = get_NH4(h, totals, ks_constants)
    if "H4SiO4" in totals:
        assert "H4SiO4" in ks_constants
        solutes["H4SiO4"] = get_H4SiO4(h, totals, ks_constants)
        solutes["H3SiO4"] = get_H3SiO4(h, totals, ks_constants)
    if "NO2" in totals:
        assert "HNO2" in ks_constants
        solutes["HNO2"] = get_HNO2(h, totals, ks_constants)
        solutes["NO2"] = get_NO2(h, totals, ks_constants)
    # what about MgOH......
    return solutes


@jax.jit
def get_solutes(totals, ks_constants, pfixed):
    fixed = OrderedDict((k, 10.0 ** -v) for k, v in pfixed.items())
    solutes = OrderedDict()
    solutes.update(totals)
    solutes["H"] = h = fixed["H"]  # H must always be fixed (for now)
    if "F" in fixed:
        solutes["F"] = f = fixed["F"]
    else:
        if "F" in totals:
            solutes["F"] = f = get_F(h, totals, ks_constants)
        else:
            f = 0.0
    if "CO3" in fixed:
        solutes["CO3"] = co3 = fixed["CO3"]
    else:
        solutes["CO3"] = co3 = get_CO3(h, totals, ks_constants)
    if "PO4" in fixed:
        solutes["PO4"] = po4 = fixed["PO4"]
    else:
        if "PO4" in totals:
            solutes["PO4"] = po4 = get_PO4(h, totals, ks_constants)
        else:
            po4 = 0.0
    if "H2O" in ks_constants:
        solutes["OH"] = get_OH(h, ks_constants)
    if "SO4" in totals and "HSO4" in ks_constants:
        solutes["HSO4"] = get_HSO4(h, totals, ks_constants)
        solutes["SO4"] = get_HSO4(h, totals, ks_constants)
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
    if "F" in fixed and "F" in totals:
        if "Ca" in totals and "CaF" in ks_constants:
            solutes["CaF"] = get_CaF(h, f, co3, po4, totals, ks_constants)
        if "Mg" in totals and "MgF" in ks_constants:
            solutes["MgF"] = get_MgF(h, f, co3, po4, totals, ks_constants)
    if "F" in totals and "HF" in ks_constants:
        solutes["HF"] = get_HF(h, f, ks_constants)
    if "CO3" in fixed:
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
    if "PO4" in fixed and "PO4" in totals:
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
