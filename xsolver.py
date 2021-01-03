import jax, copy
from jax import numpy as np


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


def get_ls(h, f, co3, po4, totals, k_constants):
    # Get all components
    OH = get_OH(h, k_constants)
    # Mg = get_Mg(h, f, co3, po4, totals, k_constants)
    # Ca = get_Ca(h, f, co3, po4, totals, k_constants)
    # Sr = get_Sr(co3, totals, k_constants)
    MgOH = get_MgOH(h, f, co3, po4, totals, k_constants)
    HF = get_HF(h, f, k_constants)
    CO2 = get_CO2(h, co3, k_constants)
    HCO3 = get_HCO3(h, co3, k_constants)
    HPO4 = get_HPO4(h, po4, k_constants)
    H2PO4 = get_H2PO4(h, po4, k_constants)
    H3PO4 = get_H3PO4(h, po4, k_constants)
    MgCO3 = get_MgCO3(h, f, co3, po4, totals, k_constants)
    CaCO3 = get_CaCO3(h, f, co3, po4, totals, k_constants)
    SrCO3 = get_SrCO3(co3, totals, k_constants)
    MgH2PO4 = get_MgH2PO4(h, f, co3, po4, totals, k_constants)
    MgHPO4 = get_MgHPO4(h, f, co3, po4, totals, k_constants)
    MgPO4 = get_MgPO4(h, f, co3, po4, totals, k_constants)
    CaH2PO4 = get_CaH2PO4(h, f, co3, po4, totals, k_constants)
    CaHPO4 = get_CaHPO4(h, f, co3, po4, totals, k_constants)
    CaPO4 = get_CaPO4(h, f, co3, po4, totals, k_constants)
    SO4 = get_SO4(h, totals, k_constants)
    HS = get_HS(h, totals, k_constants)
    # H2S = get_H2S(h, totals, k_constants)
    BOH4 = get_BOH4(h, totals, k_constants)
    NH3 = get_NH3(h, totals, k_constants)
    # H3SiO4 = get_H3SiO4(h, totals, k_constants)
    # HNO2 = get_HNO2(h, totals, k_constants)
    CaF = get_CaF(h, f, co3, po4, totals, k_constants)
    MgF = get_MgF(h, f, co3, po4, totals, k_constants)
    # Alkalinity
    l_H = (
        OH
        - h
        + MgOH
        - HF
        + HCO3
        + 2 * co3
        + HPO4
        + 2 * po4
        - H3PO4
        + 2 * MgCO3
        + 2 * CaCO3
        + 2 * SrCO3
        + MgHPO4
        + 2 * MgPO4
        + CaHPO4
        + 2 * CaPO4
        + SO4
        + HS
        + BOH4
        + NH3
        # + H3SiO4
        # - HNO2
    )
    # Total fluorine
    l_F = f + HF + MgF + CaF
    # Total CO2
    l_CO3 = co3 + CO2 + HCO3 + CaCO3 + MgCO3 + SrCO3
    # Total PO4
    l_PO4 = (
        po4 + HPO4 + H2PO4 + H3PO4
        + MgPO4 + MgHPO4 + MgH2PO4
        + CaPO4 + CaHPO4 + CaH2PO4)
    # # Fluoralinity
    # l_F = (
    #     h
    #     + Mg
    #     + Ca
    #     - f
    #     - OH
    #     + CO2
    #     - co3
    #     - SrCO3
    #     + 2 * H3PO4
    #     + H2PO4
    #     - po4
    #     + 2 * MgH2PO4
    #     + MgHPO4
    #     + 2 * CaH2PO4
    #     + CaHPO4
    #     - SO4
    #     # + H2S
    #     # - BOH4
    #     # - NH3
    #     # - H3SiO4
    #     # + HNO2
    # )
    # # Carbalinity
    # l_CO3 = (
    #     Mg
    #     + Ca
    #     + Sr
    #     + h
    #     - OH
    #     + CO2
    #     - co3
    #     + CaF
    #     + MgF
    #     + HF
    #     + 2 * H3PO4
    #     + H2PO4
    #     - po4
    #     + 2 * MgH2PO4
    #     + MgHPO4
    #     + 2 * CaH2PO4
    #     + CaHPO4
    #     - SO4
    #     # + H2S
    #     # - BOH4
    #     # - NH3
    #     # - H3SiO4
    #     # + HNO2
    # )
    # # Phosphalinity
    # l_PO4 = (
    #     h
    #     - OH
    #     + 2 * H3PO4
    #     + H2PO4
    #     - po4
    #     + CO2
    #     - co3
    #     - SrCO3
    #     - CaCO3
    #     - MgCO3
    #     + HF
    #     - MgPO4
    #     + MgH2PO4
    #     - CaPO4
    #     + CaH2PO4
    #     - MgOH
    #     - SO4
    #     # + H2S
    #     # - BOH4
    #     # - NH3
    #     # - H3SiO4
    #     # + HNO2
    # )
    return l_H, l_F, l_CO3, l_PO4


def get_es(totals):
    t = totals
    # Alkalinity
    e_H = (
        t["Na"]
        + t["K"]
        - t["Cl"]
        - t["Br"]
        + 2 * t["Mg"]
        + 2 * t["Ca"]
        + 2 * t["Sr"]
        - t["F"]
        # - t["PO4"]
        - t["SO4"]
        + t["NH3"]
        # - t["NO2"]
    )
    # Total fluorine
    t_F = t["F"]
    # Total CO2
    t_CO3 = t["CO2"]
    # Total PO4
    t_PO4 = t["PO4"]
    # # Fluoralinity
    # e_F = (
    #     t["Cl"]
    #     + t["Br"]
    #     - t["Na"]
    #     - t["K"]
    #     + t["CO2"]
    #     - t["Ca"]
    #     - t["Mg"]
    #     - 2 * t["Sr"]
    #     + 2 * t["PO4"]
    #     + t["SO4"]
    #     # + t["H2S"]
    #     # - t["NH3"]
    #     # + t["NO2"]
    # )
    # # Carbalinity
    # e_CO3 = (
    #     t["Cl"]
    #     + t["Br"]
    #     - t["Na"]
    #     - t["K"]
    #     + t["CO2"]
    #     - t["Ca"]
    #     - t["Mg"]
    #     - t["Sr"]
    #     + t["F"]
    #     + 2 * t["PO4"]
    #     + t["SO4"]
    #     # + t["H2S"]
    #     # - t["NH3"]
    #     # + t["NO2"]
    # )
    # # Phosphalinity
    # e_PO4 = (
    #     t["Cl"]
    #     + t["Br"]
    #     - t["Na"]
    #     - t["K"]
    #     + t["CO2"]
    #     + t["F"]
    #     - 2 * t["Sr"]
    #     - 2 * t["Mg"]
    #     - 2 * t["Ca"]
    #     + 2 * t["PO4"]
    #     + t["SO4"]
    #     # + t["H2S"]
    #     # - t["NH3"]
    #     # + t["NO2"]
    # )
    # return e_H, e_F, e_CO3, e_PO4
    return e_H, t_F, t_CO3, t_PO4


# def f_arctan(x):
#     return (np.arctan(x) + np.pi / 2) / np.pi


def solver_to_molalities(x_solver):
    # pH, fF, fCO3, fPO4 = x_solver
    # t = totals
    # return [
    #     10.0 ** -pH,
    #     f_arctan(fF) * t["F"],
    #     f_arctan(fCO3) * t["CO2"],
    #     f_arctan(fPO4) * t["PO4"],
    # ]
    return 10.0 ** -x_solver


#%%
import pytzer as pz

totals = pz.prepare.salinity_to_totals_MFWM08()
totals["Cl"] += 0.029246
totals["NH3"] = 1e-6
totals["NO2"] = 2e-6
totals["H2S"] = 3e-6
totals["PO4"] = 5e-6
# es = e_H, e_F, e_CO3, e_PO4 = get_es(totals)
es = e_H, t_F, t_CO3, t_PO4 = get_es(totals)

k_constants = pz.dissociation.assemble()
pk_constants = {k: -np.log10(v) for k, v in k_constants.items()}

molalities_0 = np.array([
    1e-8,
    totals["F"] / 2,
    totals["CO2"] / 10,
    totals["PO4"] / 2,
])
solver_0 = -np.log10(molalities_0)

l_H, l_F, l_CO3, l_PO4 = get_ls(*molalities_0, totals, k_constants)


def solver_func(solver_x, es, totals, k_constants):
    molalities = solver_to_molalities(solver_x)
    # e_H, e_F, e_CO3, e_PO4 = es
    # l_H, l_F, l_CO3, l_PO4 = get_ls(*molalities, totals, k_constants)
    # return np.array([e_H - l_H, e_F - l_F, e_CO3 - l_CO3, e_PO4 - l_PO4])
    e_H, t_F, t_CO3, t_PO4 = es
    l_H, l_F, l_CO3, l_PO4 = get_ls(*molalities, totals, k_constants)
    return np.array([e_H - l_H, t_F - l_F, t_CO3 - l_CO3, t_PO4 - l_PO4])


solver_jac = jax.jit(jax.jacfwd(solver_func))

test = solver_func(solver_0, es, totals, k_constants)
jtest = solver_jac(solver_0, es, totals, k_constants)

solver_x = np.array(copy.deepcopy(solver_0))

#%%
print(solver_x)
print(solver_to_molalities(solver_x))
target = -solver_func(solver_x, es, totals, k_constants)
jac = solver_jac(solver_x, es, totals, k_constants)
x_diff = np.linalg.solve(jac, target)
x_diff = np.where(x_diff > 1, 1, x_diff)
x_diff = np.where(x_diff < -1, -1, x_diff)
solver_x = solver_x + x_diff
print()
print(jac)
print()
print(x_diff)
print(solver_x)
# print(solver_to_molalities(solver_x) * 1e6)

# Check validity of solution
h, f, co3, po4 = solver_to_molalities(solver_x)

solved = s = {}
s["OH"] = get_OH(h, k_constants)
s["HF"] = get_HF(h, f, k_constants)
s["MgF"] = get_MgF(h, f, co3, po4, totals, k_constants)
s["CaF"] = get_CaF(h, f, co3, po4, totals, k_constants)
s["HCO3"] = get_HCO3(h, co3, k_constants)
s["CO2"] = get_CO2(h, co3, k_constants)
s["MgCO3"] = get_MgCO3(h, f, co3, po4, totals, k_constants)
s["CaCO3"] = get_CaCO3(h, f, co3, po4, totals, k_constants)
s["SrCO3"] = get_SrCO3(co3, totals, k_constants)


print("HF vs k_HF:")
print(h * f / s["HF"])
print(k_constants["HF"])

print("TF vs F + HF + MgF + CaF:")
print(f + s["HF"] + s["MgF"] + s["CaF"])
print(totals["F"])

print("K1:")
print(h * s["HCO3"] / s["CO2"])
print(k_constants["C1"])

print("K2:")
print(h * co3 / s["HCO3"])
print(k_constants["C2"])

print("TCO2:")
print(co3 + s["CO2"] + s["HCO3"] + s["MgCO3"] + s["CaCO3"] + s["SrCO3"])
print(totals["CO2"])

print("TP:")
print(
    po4
    + get_HPO4(h, po4, k_constants)
    + get_H2PO4(h, po4, k_constants)
    + get_H3PO4(h, po4, k_constants)
    + get_MgPO4(h, f, co3, po4, totals, k_constants)
    + get_MgHPO4(h, f, co3, po4, totals, k_constants)
    + get_MgH2PO4(h, f, co3, po4, totals, k_constants)
    + get_CaPO4(h, f, co3, po4, totals, k_constants)
    + get_CaHPO4(h, f, co3, po4, totals, k_constants)
    + get_CaH2PO4(h, f, co3, po4, totals, k_constants)
)
print(totals['PO4'])

#%%
# from matplotlib import pyplot as plt
# from scipy.special import erf

# fx = np.linspace(-5, 5, 1000)
# fy = 3 - np.exp(-fx)
# fz = f_arctan(fx)  # this is the best one
# fe = (erf(fx / 2) + 1) / 2

# ty = 10.0 ** -fy

# fig, ax = plt.subplots(dpi=300)
# ax.plot(fx, fy)
# ax.plot(fx, ty)
# ax.plot(fx, fz)
# ax.plot(fx, fe)
# ax.grid(alpha=0.5)
# ax.set_ylim([-4, 4])
