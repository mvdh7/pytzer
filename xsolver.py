import jax, copy
from jax import numpy as np, lax
import pytzer as pz

components = pz.solvers.components


def get_ls(h, f, co3, po4, totals, k_constants):
    # Get all components
    OH = components.get_OH(h, k_constants)
    # Mg = components.get_Mg(h, f, co3, po4, totals, k_constants)
    # Ca = components.get_Ca(h, f, co3, po4, totals, k_constants)
    # Sr = components.get_Sr(co3, totals, k_constants)
    MgOH = components.get_MgOH(h, f, co3, po4, totals, k_constants)
    HF = components.get_HF(h, f, k_constants)
    CO2 = components.get_CO2(h, co3, k_constants)
    HCO3 = components.get_HCO3(h, co3, k_constants)
    HPO4 = components.get_HPO4(h, po4, k_constants)
    H2PO4 = components.get_H2PO4(h, po4, k_constants)
    H3PO4 = components.get_H3PO4(h, po4, k_constants)
    MgCO3 = components.get_MgCO3(h, f, co3, po4, totals, k_constants)
    CaCO3 = components.get_CaCO3(h, f, co3, po4, totals, k_constants)
    SrCO3 = components.get_SrCO3(co3, totals, k_constants)
    MgH2PO4 = components.get_MgH2PO4(h, f, co3, po4, totals, k_constants)
    MgHPO4 = components.get_MgHPO4(h, f, co3, po4, totals, k_constants)
    MgPO4 = components.get_MgPO4(h, f, co3, po4, totals, k_constants)
    CaH2PO4 = components.get_CaH2PO4(h, f, co3, po4, totals, k_constants)
    CaHPO4 = components.get_CaHPO4(h, f, co3, po4, totals, k_constants)
    CaPO4 = components.get_CaPO4(h, f, co3, po4, totals, k_constants)
    HSO4 = components.get_HSO4(h, totals, k_constants)
    HS = components.get_HS(h, totals, k_constants)
    # H2S = components.get_H2S(h, totals, k_constants)
    BOH4 = components.get_BOH4(h, totals, k_constants)
    NH3 = components.get_NH3(h, totals, k_constants)
    # H3SiO4 = components.get_H3SiO4(h, totals, k_constants)
    # HNO2 = components.get_HNO2(h, totals, k_constants)
    CaF = components.get_CaF(h, f, co3, po4, totals, k_constants)
    MgF = components.get_MgF(h, f, co3, po4, totals, k_constants)
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
        - HSO4
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
        po4 + HPO4 + H2PO4 + H3PO4 + MgPO4 + MgHPO4 + MgH2PO4 + CaPO4 + CaHPO4 + CaH2PO4
    )
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
        - t["PO4"]
        - 2 * t["SO4"]
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


totals = pz.prepare.salinity_to_totals_MFWM08()
totals["NH3"] = 1e-6 * 0
totals["NO2"] = 2e-6 * 0
totals["H2S"] = 3e-6 * 0
totals["PO4"] = 5e-6
# es = e_H, e_F, e_CO3, e_PO4 = get_es(totals)
es = e_H, t_F, t_CO3, t_PO4 = get_es(totals)

k_constants = pz.dissociation.assemble()
pk_constants = {k: -np.log10(v) for k, v in k_constants.items()}

molalities_0 = np.array(
    [
        1e-8,
        totals["F"] / 2,
        totals["CO2"] / 10,
        totals["PO4"] / 2,
    ]
)
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

test2 = pz.solvers.stoichiometric.solver_func(solver_0, totals, k_constants, *es)
jtest2 = pz.solvers.stoichiometric.solver_jac(solver_0, totals, k_constants, *es)

solver_x = np.array(copy.deepcopy(solver_0))
solver_x2 = np.array(copy.deepcopy(solver_0))

#%%


def solver_step(solver_x):
    target = -solver_func(solver_x, es, totals, k_constants)
    jac = solver_jac(solver_x, es, totals, k_constants)
    x_diff = np.linalg.solve(jac, target)
    x_diff = np.where(x_diff > 1, 1, x_diff)
    x_diff = np.where(x_diff < -1, -1, x_diff)
    solver_x = solver_x + x_diff
    return solver_x


@jax.jit
def solve_now(solver_x, es, totals, k_constants):

    tol_alkalinity = 1e-9
    tol_total_F = 1e-9
    tol_total_CO2 = 1e-9
    tol_total_PO4 = 1e-9
    tols = np.array([tol_alkalinity, tol_total_F, tol_total_CO2, tol_total_PO4])

    def cond(solver_x):
        target = solver_func(solver_x, es, totals, k_constants)
        return np.any(np.abs(target) > tols)

    def body(solver_x):
        target = -solver_func(solver_x, es, totals, k_constants)
        jac = solver_jac(solver_x, es, totals, k_constants)
        x_diff = np.linalg.solve(jac, target)
        x_diff = np.where(x_diff > 1, 1, x_diff)
        x_diff = np.where(x_diff < -1, -1, x_diff)
        return solver_x + x_diff

    return lax.while_loop(cond, body, solver_x)


#%%
solver_x = solve_now(solver_x, es, totals, k_constants)
solver_x2 = pz.solvers.stoichiometric.solve_now(solver_x2, totals, k_constants, *es)

# solver_x = solver_step(solver_x)
print(solver_x)
print(solver_to_molalities(solver_x) * 1e6)

# print()
# print(jac)
# print()
# print(x_diff)
# print(solver_x)
# print(solver_to_molalities(solver_x) * 1e6)

#%% Check validity of solution
h, f, co3, po4 = solver_to_molalities(solver_x)

solved = s = {}
s["OH"] = components.get_OH(h, k_constants)
s["HF"] = components.get_HF(h, f, k_constants)
s["MgF"] = components.get_MgF(h, f, co3, po4, totals, k_constants)
s["CaF"] = components.get_CaF(h, f, co3, po4, totals, k_constants)
s["HCO3"] = components.get_HCO3(h, co3, k_constants)
s["CO2"] = components.get_CO2(h, co3, k_constants)
s["MgCO3"] = components.get_MgCO3(h, f, co3, po4, totals, k_constants)
s["CaCO3"] = components.get_CaCO3(h, f, co3, po4, totals, k_constants)
s["SrCO3"] = components.get_SrCO3(co3, totals, k_constants)


print("TF vs F + HF + MgF + CaF:")
print(f + s["HF"] + s["MgF"] + s["CaF"])
print(totals["F"])

print("TCO2:")
print(co3 + s["CO2"] + s["HCO3"] + s["MgCO3"] + s["CaCO3"] + s["SrCO3"])
print(totals["CO2"])

print("TP:")
print(
    po4
    + components.get_HPO4(h, po4, k_constants)
    + components.get_H2PO4(h, po4, k_constants)
    + components.get_H3PO4(h, po4, k_constants)
    + components.get_MgPO4(h, f, co3, po4, totals, k_constants)
    + components.get_MgHPO4(h, f, co3, po4, totals, k_constants)
    + components.get_MgH2PO4(h, f, co3, po4, totals, k_constants)
    + components.get_CaPO4(h, f, co3, po4, totals, k_constants)
    + components.get_CaHPO4(h, f, co3, po4, totals, k_constants)
    + components.get_CaH2PO4(h, f, co3, po4, totals, k_constants)
)
print(totals["PO4"])

print("TMg:")
print(
    components.get_Mg(h, f, co3, po4, totals, k_constants)
    + components.get_MgF(h, f, co3, po4, totals, k_constants)
    + components.get_MgOH(h, f, co3, po4, totals, k_constants)
    + components.get_MgPO4(h, f, co3, po4, totals, k_constants)
    + components.get_MgHPO4(h, f, co3, po4, totals, k_constants)
    + components.get_MgH2PO4(h, f, co3, po4, totals, k_constants)
    + components.get_MgCO3(h, f, co3, po4, totals, k_constants)
)
print(totals["Mg"])

print("TCa:")
print(
    components.get_Ca(h, f, co3, po4, totals, k_constants)
    + components.get_CaF(h, f, co3, po4, totals, k_constants)
    + components.get_CaPO4(h, f, co3, po4, totals, k_constants)
    + components.get_CaHPO4(h, f, co3, po4, totals, k_constants)
    + components.get_CaH2PO4(h, f, co3, po4, totals, k_constants)
    + components.get_CaCO3(h, f, co3, po4, totals, k_constants)
)
print(totals["Ca"])

l_H, l_F, l_CO3, l_PO4 = get_ls(*solver_to_molalities(solver_x), totals, k_constants)
