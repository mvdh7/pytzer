import jax, copy
from jax import numpy as np


def lh(h_f_co3, totals, k_constants):
    h, f, co3 = h_f_co3
    _, t_Mg, _, _ = totals
    k_H2O, k_MgOH, k_HF, k_MgF, _, _, k_C2, k_MgCO3, _ = k_constants
    Mg = t_Mg / (1 + k_MgOH / h + k_MgF * f + k_MgCO3 * co3)
    HCO3 = h * co3 / k_C2
    return k_H2O / h - h + k_MgOH * Mg / h - h * f / k_HF + HCO3 + 2 * co3


def lf(h_f_co3, totals, k_constants):
    h, f, co3 = h_f_co3
    _, t_Mg, t_Ca, _ = totals
    k_H2O, k_MgOH, k_HF, k_MgF, k_CaF, _, k_C2, k_MgCO3, k_CaCO3 = k_constants
    Mg = t_Mg / (1 + k_MgF * f + k_MgOH / h + k_MgCO3 * co3)
    Ca = t_Ca / (1 + k_CaF * f + k_CaCO3 * co3)
    HCO3 = h * co3 / k_C2
    return h + Mg + Ca - f - k_H2O / h - HCO3 - 2 * co3


def lc(h_f_co3, totals, k_constants):
    h, f, co3 = h_f_co3
    _, t_Mg, t_Ca, _ = totals
    k_H2O, k_MgOH, k_HF, k_MgF, k_CaF, k_C1, k_C2, k_MgCO3, k_CaCO3 = k_constants
    Mg = t_Mg / (1 + k_MgF * f + k_MgOH / h + k_MgCO3 * co3)
    Ca = t_Ca / (1 + k_CaF * f + k_CaCO3 * co3)
    OH = k_H2O / h
    CO2 = h ** 2 * co3 / (k_C1 * k_C2)
    return Mg + Ca + h - co3 - OH - f + CO2


def eh(totals, Na=0, Cl=0):
    t_F, t_Mg, t_Ca, _ = totals
    return Na - Cl + 2 * t_Mg + 2 * t_Ca - t_F


def ef(totals, Na=0, Cl=0):
    _, t_Mg, t_Ca, _ = totals
    return Cl - Na - t_Mg - t_Ca


def ec(totals, Na=0, Cl=0):
    _, t_Mg, t_Ca, t_CO2 = totals
    return Cl - Na - t_Mg - t_Ca + t_CO2


def eh_ef(totals):
    return eh(totals) + ef(totals)


def lh_lf(h_f, totals, k_constants):
    return lh(h_f, totals, k_constants) + lf(h_f, totals, k_constants)


def total_F(h_f, totals, k_constants):
    h, f = h_f
    _, t_Mg, t_Ca, _ = totals
    _, k_MgOH, k_HF, k_MgF, k_CaF, _, _ = k_constants
    HF = h * f / k_HF
    Mg = t_Mg / (1 + k_MgOH / h + k_MgF * f)
    MgF = k_MgF * Mg * f
    Ca = t_Ca / (1 + k_CaF * f)
    CaF = k_CaF * Ca * f
    return f + HF + MgF + CaF


@jax.jit
def h_f_co3_funcs(ph_pf_pco3, totals, k_constants, Na=2250e-6, Cl=0.1399):
    h_f_co3 = 10.0 ** -ph_pf_pco3
    # t_F = totals[0]
    return np.array(
        [
            eh(totals, Na=Na, Cl=Cl) - lh(h_f_co3, totals, k_constants),
            ef(totals, Na=Na, Cl=Cl) - lf(h_f_co3, totals, k_constants),
            ec(totals, Na=Na, Cl=Cl) - lc(h_f_co3, totals, k_constants),
        ]
    )


h_f_co3_jac = jax.jit(jax.jacfwd(h_f_co3_funcs))

totals = t_F, t_Mg, t_Ca, t_CO2 = 0.0001, 0.06, 0.01, 2000e-6
pk_constants = (
    pk_H2O,
    pk_MgOH,
    pk_HF,
    pk_MgF,
    pk_CaF,
    pk_C1,
    pk_C2,
    pk_MgCO3,
    pk_CaCO3,
) = (
    14,
    10,
    3,
    -1.8,
    -1.3,
    5.8,
    8.9,
    1.028 + 298.15 * 0.0066154,
    1.178 + 298.15 * 0.0066154,
)
k_constants = [10.0 ** -pk for pk in pk_constants]

ph_pf_pco3_i = np.array([8, -np.log10(t_F / 2), -np.log10(t_CO2 / 10)])
ph_pf_pco3 = copy.deepcopy(ph_pf_pco3_i)

#%% This is the iterative solver, following
# https://en.wikipedia.org/wiki/Newton%27s_method#k_variables,_k_functions
print(ph_pf_pco3)
target = -h_f_co3_funcs(ph_pf_pco3, totals, k_constants)
jac = h_f_co3_jac(ph_pf_pco3, totals, k_constants)
x_diff = np.linalg.solve(jac, target)
ph_pf_pco3 = ph_pf_pco3 + x_diff
print(target)
print(jac)
print(x_diff)
print(ph_pf_pco3)
h, f, co3 = h_f_co3 = 10.0 ** -ph_pf_pco3
print(h_f_co3 * 1e6)
