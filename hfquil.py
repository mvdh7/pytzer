import jax, copy
from jax import numpy as np


def lh(h_f, totals, k_constants):
    h, f = h_f
    t_Mg = totals[1]
    k_H2O, k_MgOH, k_HF, k_MgF, _ = k_constants
    Mg = t_Mg / (1 + k_MgOH / h + k_MgF * f)
    return k_H2O / h - h + k_MgOH * Mg / h - h * f / k_HF


def lf(h_f, totals, k_constants):
    h, f = h_f
    _, t_Mg, t_Ca = totals
    k_H2O, k_MgOH, k_HF, k_MgF, k_CaF = k_constants
    Mg = t_Mg / (1 + k_MgF * f + k_MgOH / h)
    Ca = t_Ca / (1 + k_CaF * f)
    return h + Mg + Ca - f - k_H2O / h


def eh(totals, Na=0, Cl=0):
    t_F, t_Mg, t_Ca = totals
    return Na - Cl + 2 * t_Mg + 2 * t_Ca - t_F


def ef(totals, Na=0, Cl=0):
    _, t_Mg, t_Ca = totals
    return Cl - Na - t_Mg - t_Ca


def eh_ef(totals):
    return eh(totals) + ef(totals)


def lh_lf(h_f, totals, k_constants):
    return lh(h_f, totals, k_constants) + lf(h_f, totals, k_constants)


def total_F(h_f, totals, k_constants):
    h, f = h_f
    _, t_Mg, t_Ca = totals
    _, k_MgOH, k_HF, k_MgF, k_CaF = k_constants
    HF = h * f / k_HF
    Mg = t_Mg / (1 + k_MgOH / h + k_MgF * f)
    MgF = k_MgF * Mg * f
    Ca = t_Ca / (1 + k_CaF * f)
    CaF = k_CaF * Ca * f
    return f + HF + MgF + CaF


@jax.jit
def h_f_funcs(ph_pf, totals, k_constants, Na=2250e-6, Cl=0.1399):
    h_f = 10.0 ** -ph_pf
    # t_F = totals[0]
    return np.array(
        [
            eh(totals, Na=Na, Cl=Cl) - lh(h_f, totals, k_constants),
            ef(totals, Na=Na, Cl=Cl) - lf(h_f, totals, k_constants),
        ]
    )


h_f_jac = jax.jit(jax.jacfwd(h_f_funcs))

totals = t_F, t_Mg, t_Ca = 0.0001, 0.06, 0.01
pk_constants = pk_H2O, pk_MgOH, pk_HF, pk_MgF, pk_CaF = 14, 10, 3, -1.8, -1.3
k_constants = [10.0 ** -pk for pk in pk_constants]

ph_pf_i = np.array([8, -np.log10(t_F / 2)])
ph_pf = copy.deepcopy(ph_pf_i)

#%% This is the iterative solver, following
# https://en.wikipedia.org/wiki/Newton%27s_method#k_variables,_k_functions
print(ph_pf)
target = -h_f_funcs(ph_pf, totals, k_constants)
jac = h_f_jac(ph_pf, totals, k_constants)
x_diff = np.linalg.solve(jac, target)
ph_pf = ph_pf + x_diff
print(target)
print(jac)
print(x_diff)
print(ph_pf)
h, f = h_f = 10.0 ** -ph_pf
