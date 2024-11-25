import jax
from jax import numpy as np
import pytzer as pz

solutes = {
    "Na": 1.0,
    "K": 1.0,
    "Mg": 1.0,
    "Cl": 1.0,
    "HSO4": 1.0,
    "SO4": 1.0,
    "CO2": 1.0,
}
solutes.update({k: 1.0 for k in pz.model.library["solutes_all"] if k not in solutes})
tp = (298.15, 10.1325)

Gex = pz.model.Gibbs_nRT(solutes, *tp)
Gex_new = pz.model._Gibbs_nRT_wow(solutes, *tp)

print(Gex)
print(Gex_new)

# %%

# %%
# # solutes = {
# #     "Na": 1.5,
# #     "Ca": 1.0,
# #     "H": 1.0,
# #     "Cl": 3.0,
# #     "SO4": 0.75,
# #     "CO2": 0.5,
# # }
# # temperature = 298.15
# # pressure = 10.1
# prmlib = pz.libraries.Clegg23
# # prmlib.update_ca("Na", "Cl", pz.parameters.bC_Na_Cl_A92ii)  # for testing
# # prmlib.update_nn("CO2", "CO2", pz.parameters.theta_BOH4_Cl_CWTD23)  # for testing
# # prmlib.update_nnn("CO2", pz.parameters.mu_tris_tris_tris_LTA21)  # for testing
# pz = prmlib.set_func_J(pz)


# # Gargs = (solutes, temperature, pressure)
# # Gex_nRT = pz.model.Gibbs_nRT
# # G = Gex_nRT(*Gargs)

# # fGsj = pz.model.log_activity_coefficients

# # Gs = jax.grad(Gex_nRT)(*Gargs)
# # Gsj = fGsj(*Gargs)
# # Gt = jax.grad(Gex_nRT, argnums=1)(*Gargs)
# # Gp = jax.grad(Gex_nRT, argnums=2)(*Gargs)

# # osolutes = pz.odict(solutes)
# # params = prmlib.get_parameters(
# #     solutes=osolutes, temperature=temperature, pressure=pressure
# # )
# # G_old = pz.model_old.Gibbs_nRT(osolutes, **params)
# # Gs_old = pz.model_old.log_activity_coefficients(osolutes, **params)

# # pz.update_library(pz, "Seawater")
# # Gl = pz.model.Gibbs_nRT(*Gargs)
# # print(G)
# # print(G_old)
# # print(Gl)
# # pz.update_library(pz, "Clegg23")
# # pz = prmlib.set_func_J(pz)

# # %%
# solutes = {
#     # -1 anions
#     "Cl": 1.0,
#     "HSO4": 1.0,
#     "OH": 1.0,
#     "Br": 1.0,
#     "HCO3": 1.0,
#     "BOH4": 1.0,
#     "F": 1.0,
#     # -2 anions
#     "SO4": 1.0,
#     "CO3": 1.0,
#     # +1 cations
#     "H": 1.0,
#     "Na": 1.0,
#     "K": 1.0,
#     "MgF": 1.0,
#     "CaF": 1.0,
#     "MgOH": 1.0,
#     # +2 cations
#     "Mg": 1.0,
#     "Ca": 1.0,
#     "Sr": 1.0,
#     # 0 neutrals
#     "BOH3": 1.0,
#     "CO2": 1.0,
#     "HF": 1.0,
#     "MgCO3": 1.0,
#     "CaCO3": 1.0,
#     "SrCO3": 1.0,
# }
# solutes_old = pz.odict(solutes)
# temperature = 278.15
# pressure = 10.1325

# params = prmlib.get_parameters(solutes_old, temperature=temperature, pressure=pressure)
# acfs_old = pz.model_old.activity_coefficients(solutes_old, **params)

# acfs = pz.model.activity_coefficients(solutes, temperature, pressure)

# acfs_diff = {k: (acfs[k] - acfs_old[k]).item() for k in acfs}
# for k, v in acfs_diff.items():
#     print(k, v)

# # %%
# totals = {
#     "CO2": 0.002,
#     "Na": 0.5023,
#     "K": 0.081,
#     "Cl": 0.6,
#     "BOH3": 0.0004,
#     "SO4": 0.02,
#     "F": 0.001,
#     "Sr": 0.02,
#     "Mg": 0.01,
#     "Ca": 0.05,
#     "Br": 0.1,
# }

# # This stuff is useful for printing results, but not necessary
# stoich = np.array(
#     [
#         7.0,
#         -np.log10(totals["CO2"] / 2),
#         -np.log10(totals["F"] / 2),
#     ]
# )
# thermo = np.array(
#     [
#         pz.model.library["equilibria"][eq](temperature)
#         for eq in pz.model.library["equilibria_all"]
#     ]
# )

# # Solve!
# print(stoich)
# print(thermo)
# scr = pz.equilibrate.new.solve_combined(
#     totals, temperature, pressure, stoich=stoich, thermo=thermo
# )
# stoich, thermo = scr[:2]
# print(stoich)
# print(thermo)
# # [8.37045773 3.87553018]
# # [-31.48490879 -13.73160454 -21.87627349 -20.28119342  -2.4239057
# #   -6.5389275    3.53635291   3.19644608   3.14889391  -3.03226206]

# solutes = pz.model.library["funcs_eq"]["solutes"](totals, stoich, thermo)
# ks_constants = pz.model.library["funcs_eq"]["ks_constants"](thermo)


# # %% SLOW to compile, then fast
# def solve_combined_CO2(total_CO2, totals, temperature, pressure):
#     totals["CO2"] = total_CO2
#     return pz.equilibrate.new.solve_combined(totals, temperature, pressure).stoich[0]


# # ^ this can be gradded w.r.t. total_CO2:
# scgrad = jax.jacfwd(solve_combined_CO2)(0.002, totals, temperature, pressure)
# # Can probably just grad the solve_combined function too to get a dict out,  not tried
# # These also work, but again slow to compile:  (NO LONGER WORKS WITH OLD/NEW ALKALINITY)
# tgrad = jax.jacfwd(
#     lambda temperature: solve_combined(totals, temperature, pressure)[0][0]
# )(temperature)
# pgrad = jax.jacfwd(
#     lambda pressure: solve_combined(totals, temperature, pressure)[0][0]
# )(pressure)

# %% pH solver (jax)
# pH_lnks = np.array([pH_guess, *coeffs])

# def cond_pH(pH):
#     return np.abs(get_shot_alkalinity(pH, *coeffs, totals["CO2"])) > 1e-12


# def body_pH(pH):
#     missed_alk = get_shot_alkalinity(pH, *coeffs, totals["CO2"])
#     missed_alk_grad = jax.grad(get_shot_alkalinity)(pH, *coeffs, totals["CO2"])
#     adjust_alk = -missed_alk / missed_alk_grad
#     return pH + adjust_alk


# def solve_pH(pH):
#     return jax.lax.while_loop(cond_pH, body_pH, pH)


# # thermo solver (jax)
# def cond_thermo(lnks):
#     return np.any(np.abs(get_shots(lnks, pH_guess, totals["CO2"])) > 1e-12)


# def body_thermo(lnks):
#     missed_by = get_shots(lnks, pH_guess, totals["CO2"])
#     missed_jac = np.array(jax.jacfwd(get_shots)(lnks, pH_guess, totals["CO2"]))
#     adjust_aim = np.linalg.solve(-missed_jac, missed_by)
#     return lnks + adjust_aim


# @jax.jit
# def solve_thermo(lnks):
#     return jax.lax.while_loop(cond_thermo, body_thermo, lnks)


# # %% combined solver (jax)
# def cond_combo(pH_lnks):
#     cond_pH = np.abs(get_shot_alkalinity(*pH_lnks, totals["CO2"])) > 1e-8
#     # pH more sensitive to thermo than thermo is to pH => don't need cond_thermo?
#     cond_thermo = np.abs(get_shots(pH_lnks[1:], pH_lnks[0], totals["CO2"])) > 1e-8
#     cond = np.array([cond_pH, *cond_thermo])
#     return np.any(cond)


# def body_combo(pH_lnks):
#     pH = pH_lnks[0]
#     lnks = pH_lnks[1:]
#     # pH part
#     missed_alk = get_shot_alkalinity(pH, *lnks, totals["CO2"])
#     missed_alk_grad = jax.grad(get_shot_alkalinity)(pH, *lnks, totals["CO2"])
#     adjust_alk = -missed_alk / missed_alk_grad
#     # lnks part
#     missed_by = get_shots(lnks, pH_guess, totals["CO2"])
#     missed_jac = np.array(jax.jacfwd(get_shots)(lnks, pH_guess, totals["CO2"]))
#     adjust_aim = np.linalg.solve(-missed_jac, missed_by)
#     return np.array([pH + adjust_alk, *(lnks + adjust_aim)])


# def solve_combo(pH_lnks):
#     # -- SUPER SLOW, doesn't work
#     return jax.lax.while_loop(cond_combo, body_combo, pH_lnks)

# %%
# print(pH_lnks)
# pH_lnks = solve_combo(pH_lnks)
# print(pH_lnks)

# # %%
# print(pH_guess)
# pH_guess = solve_pH(pH_guess)
# print(pH_guess)
# print(coeffs)
# coeffs = solve_thermo(coeffs)
# print(coeffs)


# %% pH solver (manual)

# missed_alk = get_shot_alkalinity(pH_guess, *coeffs, totals["CO2"])
# missed_alk_grad = jax.grad(get_shot_alkalinity)(pH_guess, *coeffs, totals["CO2"])
# adjust_alk = -missed_alk / missed_alk_grad

# print(pH_guess, missed_alk * 1e6)
# pH_guess += adjust_alk
# print(pH_guess, missed_alk * 1e6)

# # %% thermo solver (manual)
# # coeffs = test.x

# missed_by = get_shots(coeffs, pH_guess, totals["CO2"])
# missed_jac = np.array(jax.jacfwd(get_shots)(coeffs, pH_guess, totals["CO2"]))
# adjust_aim = np.linalg.solve(-missed_jac, missed_by)
# # adjust_aim = np.where(adjust_aim > 0.5, 0.5, adjust_aim)
# # adjust_aim = np.where(adjust_aim < -0.5, -0.5, adjust_aim)

# print(coeffs)
# coeffs += adjust_aim
# print(coeffs)

# # %%
# from jax.scipy import optimize

# test = optimize.minimize(get_shots, coeffs, args=(totals["CO2"],), method="BFGS")

# %%
# solutes = {"Na": 2.0, "Mg": 1.0, "Cl": 4.0}
# charges = {"Na": 1, "Mg": 2, "Cl": -1}


# cations = {"Na", "Mg"}
# anions = {"Cl"}

# multipliers = {"Na": dict(Cl=1.5), "Mg": dict(Cl=1.2)}


# def iterm(cation_anion):
#     cation, anion = cation_anion
#     return solutes[cation] * solutes[anion] * multipliers[cation][anion]

# cation = "Na"
# anion = "Cl"
# it_Na = iterm((cation, anion))
# it_Mg = iterm(("Mg", "Cl"))

# pr = list(itertools.product(cations, anions))

# it_map = sum(map(iterm, pr))
# it_jaxmap = jax.lax.map(iterm, np.array(pr))


# %%
# def _ionic_strength(solute, charge):
#     return 0.5 * solute * charge ** 2


# def dict_sum(kv):
#     s = 0.0
#     for v in kv.values():
#         s += v
#     return s


# def ionic_strength(solutes, charges):
#     istrs = jax.tree.map(_ionic_strength, solutes, charges)
#     return dict_sum(istrs)


# def ionic_strength_2(solutes, charges):
#     s = 0.0
#     for k, v in solutes.items():
#         s += 0.5 * v * charges[k] ** 2
#     return s

# istr = ionic_strength(solutes, charges)
# istr2 = ionic_strength_2(solutes, charges)
