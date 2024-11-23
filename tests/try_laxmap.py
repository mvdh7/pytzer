import jax
from jax import numpy as np
import pytzer as pz


solutes = {
    "Na": 1.5,
    "Ca": 1.0,
    "H": 1.0,
    "Cl": 3.0,
    "SO4": 0.75,
    "CO2": 0.5,
}
temperature = 298.15
pressure = 10.1
prmlib = pz.libraries.Clegg23
prmlib.update_ca("Na", "Cl", pz.parameters.bC_Na_Cl_A92ii)  # for testing
prmlib.update_nn("CO2", "CO2", pz.parameters.theta_BOH4_Cl_CWTD23)  # for testing
prmlib.update_nnn("CO2", pz.parameters.mu_tris_tris_tris_LTA21)  # for testing
pz = prmlib.set_func_J(pz)


Gargs = (solutes, temperature, pressure)
Gex_nRT = pz.model.Gibbs_nRT
G = Gex_nRT(*Gargs)

fGsj = pz.model.log_activity_coefficients

Gs = jax.grad(Gex_nRT)(*Gargs)
Gsj = fGsj(*Gargs)
Gt = jax.grad(Gex_nRT, argnums=1)(*Gargs)
Gp = jax.grad(Gex_nRT, argnums=2)(*Gargs)

osolutes = pz.odict(solutes)
params = prmlib.get_parameters(
    solutes=osolutes, temperature=temperature, pressure=pressure
)
G_old = pz.model_old.Gibbs_nRT(osolutes, **params)
Gs_old = pz.model_old.log_activity_coefficients(osolutes, **params)

pz.update_library(pz, "Seawater")

Gl = pz.model.Gibbs_nRT(*Gargs)

print(G)
print(G_old)
print(Gl)


# %% Solver
def alkalinity_from_pH_ks(stoich, totals, thermo):
    pH = stoich[0]
    lnks_H2CO3, lnks_HCO3, lnks_H2O = thermo
    h = 10**-pH
    ks_H2CO3 = np.exp(lnks_H2CO3)
    ks_HCO3 = np.exp(lnks_HCO3)
    ks_H2O = np.exp(lnks_H2O)
    alkalinity = (
        totals["CO2"]
        * ks_H2CO3
        * (h + 2 * ks_HCO3)
        / (h**2 + ks_H2CO3 * h + ks_H2CO3 * ks_HCO3)
        + ks_H2O / h
        + h
    )
    return alkalinity


def get_stoich_error(stoich, totals, thermo, stoich_targets):
    return np.array([alkalinity_from_pH_ks(stoich, totals, thermo)]) - stoich_targets


def get_thermo_error(thermo, totals, temperature, pressure, stoich, thermo_targets):
    pH = stoich[0]
    h = 10**-pH
    lnks_H2CO3, lnks_HCO3, lnks_H2O = thermo
    ks_H2CO3, ks_HCO3, ks_H2O = np.exp(thermo)
    # Get shots at the targets
    co2_denom = h**2 + h * ks_H2CO3 + ks_H2CO3 * ks_HCO3
    solutes = {
        "Na": totals["Na"],
        "Cl": totals["Cl"],
        "H": h,
        "OH": ks_H2O / h,
        "CO2": totals["CO2"] * h**2 / co2_denom,
        "HCO3": totals["CO2"] * ks_H2CO3 * h / co2_denom,
        "CO3": totals["CO2"] * ks_H2CO3 * ks_HCO3 / co2_denom,
    }
    ln_acfs = pz.log_activity_coefficients(solutes, temperature, pressure)
    ln_aw = pz.log_activity_water(solutes, temperature, pressure)
    lnk_H2CO3 = lnks_H2CO3 + ln_acfs["HCO3"] + ln_acfs["H"] - ln_acfs["CO2"] - ln_aw
    lnk_HCO3 = lnks_HCO3 + ln_acfs["CO3"] + ln_acfs["H"] - ln_acfs["HCO3"]
    lnk_H2O = lnks_H2O + ln_acfs["H"] + ln_acfs["OH"] - ln_aw
    thermo_attempt = np.array([lnk_H2CO3, lnk_HCO3, lnk_H2O])
    thermo_error = thermo_attempt - thermo_targets
    return thermo_error


# %% RESET
totals = {"CO2": 0.002, "Na": 0.7023, "Cl": 0.7}
pH_guess = 8.0
_k_constants = {
    "H2CO3": pz.model.library["equilibria"]["H2CO3"](temperature),
    "HCO3": pz.model.library["equilibria"]["HCO3"](temperature),
    "H2O": pz.model.library["equilibria"]["H2O"](temperature),
}
lnks_H2CO3_guess = _k_constants["H2CO3"]
lnks_HCO3_guess = _k_constants["HCO3"]
lnks_H2O_guess = _k_constants["H2O"]
coeffs = np.array([lnks_H2CO3_guess, lnks_HCO3_guess, lnks_H2O_guess])


@jax.jit
def solve_combined(
    totals,
    temperature,
    pressure,
    stoich=None,
    thermo=None,
    iter_thermo=5,
    iter_stoich_per_thermo=3,
):
    """Solve for equilibrium.

    Parameters
    ----------
    totals : dict
        The total molality of each solute or solute system.
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in dbar.
    stoich : array-like, optional
        First guess for pH.
    thermo : array-like, optional
        First guesses for the natural logarithms of the stoichiometric equilibrium
        constants.
    iter_thermo : int, optional
        How many times to iterate the thermo part of the solver, by default 5.
    iter_pH_per_thermo : int, optional
        How many times to iterate the stoich part of the solver per thermo loop, by
        default 3.

    Returns
    -------
    stoich : float
        Final pH.
    thermo : array-like
        Final natural logarithms of the stoichiometric equilibrium constants.
    """
    # Solver targets---known from the start
    stoich_targets = np.array(
        [
            pz.equilibrate.stoichiometric.get_explicit_alkalinity(totals),
        ]
    )
    thermo_targets = np.array(
        [
            pz.model.library["equilibria"]["H2CO3"](temperature),
            pz.model.library["equilibria"]["HCO3"](temperature),
            pz.model.library["equilibria"]["H2O"](temperature),
        ]
    )  # these are ln(k)
    if stoich is None:
        stoich = np.array([8.0])  # TODO improve this
    if thermo is None:
        thermo = thermo_targets.copy()
    # Solve!
    for _ in range(iter_thermo):
        for _ in range(iter_stoich_per_thermo):
            stoich_error = get_stoich_error(stoich, totals, thermo, stoich_targets)
            stoich_error_jac = jax.jacfwd(get_stoich_error)(
                stoich, totals, thermo, stoich_targets
            )
            stoich_adjust = np.linalg.solve(-stoich_error_jac, stoich_error)
            stoich = stoich + stoich_adjust
        thermo_error = get_thermo_error(
            thermo, totals, temperature, pressure, stoich, thermo_targets
        )
        thermo_error_jac = np.array(
            jax.jacfwd(get_thermo_error)(
                thermo, totals, temperature, pressure, stoich, thermo_targets
            )
        )
        thermo_adjust = np.linalg.solve(-thermo_error_jac, thermo_error)
        thermo = thermo + thermo_adjust
    return stoich, thermo


thermo = coeffs
stoich = np.array([pH_guess])
pH, coeffs_final = solve_combined(totals, temperature, pressure)
print(pH_guess)
print(pH)
print(coeffs)
print(coeffs_final)


# %%
def solve_combined_CO2(total_CO2, totals, temperature, pressure):
    totals["CO2"] = total_CO2
    return solve_combined(totals, temperature, pressure)


# ^ this can be gradded w.r.t. total_CO2

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
