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
equilibria = ("H2O", "H2CO3", "HCO3")
targets = ("H",)
# TODO ^ to be removed eventually, once all reactions have been added below


# TODO just define this function explicitly for each ParameterLibrary, so I don't need
# to make it so flexible with all the if statements?
def alkalinity_from_pH_ks(stoich, totals, thermo):
    # TODO uncomment below:
    # equilibria = pz.model.library["equilibria_all"]
    # targets = pz.model.library["solver_targets"]
    exp_thermo = np.exp(thermo)

    # Old (well, new) version:
    pH = stoich[targets.index("H")]
    h = 10**-pH
    alkalinity = 0.0
    ks_H2O = exp_thermo[equilibria.index("H2O")]
    alkalinity = alkalinity + ks_H2O / h - h
    ks_H2CO3 = np.exp(thermo[equilibria.index("H2CO3")])
    ks_HCO3 = np.exp(thermo[equilibria.index("HCO3")])
    alkalinity = alkalinity + (
        totals["CO2"]
        * ks_H2CO3
        * (h + 2 * ks_HCO3)
        / (h**2 + ks_H2CO3 * h + ks_H2CO3 * ks_HCO3)
    )
    # ks_BOH3 = np.exp(thermo[equilibria.index("BOH3")])
    # alkalinity = alkalinity + ks_BOH3 * totals["BOH3"] / (h + ks_BOH3)

    # # Possible new (well, old) version:
    # # But this breaks the temperature grad!  Go back to the old (well, new) approach
    # ks = {eq: exp_thermo[equilibria.index(eq)] for eq in equilibria}
    # ptargets = {t: stoich[targets.index(t)] for t in targets}
    # solutes = pz.equilibrate.components.get_solutes(totals, ks, ptargets)
    # alkalinity = pz.equilibrate.stoichiometric.get_alkalinity(solutes)
    return alkalinity


def get_stoich_error(stoich, totals, thermo, stoich_targets):
    return np.array([alkalinity_from_pH_ks(stoich, totals, thermo)]) - stoich_targets


def get_thermo_error(thermo, totals, temperature, pressure, stoich, thermo_targets):
    # TODO uncomment below:
    # equilibria = pz.model.library["equilibria_all"]
    # targets = pz.model.library["solver_targets"]
    pH = stoich[0]
    h = 10**-pH
    lnks = {}
    if "H2O" in equilibria:
        lnks["H2O"] = thermo[equilibria.index("H2O")]
    if "H2CO3" in equilibria:
        lnks["H2CO3"] = thermo[equilibria.index("H2CO3")]
    if "HCO3" in equilibria:
        lnks["HCO3"] = thermo[equilibria.index("HCO3")]
    ks = {k: np.exp(v) for k, v in lnks.items()}
    # Get shots at the targets
    solutes = totals.copy()
    if "H2O" in equilibria:
        solutes["H"] = h
        solutes["OH"] = ks["H2O"] / h
    if "CO2" in totals:
        total_CO2 = totals["CO2"]
    else:
        total_CO2 = 0.0
    if "H2CO3" in equilibria and "HCO3" in equilibria:
        # If only one of them is present, then the calculation below needs to change
        # accordingly
        co2_denom = h**2 + h * ks["H2CO3"] + ks["H2CO3"] * ks["HCO3"]
        solutes["CO2"] = total_CO2 * h**2 / co2_denom
        solutes["HCO3"] = total_CO2 * ks["H2CO3"] * h / co2_denom
        solutes["CO3"] = total_CO2 * ks["H2CO3"] * ks["HCO3"] / co2_denom
    ln_acfs = pz.log_activity_coefficients(solutes, temperature, pressure)
    ln_aw = pz.log_activity_water(solutes, temperature, pressure)
    lnk = {}
    if "H2O" in equilibria:
        lnk["H2O"] = lnks["H2O"] + ln_acfs["H"] + ln_acfs["OH"] - ln_aw
    if "H2CO3" in equilibria:
        lnk["H2CO3"] = (
            lnks["H2CO3"] + ln_acfs["HCO3"] + ln_acfs["H"] - ln_acfs["CO2"] - ln_aw
        )
    if "HCO3" in equilibria:
        lnk["HCO3"] = lnks["HCO3"] + ln_acfs["CO3"] + ln_acfs["H"] - ln_acfs["HCO3"]
    thermo_attempt = []
    for eq in equilibria:
        thermo_attempt.append(lnk[eq])
    thermo_attempt = np.array(thermo_attempt)
    thermo_error = thermo_attempt - thermo_targets
    return thermo_error


# %% RESET
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
    # TODO uncomment below:
    # equilibria = pz.model.library["equilibria_all"]
    # targets = pz.model.library["solver_targets"]  # and use this!
    stoich_targets = np.array(
        [
            pz.equilibrate.stoichiometric.get_explicit_alkalinity(totals),
        ]
    )
    thermo_targets = np.array(
        [pz.model.library["equilibria"][eq](temperature) for eq in equilibria]
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
            stoich_adjust = np.where(
                np.abs(stoich_adjust) > 1, np.sign(stoich_adjust), stoich_adjust
            )
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
    # TODO return thermo as a dict using `equilibria` for its order (stoich too?)
    # TODO return results as a named tuple also including the final xxx_adjust values
    return stoich, thermo


# %%
all_totals = {
    # Equilibrating
    # "BOH3",
    "Ca",
    "CO2",
    # "F",
    "Mg",
    "Sr",
    # "SO4",
    # Non-equilibrating
    "Br",
    "Cl",
    "K",
    "Na",
}
all_solutes = {
    # "BOH3",
    # "BOH4",
    "Br",
    "Ca",
    # "CaCO3",
    # "CaF",
    "Cl",
    "CO2",
    "CO3",
    # "F",
    "HCO3",
    # "HF",
    # "HSO4",
    "K",
    "Mg",
    # "MgCO3",
    # "MgF",
    # "MgOH",
    "Na",
    # "SO4",
    "Sr",
    # "SrCO3",
    # "SrF",
}

totals = {
    "CO2": 0.002,
    "Na": 0.5023,
    "K": 0.2,
    "Cl": 0.7,
}
totals.update({t: 0.0 for t in all_totals if t not in totals})

# This stuff is obsolete but still just useful for printing results
pH_guess = 8.0
_k_constants = {
    "H2O": pz.model.library["equilibria"]["H2O"](temperature),
    "H2CO3": pz.model.library["equilibria"]["H2CO3"](temperature),
    "HCO3": pz.model.library["equilibria"]["HCO3"](temperature),
}
lnks_H2O_guess = _k_constants["H2O"]
lnks_H2CO3_guess = _k_constants["H2CO3"]
lnks_HCO3_guess = _k_constants["HCO3"]
coeffs = np.array([lnks_H2O_guess, lnks_H2CO3_guess, lnks_HCO3_guess])
thermo = coeffs
stoich = np.array([pH_guess])

# Solve!
pH, coeffs_final = solve_combined(totals, temperature, pressure, stoich=stoich)
print(stoich)
print(pH)
print(coeffs)
print(coeffs_final)
# [8.]
# [8.80791499]
# [-32.22023869 -14.62482355 -23.78504939]
# [-31.56800072 -13.74433547 -22.05412727]


# %% SLOW to compile, then fast
def solve_combined_CO2(total_CO2, totals, temperature, pressure):
    totals["CO2"] = total_CO2
    return solve_combined(totals, temperature, pressure)[0][0]


# ^ this can be gradded w.r.t. total_CO2:
scgrad = jax.jacfwd(solve_combined_CO2)(0.002, totals, temperature, pressure)
# Can probably just grad the solve_combined function too to get a dict out,  not tried
# These also work, but again slow to compile:  (NO LONGER WORKS WITH OLD/NEW ALKALINITY)
tgrad = jax.jacfwd(
    lambda temperature: solve_combined(totals, temperature, pressure)[0][0]
)(temperature)
pgrad = jax.jacfwd(
    lambda pressure: solve_combined(totals, temperature, pressure)[0][0]
)(pressure)

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
