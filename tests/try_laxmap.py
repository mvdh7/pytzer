import jax
from jax import numpy as np
import pytzer as pz
from collections import namedtuple
import warnings

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
pz.update_library(pz, "Clegg23")


# %%
@jax.jit
def get_thermo_error(thermo, totals, temperature, pressure, stoich, thermo_targets):
    # Prepare inputs for calculations
    lnks = {
        eq: thermo[pz.model.library["equilibria_all"].index(eq)]
        for eq in pz.model.library["equilibria_all"]
    }
    # Calculate speciation
    solutes = pz.model.library["funcs_eq"]["solutes"](totals, stoich, thermo)
    # Calculate solute and water activities
    ln_acfs = pz.model.log_activity_coefficients(solutes, temperature, pressure)
    ln_aw = pz.model.log_activity_water(solutes, temperature, pressure)
    # Calculate what the log(K)s apparently are with these stoich/thermo values
    lnk_error = {
        eq: pz.equilibrate.thermodynamic.all_reactions[eq](
            thermo_targets[pz.model.library["equilibria_all"].index(eq)],
            lnks[eq],
            ln_acfs,
            ln_aw,
        )
        for eq in pz.model.library["equilibria_all"]
    }
    thermo_error = np.array(
        [lnk_error[eq] for eq in pz.model.library["equilibria_all"]]
    )
    return thermo_error


get_thermo_error_jac = jax.jit(jax.jacfwd(get_thermo_error))


@jax.jit
def get_thermo_adjust(thermo, totals, temperature, pressure, stoich, thermo_targets):
    thermo_error = get_thermo_error(
        thermo, totals, temperature, pressure, stoich, thermo_targets
    )
    thermo_error_jac = np.array(
        get_thermo_error_jac(
            thermo, totals, temperature, pressure, stoich, thermo_targets
        )
    )
    thermo_adjust = np.linalg.solve(-thermo_error_jac, thermo_error)
    return thermo_adjust


# %%
SolveCombinedResult = namedtuple(
    "SolveCombinedResult", ["stoich", "thermo", "stoich_adjust", "thermo_adjust"]
)


def solve_combined(
    totals,
    temperature,
    pressure,
    stoich=None,
    thermo=None,
    iter_thermo=6,
    iter_stoich_per_thermo=3,
    verbose=False,
    warn_cutoff=1e-8,
):
    """Solve for equilibrium.

    This function is not JIT-ed by default, because it takes a lot longer to compile,
    but you can JIT it yourself to get a ~2x speedup.

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
        How many times to iterate the thermo part of the solver, by default 6.
    iter_pH_per_thermo : int, optional
        How many times to iterate the stoich part of the solver per thermo loop, by
        default 3.
    verbose : bool, optional
        Whether to print solving progress, by default False.
    warn_cutoff : float, optional
        If any of the final rounds of adjustments for stoich or thermo are greater than
        this value, print a convergence warning.

    Returns
    -------
    stoich : float
        Final pH.
    thermo : array-like
        Final natural logarithms of the stoichiometric equilibrium constants.
    """
    # Solver targets---known from the start
    totals = totals.copy()
    totals.update({t: 0.0 for t in pz.model.library["totals_all"] if t not in totals})
    stoich_targets = pz.model.library["funcs_eq"]["stoich_targets"](totals)
    thermo_targets = np.array(
        [
            pz.model.library["equilibria"][eq](temperature)
            for eq in pz.model.library["equilibria_all"]
        ]
    )  # these are ln(k)
    if stoich is None:
        stoich = np.array(
            [
                7.0,
                -np.log10(totals["CO2"] / 2),
                -np.log10(totals["F"] / 2),
            ]
        )
    if thermo is None:
        thermo = thermo_targets.copy()
    # Solve!
    for _t in range(iter_thermo):
        for _s in range(iter_stoich_per_thermo):
            stoich_adjust = pz.model.library["funcs_eq"]["stoich_adjust"](
                stoich, totals, thermo, stoich_targets
            )
            if verbose:
                print("STOICH", _t + 1, _s + 1)
                print(stoich_adjust)
            stoich = stoich + stoich_adjust
        thermo_adjust = get_thermo_adjust(
            thermo, totals, temperature, pressure, stoich, thermo_targets
        )
        if verbose:
            print("THERMO", _t + 1)
            print(thermo_adjust)
        thermo = thermo + thermo_adjust
    if np.any(np.abs(stoich_adjust) > warn_cutoff):
        warnings.warn(
            "Solver did not converge below `warn_cutoff` - "
            + "try increasing `iter_stoich_per_thermo`."
        )
    if np.any(np.abs(thermo_adjust) > warn_cutoff):
        warnings.warn(
            "Solver did not converge below `warn_cutoff` - "
            + "try increasing `iter_thermo`."
        )
    return SolveCombinedResult(stoich, thermo, stoich_adjust, thermo_adjust)


# %%
totals = {
    "CO2": 0.002,
    "Na": 0.5023,
    "K": 0.081,
    "Cl": 0.6,
    "BOH3": 0.0004,
    "SO4": 0.02,
    "F": 0.001,
    "Sr": 0.02,
    "Mg": 0.01,
    "Ca": 0.05,
    "Br": 0.1,
}

# This stuff is useful for printing results, but not necessary
stoich = np.array(
    [
        7.0,
        -np.log10(totals["CO2"] / 2),
        -np.log10(totals["F"] / 2),
    ]
)
thermo = np.array(
    [
        pz.model.library["equilibria"][eq](temperature)
        for eq in pz.model.library["equilibria_all"]
    ]
)

# Solve!
print(stoich)
print(thermo)
scr = solve_combined(totals, temperature, pressure, stoich=stoich, thermo=thermo)
stoich, thermo = scr[:2]
print(stoich)
print(thermo)
# [8.37045773 3.87553018]
# [-31.48490879 -13.73160454 -21.87627349 -20.28119342  -2.4239057
#   -6.5389275    3.53635291   3.19644608   3.14889391  -3.03226206]

solutes = pz.model.library["funcs_eq"]["solutes"](totals, stoich, thermo)
ks_constants = pz.model.library["funcs_eq"]["ks_constants"](thermo)


# %% SLOW to compile, then fast
def solve_combined_CO2(total_CO2, totals, temperature, pressure):
    totals["CO2"] = total_CO2
    return solve_combined(totals, temperature, pressure).stoich[0]


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
