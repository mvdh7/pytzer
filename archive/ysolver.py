from collections import OrderedDict
from scipy import optimize
from jax import numpy as np
import pytzer as pz

totals = pz.prepare.salinity_to_totals_MFWM08()
totals["NH3"] = 1e-6
totals["NO2"] = 2e-6
totals["H2S"] = 3e-6
totals["PO4"] = 5e-6
# totals["Cl"] += 0.005  # required to work for HSO4 - equilibrium needs to be active at low pH

ks_constants = pz.dissociation.assemble()  # Needs to be replaced with PyCO2SYS
pks_constants = {k: -np.log10(v) for k, v in ks_constants.items()}

targets = pz.equilibrate.stoichiometric.get_total_targets(totals)
solver_x = -np.log10(
    np.array(
        [
            1e-8,
            totals["F"] / 2,
            totals["CO2"] / 10,
            totals["PO4"] / 2,
        ]
    )
)
# ks_constants["HSO4"] = 10.0 ** -6  # required to work for HSO4 - better starting point?
print(-np.log10(ks_constants["HSO4"]))
solver_x = pz.equilibrate.stoichiometric.solve(solver_x, totals, ks_constants, *targets)
print(solver_x)
solver_molalities = [10.0 ** -x for x in solver_x]
solutes = pz.equilibrate.components.get_solutes(
    *solver_molalities, totals, ks_constants
)

#%%
params = pz.libraries.Seawater.get_parameters(solutes, verbose=False)

#%%
equilibria_to_model = ["H2O", "HF"]
log_kt_constants = OrderedDict(
    (rxn, pz.dissociation.all_log_ks[rxn]()) for rxn in equilibria_to_model  # , "HSO4"]
)
pks_constants_to_solve = np.array(
    [-np.log10(np.exp(log_kt)) for log_kt in log_kt_constants.values()]
)
args = (solver_x, totals, ks_constants, targets, params, log_kt_constants)

#%% Warm up the solver functions
# pks_constants_to_solve = np.array([3.5])
print(pks_constants_to_solve)
g_total = pz.equilibrate.thermodynamic.get_Gibbs_equilibria(
    pks_constants_to_solve, *args
)
print(g_total)
g_jac = pz.equilibrate.thermodynamic.jac_Gibbs_equilibria(pks_constants_to_solve, *args)
print(g_jac)
#%
x_diff = np.linalg.solve(g_jac, -g_total)
print(x_diff)
x_diff = np.where(x_diff > 0.1, 0.1, x_diff)
x_diff = np.where(x_diff < -0.1, -0.1, x_diff)
print(x_diff)
#%
# pks_constants_to_solve += x_diff
# print(pks_constants_to_solve)

#%% Solve solve the thermodynamics
pkstars_optresult = optimize.root(
    pz.equilibrate.thermodynamic.get_Gibbs_equilibria,
    pks_constants_to_solve,
    args=args,
    method="hybr",
    jac=pz.equilibrate.thermodynamic.jac_Gibbs_equilibria,
)

# Should be zero:
g_solved = pz.equilibrate.thermodynamic.get_Gibbs_equilibria(
    pkstars_optresult["x"], *args
)

solver_test = pz.equilibrate.thermodynamic.solve(
    equilibria_to_model, solver_x, totals, ks_constants, params
)

#%% Final stoichiometric solution
for i, rxn in enumerate(log_kt_constants):
    ks_constants[rxn] = 10.0 ** -pkstars_optresult["x"][i]
solver_final = pz.equilibrate.stoichiometric.solve(
    solver_x, totals, ks_constants, *targets
)
solver_molalities_final = [10.0 ** -x for x in solver_final]
solutes_final = pz.equilibrate.components.get_solutes(
    *solver_molalities_final, totals, ks_constants
)
