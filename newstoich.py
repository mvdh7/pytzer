from collections import OrderedDict
from jax import numpy as np
import pytzer as pz

totals = pz.prepare.salinity_to_totals_MFWM08(salinity=35)
totals["NH3"] = 1e-6
totals["NO2"] = 2e-6
totals["H2S"] = 3e-6
totals["PO4"] = 5e-6

temperature = 298.15
ks_constants = pz.dissociation.assemble(
    temperature=temperature
)  # Needs to be replaced with PyCO2SYS

# which_pms = {"H": None, "F": None, "CO2": None, "PO4": None}
# pm_initial = pz.equilibrate.stoichiometric.guess_pm_initial(totals, which_pms)
# m_initial = 10.0 ** - pm_initial

pfixed = pz.equilibrate.stoichiometric.guess_pfixed(totals, ["H", "F", "CO3", "PO4"])
fixed = OrderedDict((k, 10.0 ** -v) for k, v in pfixed.items())
pfixed_values = np.array([v for v in pfixed.values()])
fixed_values = 10.0 ** -pfixed_values

solutes0 = pz.equilibrate.components.get_all(*fixed_values, totals, ks_constants)
solutes1 = pz.equilibrate.components.get_all_v2(fixed, totals, ks_constants)

total_targets = pz.equilibrate.stoichiometric.get_total_targets(totals, fixed)

# sfunc0 = pz.equilibrate.stoichiometric.solver_func(pfixed_values, totals, ks_constants, total_targets)
sfunc1 = pz.equilibrate.stoichiometric.solver_func_v2(
    pfixed_values, pfixed, totals, ks_constants
)

# sjac0 = pz.equilibrate.stoichiometric.solver_jac(pfixed_values, totals, ks_constants, total_targets)
sjac1 = pz.equilibrate.stoichiometric.solver_jac_v2(
    pfixed_values, pfixed, totals, ks_constants
)

# solve0 = pz.equilibrate.stoichiometric.solve(pfixed_values, totals, ks_constants, total_targets)
solve1 = pz.equilibrate.stoichiometric.solve_v2(pfixed, totals, ks_constants)

equilibria_to_solve = ["H2O", "HF", "H2CO3", "HCO3", "BOH3", "MgOH"]

params0 = pz.libraries.Seawater.get_parameters(
    solutes0, temperature=temperature, verbose=False
)
params1 = pz.libraries.Seawater.get_parameters(
    solutes1, temperature=temperature, verbose=False
)

# tsolve0 = pz.equilibrate.thermodynamic.solve(equilibria_to_solve, pfixed_values, totals, ks_constants, params, pfixed)
tsolve1 = pz.equilibrate.thermodynamic.solve_v2(
    equilibria_to_solve, pfixed, totals, ks_constants, params1
)

solutes_final, ks_constants_final = pz.solve(
    equilibria_to_solve, pfixed_values, totals, ks_constants, params0, total_targets
)
solutes_final_v2, ks_constants_final_v2 = pz.equilibrate.solve_v2(
    equilibria_to_solve, pfixed, totals, ks_constants, params1
)
print(-np.log10(solutes_final["H"]))
print(-np.log10(solutes_final_v2["H"]))

# Display results
pH_final = -np.log10(solutes_final_v2["H"])
print("pH_final = {}".format(pH_final))
for eq in equilibria_to_solve:
    print(
        "{:>5}: {:>6.3f} -> {:>6.3f}".format(
            eq, -np.log10(ks_constants[eq]), -np.log10(ks_constants_final_v2[eq])
        )
    )
