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

pfixed = pz.equilibrate.stoichiometric.guess_pfixed(totals, ["H", "F", "CO3", "PO4"])
fixed = OrderedDict((k, 10.0 ** -v) for k, v in pfixed.items())
pfixed_values = np.array([v for v in pfixed.values()])

solutes = pz.equilibrate.components.get_all(fixed, totals, ks_constants)
sfunc = pz.equilibrate.stoichiometric.solver_func(
    pfixed_values, pfixed, totals, ks_constants
)
sjac = pz.equilibrate.stoichiometric.solver_jac(
    pfixed_values, pfixed, totals, ks_constants
)
ssolve = pz.solve_stoichiometric(pfixed, totals, ks_constants)
equilibria_to_solve = ["H2O", "HF", "H2CO3", "HCO3", "BOH3", "MgOH"]
params = pz.libraries.Seawater.get_parameters(
    solutes, temperature=temperature, verbose=False
)
tsolve = pz.solve_thermodynamic(
    equilibria_to_solve, pfixed, totals, ks_constants, params
)

solutes_final, ks_constants_final = pz.solve(
    equilibria_to_solve, pfixed, totals, ks_constants, params
)

# Display results
pH_final = -np.log10(solutes_final["H"])
print("pH_final = {}".format(pH_final))
for eq in equilibria_to_solve:
    print(
        "{:>5}: {:>6.3f} -> {:>6.3f}".format(
            eq, -np.log10(ks_constants[eq]), -np.log10(ks_constants_final[eq])
        )
    )
