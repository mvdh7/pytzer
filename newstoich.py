from jax import numpy as np
import pytzer as pz
import PyCO2SYS as pyco2

temperature = 273.15
pressure = 10.10325
salinity = 35

totals = pz.prepare.salinity_to_totals_MFWM08(salinity=salinity)
totals["NH3"] = 1e-6
totals["NO2"] = 2e-6
totals["H2S"] = 3e-6
totals["PO4"] = 5e-6
totals["H4SiO4"] = 50e-6

# ks_constants_v1 = pz.dissociation.assemble_v1(
#     temperature=temperature
# )  # Needs to be replaced with PyCO2SYS - made a start below:

ks_constants = pz.dissociation.assemble(
    temperature=temperature,
    totals=totals,
)  # Needs to be replaced with PyCO2SYS - made a start below:
# Also need to add HNO2 equilibrium
WhichKs = 16
WhoseKSO4 = 1
WhoseKF = 1
WhichR = 3
totals_pyco2 = {
    "Sal": salinity,
    "TSO4": totals["SO4"],
    "TF": totals["F"],
}
ks_pyco2 = pyco2.equilibria.assemble(
    temperature - 273.15,
    pressure - 10.10325,
    totals_pyco2,
    3,
    WhichKs,
    WhoseKSO4,
    WhoseKF,
    WhichR,
)
pyco2_to_pytzer = {
    "KSO4": "HSO4",
    "KF": "HF",
    "KB": "BOH3",
    "KW": "H2O",
    "KP1": "H3PO4",
    "KP2": "H2PO4",
    "KP3": "HPO4",
    "KSi": "H4SiO4",
    "K1": "H2CO3",
    "K2": "HCO3",
    "KH2S": "H2S",
    "KNH3": "NH4",
}
for y, z in pyco2_to_pytzer.items():
    ks_constants.update({z: ks_pyco2[y]})


ptargets = pz.equilibrate.stoichiometric.create_ptargets(totals, ks_constants)
targets = pz.odict((k, 10.0 ** -v) for k, v in ptargets.items())
ptargets_values = np.array([v for v in ptargets.values()])

solutes = pz.equilibrate.components.get_solutes(totals, ks_constants, ptargets)
sfunc = pz.equilibrate.stoichiometric.solver_func(
    ptargets_values, ptargets, totals, ks_constants
)
sjac = pz.equilibrate.stoichiometric.solver_jac(
    ptargets_values, ptargets, totals, ks_constants
)
ssolve = pz.solve_stoichiometric(totals, ks_constants)
# equilibria_to_solve = ["H2O", "HF", "H2CO3", "HCO3", "BOH3", "MgOH", "HSO4"]
equilibria_to_solve = pz.libraries.Seawater.get_equilibria(temperature=temperature)

# These ones not coded in to pytzer.thermodynamic yet:
equilibria_to_solve.pop("NH4")

params = pz.libraries.Seawater.get_parameters(
    solutes, temperature=temperature, verbose=False
)
tsolve = pz.solve_thermodynamic(totals, ks_constants, params, equilibria_to_solve)

# This one is all that's really needed!
solutes_final, ks_constants_final = pz.solve_manual(
    totals, ks_constants, params, equilibria_to_solve
)

# Display results
pH_initial = ssolve["H"]
pH_final = -np.log10(solutes_final["H"])
print("pH init. = {:.4f}".format(pH_initial))
print("pH final = {:.4f}".format(pH_final))
for eq in equilibria_to_solve:
    print(
        "{:>5}: {:>6.3f} -> {:>6.3f}".format(
            eq, -np.log10(ks_constants[eq]), -np.log10(ks_constants_final[eq])
        )
    )
