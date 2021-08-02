from collections import OrderedDict
from jax import numpy as np
import pytzer as pz, PyCO2SYS as pyco2

# Define conditions
temperature = 273.15
pressure = 10.10325

# Calculate total molalities
salinity = 35
totals = {
    "Mg": 0.25,
    "CO2": 0.34,
}

# Use thermodynamic constants as first estimate of stoichiometrics
ks_constants = pz.dissociation.assemble(
    temperature=temperature,
    totals=totals,
)
# ks_constants = OrderedDict(
#     (k, v) for k, v in ks_constants.items() if k in ["H2O", "H2CO3", "HCO3", "CaCO3"]
# )

# Get the array of molalities that are the solver targets (H, CO3, F, PO4)
pfixed = pz.equilibrate.stoichiometric.create_pfixed(totals=totals)
fixed = pz.odict((k, 10.0 ** -v) for k, v in pfixed.items())
pfixed_values = np.array([v for v in pfixed.values()])

# Get first esimates of all the individual solute molalities
solutes = pz.equilibrate.components.get_solutes(totals, ks_constants, pfixed)

#%% Get the thermodynamic equilibrium constants
equilibria = pz.libraries.Seawater.get_equilibria(temperature=temperature)
equilibria = OrderedDict(
    (k, v) for k, v in equilibria.items() if k in ["H2O", "H2CO3", "HCO3", "CaCO3"]
)

# Evaluate Pitzer model parameters
params = pz.libraries.Seawater.get_parameters(
    solutes, temperature=temperature, verbose=False
)

# Solve for thermodynamic equilibrium
solutes_final, ks_constants_final = pz.solve(equilibria, totals, ks_constants, params)

# Get K0
results = pyco2.sys(temperature=temperature - 273.15, salinity=salinity,
                    pressure=pressure - 10.10325)
K0 = results["k_CO2"]

# Display results
pH_final = -np.log10(solutes_final["H"])
pCO2_final = solutes_final["CO2"] * 1e6 / K0
print("pH final = {:.4f}".format(pH_final))
print("fCO2sw   = {:.1f}".format(pCO2_final))
for eq in equilibria:
    print(
        "{:>5}: {:>6.3f} -> {:>6.3f}".format(
            eq, -np.log10(ks_constants[eq]), -np.log10(ks_constants_final[eq])
        )
    )
