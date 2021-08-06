from collections import OrderedDict
from jax import numpy as np
import pandas as pd
import pytzer as pz, PyCO2SYS as pyco2

#%% Define conditions
temperature = 298.15
pressure = 10.10325

# Calculate total molalities
salinity = 35
totals = {
    "Mg": 0.25,
    "CO2": 0.25,
    "PO4": 0.001,
}

# Shortcut approach
print("Running shortcut approach...")
solutes_short, ks_constants_short = pz.solve(
    totals,
    ks_constants={"MgCO3": 10.0 ** -5.4},
    ks_only=["MgCO3"],
    temperature=temperature,
    pressure=pressure,
)
print("Shortcut done!")

#%% df approach
print("Running df approach...")
df = pd.DataFrame(
    {
        "temperature": temperature,
        "pressure": pressure,
        "Mg": [0.25, 0.3, 0.5, 0.8, 1.0],
        "CO2": 0.25,
        "PO4": 0.001,
    }
)
pz.solve_df(df)
print("df done!")

#%% newstoich.py-style long-winded version
totals = {
    "SO4": 6.0,
}

# Use thermodynamic constants as first estimate of stoichiometrics
ks_constants = pz.dissociation.assemble(
    temperature=temperature,
    totals=totals,
)
ks_constants.pop("H2O")
pks_constants = OrderedDict((k, -np.log10(v)) for k, v in ks_constants.items())

# Determine which solutes are present in the final solution
solutes = pz.equilibrate.components.find_solutes(totals, ks_constants)

# Evaluate Pitzer model parameters and thermodynamic equilibrium constants
pzlib = pz.libraries.Clegg94.copy()
pz = pzlib.set_func_J(pz)
params, log_kt_constants = pzlib.get_parameters_equilibria(
    solutes=solutes, temperature=temperature, verbose=False
)

# Solve for thermodynamic equilibrium
solutes_final, ks_constants_final = pz.solve_manual(
    totals, ks_constants, params, log_kt_constants
)
pks_constants_final = OrderedDict(
    (k, -np.log10(v)) for k, v in ks_constants_final.items()
)

# # Get K0
# results = pyco2.sys(
#     temperature=temperature - 273.15, salinity=salinity, pressure=pressure - 10.10325
# )
# K0 = results["k_CO2"]

# # Display results
# pH_final = -np.log10(solutes_final["H"])
# pCO2_final = solutes_final["CO2"] * 1e6 / K0
# print("pH final = {:.4f}".format(pH_final))
# print("fCO2sw   = {:.1f}".format(pCO2_final))
# for eq in log_kt_constants:
#     print(
#         "{:>5}: {:>6.3f} -> {:>6.3f}".format(
#             eq, -np.log10(ks_constants[eq]), -np.log10(ks_constants_final[eq])
#         )
#     )
