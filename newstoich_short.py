from jax import numpy as np
import pytzer as pz

temperature = 273.15
pressure = 10.10325
salinity = 35

totals = pz.prepare.salinity_to_totals_MFWM08(salinity=salinity)
totals["NH3"] = 1e-6
totals["NO2"] = 2e-6
totals["H2S"] = 3e-6
totals["PO4"] = 5e-6
totals["H4SiO4"] = 50e-6

ks_constants = pz.dissociation.assemble(temperature=temperature, totals=totals)
solutes = pz.find_solutes(totals, ks_constants)
ssolve = pz.solve_stoichiometric(totals, ks_constants)

params, log_kt_constants = pz.libraries.Seawater.get_parameters_equilibria(
    solutes, temperature=temperature, verbose=False
)

# This one is all that's really needed!
solutes_final, ks_constants_final = pz.solve_manual(
    totals, ks_constants, params, log_kt_constants
)

# Display results
pH_initial = ssolve["H"]
pH_final = -np.log10(solutes_final["H"])
print("pH init. = {:.4f}".format(pH_initial))
print("pH final = {:.4f}".format(pH_final))
for eq in log_kt_constants:
    print(
        "{:>5}: {:>6.3f} -> {:>6.3f}".format(
            eq, -np.log10(ks_constants[eq]), -np.log10(ks_constants_final[eq])
        )
    )
