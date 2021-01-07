from jax import numpy as np
import pytzer as pz

#%%
totals = pz.prepare.salinity_to_totals_MFWM08(salinity=35)
totals["NH3"] = 1e-6
totals["NO2"] = 2e-6
totals["H2S"] = 3e-6
totals["PO4"] = 5e-6

temperature = 298.15

ks_constants = pz.dissociation.assemble(
    temperature=temperature
)  # Needs to be replaced with PyCO2SYS

which_pms = {"H": None, "F": None, "CO2": None, "PO4": None}
pm_initial = pz.equilibrate.stoichiometric.guess_pm_initial(totals, which_pms)
total_targets = pz.equilibrate.stoichiometric.get_total_targets(totals, which_pms)
pm_initial = pz.solve_stoichiometric(pm_initial, totals, ks_constants, total_targets)
m_initial = 10.0 ** -pm_initial
solutes_initial = pz.equilibrate.components.get_all(*m_initial, totals, ks_constants)

params = pz.libraries.Seawater.get_parameters(
    solutes_initial, temperature=temperature, verbose=False
)

equilibria_to_solve = ["H2O", "HF", "H2CO3", "HCO3", "BOH3", "MgOH"]

solutes_final, ks_constants_final = pz.solve(
    equilibria_to_solve, pm_initial, totals, ks_constants, params, total_targets,
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


# thermodynamic_solve = pz.solve_thermodynamic(
#     equilibria_to_solve, pm_initial, totals, ks_constants, params
# )
# ks_constants = pz.equilibrate.thermodynamic.update_ks_constants(ks_constants, thermodynamic_solve)

# pm_final = pz.solve_stoichiometric(
#     pm_initial, totals, ks_constants, *targets
# )
# m_final = 10.0 ** -pm_final
# solutes_final = pz.equilibrate.components.get_all(
#     *m_final, totals, ks_constants
# )
