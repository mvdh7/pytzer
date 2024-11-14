import numpy as np
import pytzer as pz

prmlib = pz.libraries.Clegg23
totals = pz.prepare.salinity_to_totals_MFWM08(35)
solutes_per_kgH2O, pks_per_kgH2O = pz.solve(totals, temperature=298.15, library=prmlib)
totals_g_per_kgH2O = np.sum(
    [pz.properties.ion_to_mass[k] * v for k, v in totals.items()]
)
kgH2O_per_kg = 1000 / (1000 + totals_g_per_kgH2O)

solutes = {k: v * kgH2O_per_kg for k, v in solutes_per_kgH2O.items()}

# K1 = [HCO3][H]/[CO2]
pk1_per_kgH2O = pks_per_kgH2O["H2CO3"]
pk1_per_kg = pk1_per_kgH2O - np.log10(kgH2O_per_kg)
pk1_PyCO2SYS = 5.954872861413003  # Lueker et al. (2000)
# AWESOME

# K2 = [CO3][H]/[HCO3]
pk2_per_kgH2O = pks_per_kgH2O["HCO3"]
pk2_per_kg = pk2_per_kgH2O - np.log10(kgH2O_per_kg)
pk2_PyCO2SYS = 9.073671457937206  # Lueker et al. (2000)
pk2_separate = -np.log10(solutes["CO3"] * solutes["H"] / solutes["HCO3"])
# Above doesn't agree because of the MCO3 species
pk2_total = -np.log10(
    (solutes["CO3"] + solutes["MgCO3"] + solutes["CaCO3"] + solutes["SrCO3"])
    * solutes["H"]
    / solutes["HCO3"]
)


def test_pK1_PyCO2SYS():
    assert np.abs(pk1_per_kg - pk1_PyCO2SYS) < 0.011


def test_pK2_PyCO2SYS():
    assert np.abs(pk2_total - pk2_PyCO2SYS) < 0.008


# test_pK1_PyCO2SYS()
# test_pK2_PyCO2SYS()
