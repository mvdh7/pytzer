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
pk1_PyCO2SYS = 5.957401472372938
# AWESOME


def test_pK1_PyCO2SYS():
    assert np.abs(pk1_per_kg - pk1_PyCO2SYS) < 0.02


# test_pK1_PyCO2SYS()
