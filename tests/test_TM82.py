from copy import deepcopy
import pytzer as pz, numpy as np, pandas as pd
from pytzer import parameters as prm, dissociation as k

# Import data from Thurmond & Millero (1982) Table IV
table4 = pd.DataFrame(
    {
        "Cl": [0.5, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.5, 6],
        "pks1_meas": [
            5.997,
            5.984,
            5.97,
            5.964,
            5.968,
            5.992,
            np.nan,
            6.046,
            6.092,
            6.13,
            6.227,
            6.264,
        ],
        "pks2_meas": [
            9.58,
            9.53,
            9.48,
            9.44,
            9.41,
            9.43,
            9.46,
            9.46,
            9.53,
            9.56,
            9.67,
            9.71,
        ],
    }
)

table4["dic"] = 5e-3
table4["Na"] = table4.Cl + table4.dic * 2
table4["pks1_pz"] = np.nan
table4["pks2_pz"] = np.nan
table4 = table4[["Na", "Cl", "dic", "pks1_meas", "pks1_pz", "pks2_meas", "pks2_pz"]]

# Use Pytzer to determine pK1* and pK2*
prmlib = deepcopy(pz.libraries.myMCS)
# prmlib.update_ca("Na", "Cl", prm.bC_Na_Cl_A92ii)
# prmlib.update_ca("H", "Cl", prm.bC_H_Cl_JESS)
# prmlib.update_cc("H", "Na", prm.theta_H_Na_MP98)
# prmlib.update_aa("Cl", "OH", prm.theta_Cl_OH_MP98)
# prmlib.update_cca("H", "Na", "Cl", prm.psi_H_Na_Cl_PMR97)
# prmlib.update_equilibrium("H2O", k.H2O_M88)
# prmlib.update_equilibrium("H2CO3", k.H2CO3_MP98)
# prmlib.update_equilibrium("HCO3", k.HCO3_MP98)
# prmlib.update_aa("Cl", "HCO3", prm.theta_Cl_HCO3_PP82)
prmlib.update_caa("Na", "Cl", "HCO3", prm.psi_Na_Cl_HCO3_PP82)

# i = 11
# row = table4.loc[i]
for i, row in table4.iterrows():
    totals = pz.odict({"Na": row.Na, "Cl": row.Cl, "CO2": row.dic})
    solutes, pks = pz.solve(totals, library=prmlib)
    pH = -np.log10(solutes["H"])
    table4.loc[i, "pks1_pz"] = pks["H2CO3"]
    table4.loc[i, "pks2_pz"] = pks["HCO3"]
