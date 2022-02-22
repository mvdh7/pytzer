from copy import deepcopy
from matplotlib import pyplot as plt
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
prmlib = deepcopy(pz.libraries.Seawater)
# prmlib.update_ca("Na", "Cl", prm.bC_Na_Cl_A92ii)
# prmlib.update_ca("H", "Cl", prm.bC_H_Cl_JESS)
# prmlib.update_cc("H", "Na", prm.theta_H_Na_MP98)
# prmlib.update_aa("Cl", "OH", prm.theta_Cl_OH_MP98)
# prmlib.update_cca("H", "Na", "Cl", prm.psi_H_Na_Cl_PMR97)
# prmlib.update_equilibrium("H2O", k.H2O_M88)
# prmlib.update_equilibrium("H2CO3", k.H2CO3_MP98)
# prmlib.update_equilibrium("HCO3", k.HCO3_MP98)
# prmlib.update_aa("Cl", "HCO3", prm.theta_Cl_HCO3_PP82)
# prmlib.update_caa("Na", "Cl", "HCO3", prm.psi_Na_Cl_HCO3_PP82)

for i, row in table4.iterrows():
    totals = pz.odict({"Na": row.Na, "Cl": row.Cl, "CO2": row.dic})
    solutes, pks = pz.solve(totals, library=prmlib)
    pH = -np.log10(solutes["H"])
    table4.loc[i, "pks1_pz"] = pks["H2CO3"]
    table4.loc[i, "pks2_pz"] = pks["HCO3"]

f_sqrt_Cl = np.linspace(0, 2.5, num=1000)
f_Cl = f_sqrt_Cl**2
t_CO2 = 5e-3
f_Na = f_Cl + t_CO2 * 2
pks1 = np.zeros_like(f_Cl)
pks1[0] = prmlib['equilibria']['H2CO3']() / -np.log(10)
pks2 = np.zeros_like(f_Cl)
pks2[0] = prmlib['equilibria']['HCO3']() / -np.log(10)
for i in range(1, len(f_Cl)):
    totals = pz.odict({"Na": f_Na[i], "Cl": f_Cl[i], "CO2": t_CO2})
    solutes, pks = pz.solve(totals, library=prmlib)
    # pH = -np.log10(solutes["H"])
    pks1[i] = pks["H2CO3"]
    pks2[i] = pks["HCO3"]
    
fig, axs = plt.subplots(nrows=2, dpi=300)
ax = axs[0]
ax.plot(f_Cl, pks1)
ax.scatter("Cl", "pks1_meas", data=table4)
ax = axs[1]
ax.plot(f_Cl, pks2)
ax.scatter("Cl", "pks2_meas", data=table4)
plt.tight_layout()
