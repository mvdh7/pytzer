import pandas as pd, numpy as np
from matplotlib import pyplot as plt
import pytzer as pz

# Import data from Table 1
table1 = pd.DataFrame(
    {
        "Na": [0.485, 0.959, 1.96, 2.86, 3.83, 3.81, 4.76, 5.235, 5.222],
        "Mg": [0.025, 0.05, 0.05, 0.15, 0.18, 0.2, 0.25, 0.275, 0.29],
        "Cl": [0.525, 1.05, 2.05, 3.15, 4.182, 4.2, 5.25, 5.775, 5.79],
        "HCO3": [
            0.0015,
            0.00075,
            0.00043,
            0.00042,
            0.00063,
            0.00063,
            0.00067,
            0.00065,
            0.00066,
        ],
        "CO3": [0.0034, 0.0039, 0.0046, 0.0045, 0.0043, 0.0043, 0.0043, 0.0042, 0.0042],
        "pks1": [5.973, 5.931, 5.946, 5.993, 6.071, 6.063, 6.152, 6.205, 6.22],
        "pks2": [9.33, 9.15, 9.17, 8.95, 8.93, 8.93, 8.88, 8.93, 8.87],
    }
)
table1["ionic_strength"] = (
    table1.Na + 4 * table1.Mg + table1.Cl + table1.HCO3 + 4 * table1.CO3
) / 2
table1["sqrt_is"] = np.sqrt(table1.ionic_strength)

fig, ax = plt.subplots(dpi=300)
ax.scatter("sqrt_is", "pks1", c="Mg", data=table1)

# # Calculate total salts
# table1["MgCl2"] = table1.Mg.copy()
# table1["NaCl"] = table1.Cl - 2 * table1.MgCl2
# table1["NaCO3"] = table1.Na - table1.NaCl
# table1["dic"] = table1.NaCO3 / 2

# # Solve
# totals = pz.odict()
# totals["Na"] = table1.loc[0, "Na"]
# totals["Mg"] = table1.loc[0, "Mg"]
# totals["Cl"] = table1.loc[0, "Cl"]
# totals["CO2"] = table1.loc[0, "dic"]
# solutes, pks = pz.solve(totals)
# pH = 10 ** -solutes["H"].item()

# test = pz.solve_df(table1[["Na", "Mg", "Cl", "dic"]].rename(columns={"dic": "CO2"}))
