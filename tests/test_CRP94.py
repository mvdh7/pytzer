import pandas as pd
import pytzer as pz

# Update unsymmetrical mixing function
pzlib = pz.libraries.Clegg94.copy()
pz = pzlib.set_func_J(pz)

# Import and solve
crp94 = pd.read_csv("tests/data/CRP94 Table 8.csv")
crp94["t_SO4"] = crp94.SO4
pz.solve_df(crp94, library=pzlib)

#%%
crp94["alpha"] = crp94.SO4 / (crp94.SO4 + crp94.HSO4)
# Doesn't quite agree at higher molality --- due to OH equilibrium being included?
# Can't easily cut that out with current solve_df setup.
# But testing in eqex.py suggests this is not the problem...
