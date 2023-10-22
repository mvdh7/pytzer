from collections import OrderedDict
import pandas as pd, numpy as np
import pytzer as pz

v4 = pd.read_csv("tests/data/pytzerQuickStart_py_v0_4_3__GM89.csv")
solutes = [
    k
    for k in v4.columns
    if k not in ["tempK", "pres", "aw", "osm"] and not k.startswith("g")
]
for s in solutes:
    v4["g{}_new".format(s)] = np.nan
plib = pz.libraries.GM89
plib.set_func_J(pz)

for i, row in v4.iterrows():
    print(i)
    solutes = OrderedDict(
        (k, v)
        for k, v in row.items()
        if k not in ["tempK", "pres", "aw", "osm"] and not k.startswith("g")
    )
    params = plib.get_parameters(
        solutes, temperature=row.tempK, pressure=row.pres, verbose=False
    )
    acfs = pz.activity_coefficients(solutes, **params)
    for s in solutes:
        v4.loc[i, "g{}_new".format(s)] = acfs[s]
for s in solutes:
    gsdiff = "g{}_diff".format(s)
    v4[gsdiff] = v4["g{}_new".format(s)] - v4["g{}".format(s)]
    v4.loc[v4[gsdiff].abs() < 1e-12, gsdiff] = 0
