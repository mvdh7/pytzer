import numpy as np
import pytzer as pz, pytzer4 as pz4

prmlib = pz.libraries.Waters13_Clegg22

totals = pz.odict()
totals["Na"] = 0.4861818  # - 0.04
totals["Mg"] = 0.0547402
totals["Ca"] = 0.01075004
totals["K"] = 0.01058004
totals["Cl"] = 0.5692021  # + 0.02
totals["SO4"] = 0.02927011
# totals["tris"] = 0.04

solutes, pks = pz.solve(totals, library=prmlib, temperature=298.15)
pH = -np.log10(solutes["H"])
params = prmlib.get_parameters(solutes, temperature=278.15, verbose=False)
acfs = pz.activity_coefficients(solutes, **params)
aw = pz.activity_water(solutes, **params)

waterk = (solutes["H"] * solutes["OH"] * acfs["H"] * acfs["OH"] / aw).item()
waterk_paper = 8.2481e-8 * 5.0620e-8 * 0.76188 * 0.57223 / 0.98142
waterk_official = np.exp(prmlib["equilibria"]["H2O"](278.15))

# trisk = (solutes["tris"] * solutes["H"] / solutes["trisH"]).item()
trisk_official = np.exp(prmlib["equilibria"]["trisH"](278.15))

sulfk = (
    solutes["H"]
    * solutes["SO4"]
    * acfs["H"]
    * acfs["SO4"]
    / (solutes["HSO4"] * acfs["HSO4"])
).item()
sulfk_paper = 0.02927 * 8.2481e-8 * 0.76188 * 0.0972 / (0.72148 * 1.3286e-8)
sulfk_official = np.exp(prmlib["equilibria"]["HSO4"](278.15))

mgk = (
    solutes["Mg"]
    * acfs["Mg"]
    * solutes["OH"]
    * acfs["OH"]
    / (solutes["MgOH"] * acfs["MgOH"])
).item()
mgk_paper = 0.05474 * 0.22054 * 5.062e-8 * 0.57223 / (4.5149e-8 * 0.90276)
mgk_official = np.exp(prmlib["equilibria"]["MgOH"](278.15))
# this was upside down - and all the other formation constants might be too!
# need to change in both pz.components (rearrange get_X equations)
# AND in pz.thermodynamic (swap signs of log_kt and log_ks terms)

solutes_paper = pz.odict()
solutes_paper["H"] = 8.2481e-8
solutes_paper["Ca"] = 0.01075
solutes_paper["Cl"] = 0.56920
solutes_paper["K"] = 0.01058
solutes_paper["Mg"] = 0.05474
solutes_paper["Na"] = 0.48618
solutes_paper["SO4"] = 0.02927
solutes_paper["OH"] = 5.0620e-8
solutes_paper["HSO4"] = 1.3286e-8
solutes_paper["MgOH"] = 4.5149e-8
params = prmlib.get_parameters(solutes_paper, temperature=273.15, verbose=False)
# params = pz.libraries.GM89.get_parameters(solutes_paper, temperature=273.15, verbose=False)

osm_paper = pz.osmotic_coefficient(solutes_paper, **params)
aw_paper = pz.activity_water(solutes_paper, **params)
acfs_paper = pz.activity_coefficients(solutes_paper, **params)
# print(osm_paper, aw_paper)

mols = np.vstack([v for v in solutes_paper.values()])
ions = np.array([k for k in solutes_paper.keys()])
prmlib_v4 = pz4.libraries.GM89
prmlib_v4.add_zeros(ions)
acfs_paper_v4 = pz4.model.acfs(mols, ions, 273.15, 10.10325, prmlib=prmlib_v4).ravel()

Gex_v4 = pz4.model.Gex_nRT(mols, ions, 273.15, 10.10325, prmlib=prmlib_v4)[0]
Gex_v5 = pz.Gibbs_nRT(solutes_paper, **params).item()

# for m, i in zip(acfs_paper_v4, ions):
#     print("{}: {} vs {}".format(i, acfs_paper[i], m))
