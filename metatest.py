from autograd import numpy as np
from matplotlib import pyplot as plt
import pytzer as pz

ifuncs = pz.meta.getifuncs("bC", ["Mg", "HCO3"])
ivals = pz.meta.evalifuncs(ifuncs, 298.15, 10.10325)

##%% Beta and C coefficients
# ele = 'MB(OH)4'
# varout = 'osm'
# fig, ax = plt.subplots(1, 3)
# pz.plot.bC_tempK(ax[0], 0.7, ele, 273.15, 323.15, 10.10325, varout=varout)
# pz.plot.bC_pres(ax[1], 0.7, ele, 298.15, 0., 6000., varout=varout)
# pz.plot.bC_tot(ax[2], 1e-10, 6.25, ele, 298.15, 0., varout=varout)

##%% Variable fixed-ratio composition, no equilibration
# ions = np.array(['Na', 'K', 'Ca', 'Mg', 'Cl', 'SO4', 'Li', 'F', 'H', 'tris'])
# mols_fixed = np.array([1.0, 0.5, 0.25, 1.0, 2.0, 1.0, 0.1, 0.2, 0.1, 1.0])
#
# ratios = np.linspace(0.1, 2, 50)
# mols = np.vstack(mols_fixed)*ratios
# tempK = np.full_like(mols[0], 323.15)
# pres = np.full_like(mols[0], 1000.10325)
#
# prmlibs = [
#    pz.libraries.M88,
#    pz.libraries.GM89,
#    pz.libraries.HMW84,
#    pz.libraries.WM13,
#    pz.libraries.MIAMI,
# ]
# for prmlib in prmlibs:
#    prmlib.add_zeros(ions)
#
# osms = np.vstack([pz.model.aw(mols, ions, tempK, pres, prmlib=prmlib)
#    for prmlib in prmlibs])
#
# fix, ax = plt.subplots()
# for p in range(len(prmlibs)):
#    ax.plot(ratios, osms[p])
