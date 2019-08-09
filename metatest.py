from matplotlib import pyplot as plt
import pytzer as pz

ele = 'NaCl'
varout = 'osm'

fig, ax = plt.subplots(1, 3)
pz.plot.bC_tempK(ax[0], 0.7, ele, 273.15, 323.15, 10.10325, varout=varout)
pz.plot.bC_pres(ax[1], 0.7, ele, 298.15, 0., 6000., varout=varout)
pz.plot.bC_tot(ax[2], 1e-10, 6.25, ele, 298.15, 0., varout=varout)
