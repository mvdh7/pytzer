from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytzer as pz

filepath = 'E:\Dropbox\_UEA_MPH\pytzer-support\datasets\sol'
r81 = pd.read_excel('{}.xlsx'.format(filepath))
r81['sqCa'] = np.sqrt(r81.Ca)
r81vars = mols, ions, tempK, pres = pz.io.getmols('{}.csv'.format(filepath))
zs = pz.properties.charges(ions)[0]
r81['ionstr'] = pz.model.Istr(mols, zs)
r81['sqionstr'] = np.sqrt(r81.ionstr)
r81['aw_calc'] = pz.model.aw(*r81vars)
r81['delaw'] = r81.aw - r81.aw_calc
lnacfs = pz.model.ln_acfs(*r81vars)
r81['lnacfCa'] = lnacfs[ions == 'Ca'][0]
r81['lnacfSO4'] = lnacfs[ions == 'SO4'][0]
r81['acfCaSO4'] = np.exp(pz.model.ln_acf2ln_acf_MX(
    r81.lnacfCa, r81.lnacfSO4, 2, 2))
r81['delCaSO4'] = r81.expCaSO4 - r81.acfCaSO4
r81['osm_calc'] = pz.model.osm(*r81vars)
r81['osm_meas'] = pz.model.aw2osm(mols, r81.aw)
r81['delosm'] = r81.osm_meas - r81.osm_calc

fxvar = 'sqionstr'
fyvars = ['delaw', 'delCaSO4', 'delosm']
fig, ax = plt.subplots(len(fyvars), 1)
for f, fyvar in enumerate(fyvars):
    r81.plot.scatter(fxvar, fyvar, ax=ax[f])
    ax[f].set_ylim(np.max(np.abs(r81[fyvar]))*np.array([-1, 1])*1.1)
    ax[f].set_xlim([0, np.max(r81[fxvar])*1.05])
    ax[f].plot([0, np.max(r81[fxvar])*1.05], [0, 0], c='k')
