from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytzer as pz

# Import PM18 model calculations
pm18 = pd.read_excel('testfiles/pm18model_noeq.xlsx').dropna()
ions = [pm18.ion[i] for i in pm18.index if pm18.type[i] == 'acf']
sets = [col for col in pm18.columns if col.startswith('set')]
mols = np.zeros((len(ions), len(sets)))
molions = [pm18.ion[i] for i in pm18.index if pm18.type[i] == 'mol']
for i, ion in enumerate(ions):
    if ion in molions:
        mols[i] = pm18.loc[np.logical_and(pm18.ion == ion, pm18.type == 'mol'),
            sets].values.ravel()
tempK = pm18.loc[pm18.ion == 'tempK', sets].values.ravel()
pres = np.full_like(tempK, 10.10325)
acfs_pm18 = pm18.loc[pm18.type == 'acf', sets].values

#%% Re-calculate activity coefficients with Pytzer
badMiami = deepcopy(pz.libraries.MIAMI)
badMiami.add_zeros(ions)
acfs = pz.model.acfs(mols, ions, tempK, pres, prmlib=badMiami)
dacfs = acfs - acfs_pm18
dmax = np.max(np.abs(dacfs))
dmax = 0.1

##%% Compare results
#fig, ax = plt.subplots(2, 1)
plt.clf()
im = plt.imshow(np.transpose(dacfs), cmap='RdBu', vmin=-dmax, vmax=dmax)
cb = plt.colorbar(im)
cb.ax.set_ylabel('$\gamma$(Pytzer)$ - \gamma$(PM18)')
plt.xticks(range(len(ions)), ions, rotation=-90);
plt.yticks(range(len(tempK)));
plt.ylabel('Set');

##%% Get list of active ions in given set
#fset = 12
#activeions = [ion for i, ion in enumerate(ions) if mols[i, fset] > 0]
