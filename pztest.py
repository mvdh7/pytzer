#from copy import deepcopy
import pytzer as pz
import numpy as np 

# Import test dataset
filepath = 'testfiles/'
filestem = 'GenerateConcs'

# Run black box calculation (uses MarChemSpec cfdict)
from time import time
from copy import deepcopy

cfltest = deepcopy(pz.cflibs.MarChemSpecPres)
#cfltest.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
#cfltest.bC['K-Cl'] = pz.coeffs.bC_K_Cl_ZD17

go = time()
mols, ions, T, pres, cf, Gex_nRT, osm, aw, acfs \
    = pz.blackbox(filepath + filestem + '.csv',
                  cflib=cfltest)
print(time() - go)

mols = np.transpose(mols)
osm  = np.vstack(osm)
aw   = np.vstack(aw)
acfs = np.transpose(acfs)

#%% Load results from Julia
jfilename = filepath + filestem + '_jl.csv'
jdata = np.genfromtxt(jfilename, delimiter=',', skip_header=1)
jhead = np.genfromtxt(jfilename, delimiter=',', skip_footer=np.shape(jdata)[0],
                      dtype='str')
                  
gions = [''.join(('g',ion)) for ion in ions]    
jacfs = np.concatenate([np.vstack(jdata[:,C]) 
    for C in range(np.shape(jdata)[1])
    if jhead[C] in gions],
    axis=1)

josm = np.vstack(jdata[:,jhead == 'osm'])
jaw  = np.vstack(jdata[:,jhead == 'aw' ])

# Compare and plot differences
dacfs = jacfs - acfs
dosm  = josm  - osm
daw   = jaw   - aw

from matplotlib import pyplot as plt

fig,ax = plt.subplots(1,3, gridspec_kw = {'width_ratios':[3, 1, 1]})

# Solvent activities
aabsmax = np.max(np.abs(dacfs))
cax = ax[0].imshow(dacfs, cmap='coolwarm', aspect='auto',
                   vmin=-aabsmax, vmax=aabsmax)
fig.colorbar(cax, ax=ax[0], label='Diff. in acf')

ax[0].set_xticks(np.arange(len(ions)))
ax[0].set_xticklabels(ions)

ax[0].set_ylabel('Row in file')

# Osmotic coefficients
oabsmax = np.max(np.abs(dosm))
rminmax = np.array([-0.5,len(dosm)-0.5])

ax[1].barh(np.arange(len(dosm)),dosm.ravel(),1, color=0.5*np.array([1,1,1]))

ax[1].plot(np.array([0,0]),rminmax, c='k')

ax[1].set_xlim(oabsmax*np.array([-1,1])*1.1)
ax[1].set_ylim(rminmax)

ax[1].set_xlabel('Diff. in osm')
ax[1].set_ylabel('Row in file')

ax[1].invert_yaxis()

# Water activity
wabsmax = np.max(np.abs(daw))
rminmax = np.array([-0.5,len(daw)-0.5])

ax[2].barh(np.arange(len(daw)),daw.ravel(),1, color=0.5*np.array([1,1,1]))

ax[2].plot(np.array([0,0]),rminmax, c='k')

ax[2].set_xlim(wabsmax*np.array([-1,1])*1.1)
ax[2].set_ylim(rminmax)

ax[2].set_xlabel('Diff. in aw')
ax[2].set_ylabel('Row in file')

ax[2].invert_yaxis()
