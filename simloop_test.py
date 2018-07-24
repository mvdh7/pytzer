import numpy as np
import pandas as pd
import pickle
from scipy.io import savemat
import pytzer as pz
pd2vs = pz.misc.pd2vs

# Load raw datasets
datapath = 'datasets/'
fpdbase,mols,ions = pz.data.fpd(datapath)

# Select electrolytes for analysis
fpdbase,mols,ions = pz.data.subset_ele(fpdbase,mols,ions,
                                       np.array(['NaCl']))

# Exclude smoothed datasets
S = fpdbase.smooth == 0
fpdbase = fpdbase[S]
mols    = mols   [S]

# Create initial electrolytes pivot table
fpde = pd.pivot_table(fpdbase,
                      values  = ['m'  ],
                      index   = ['ele'],
                      aggfunc = [np.min,np.max,len])

# Prepare model cdict
cf = pz.cdicts.cdict()
cf.add_zeros(fpdbase.ele)
cf.dh['Aosm'] = pz.coeffs.Aosm_MPH
cf.dh['AH'  ] = pz.coeffs.AH_MPH
cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii

# Extract metadata from fpdbase
tot  = pd2vs(fpdbase.m  )
srcs = pd2vs(fpdbase.src)
_,_,_,nC,nA = pz.data.znu(fpde.index)

# Prepare for simulation
for E,ele in enumerate(fpde.index):

    EL = fpdbase.ele == ele
    Eions = pz.data.ele2ions(np.array([ele]))[0]
    
    fpdbase['fpd_calc'] = pz.tconv.tot2fpd(tot[EL],Eions,nC[E],nA[E],cf)

fpd_calc = pd2vs(fpdbase.fpd_calc)

# Load outputs from simpytz_fpd.py
with open('pickles/simpytz_fpd.pkl','rb') as f:
    _,fpderr_rdm,fpderr_sys = pickle.load(f)
    
#%% Simulate new datasets
sele = 'NaCl'
Ureps = int(20)
fpd_sim = np.full((np.size(tot),Ureps),np.nan)
for U in range(Ureps):
    fpd_sim[:,U] = pz.sim.fpd(tot,pd2vs(fpdbase.fpd_calc),srcs,sele,
                              fpderr_rdm,fpderr_sys).ravel()

# Save results for MATLAB
fpdbase.to_csv('pickles/simloop_test.csv')
savemat('pickles/simloop_test.mat',{'fpd_sim':fpd_sim})

## Quick results viz
#from matplotlib import pyplot as plt
#fig,ax = plt.subplots(1,1)
#ax.scatter(fpdbase.m,fpd_sim-fpd_calc)

