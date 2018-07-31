import numpy as np
import pandas as pd
import pickle
from scipy.io import savemat
import pytzer as pz
pd2vs = pz.misc.pd2vs
from mvdh import ismember

# Load raw datasets
datapath = 'datasets/'
vplbase,mols,ions,T = pz.data.vpl(datapath)

# Select electrolytes for analysis
vplbase,mols,ions,T = pz.data.subset_ele(vplbase,mols,ions,T,
                                         np.array(['NaCl']))

## Exclude smoothed datasets (none for VPL)
#S = vplbase.smooth == 0
#vplbase = vplbase[S]
#mols    = mols   [S]

# Exclude datasets with T > 373.15 (i.e. my D-H functions are out of range)
Tx = vplbase.t <= 373.15
# Also, take only data at 298.15 K
Tx = np.logical_and(Tx,vplbase.t == 298.15)
vplbase = vplbase[Tx]
mols    = mols   [Tx]
T       = T      [Tx]

# Create initial electrolytes pivot table
vple = pd.pivot_table(vplbase,
                      values  = ['m'  ],
                      index   = ['ele'],
                      aggfunc = [np.min,np.max,len])

# Prepare model cdict
cf = pz.cdicts.MPH
eles = vplbase.ele
cf.add_zeros(vplbase.ele)

# Extract metadata from vplbase
tot  = pd2vs(vplbase.m  )
srcs = pd2vs(vplbase.src)
_,zC,zA,nC,nA = pz.data.znu(vple.index)

# Convert measured VPL into osmotic coeff. at 298.15 K
vplbase['osm_meas'] = pz.model.aw2osm(mols,pd2vs(vplbase.aw))
vplbase['osm_calc'] = pz.model.osm(mols,ions,T,cf)
vplbase['dosm'] = vplbase.osm_meas - vplbase.osm_calc

# Load outputs from simpytz_vpl.py
with open('pickles/simpar_vpl.pkl','rb') as f:
    _,vplerr_rdm,vplerr_sys = pickle.load(f)
    
#%% Simulate new datasets
sele = 'NaCl'
Ureps = int(20)

# Set up for fitting
alph1 = np.float_(2)
alph2 = -9
omega = np.float_(2.5)

# Define weights for fitting
#weights = np.ones(np.size(T1)) # uniform
#weights = np.sqrt(tot) # sqrt of molality
# ... based on random errors in each dataset:
weights = np.full_like(tot,1, dtype='float64')
for src in np.unique(srcs):
    SL = srcs == src
    weights[SL] = 1 / np.sqrt(np.sum(vplerr_rdm[sele][src]**2))
weights = weights

def Eopt(rseed=None):

    # Seed random numbers
    np.random.seed(rseed)

    # Simulate new VPL dataset
    Uosm = pz.sim.vpl(tot,pd2vs(vplbase.osm_calc),
                        srcs,sele,vplerr_rdm,vplerr_sys)

    # Solve for Pitzer model coefficients
    b0,b1,_,C0,C1,_,_ \
        = pz.fitting.bC(mols,zC,zA,T,alph1,alph2,omega,nC,nA,Uosm,
                        weights,'b0b1C0C1','osm')

    return Uosm,b0,b1,C0,C1

osm_sim = np.full((np.size(tot),Ureps),np.nan)
b0 = np.full(Ureps,np.nan)
b1 = np.full(Ureps,np.nan)
C0 = np.full(Ureps,np.nan)
C1 = np.full(Ureps,np.nan)

sqtot_fitted = np.vstack(np.linspace(0.001,2.5,100))
tot_fitted   = sqtot_fitted**2
mols_fitted  = np.concatenate((tot_fitted,tot_fitted), axis=1)
osm_fitted   = np.full((np.size(tot_fitted),Ureps),np.nan)
T1_fitted    = np.full_like(tot_fitted,298.15, dtype='float64')

for U in range(Ureps):
    print(U+1)
    Uosm_sim,Ub0,Ub1,UC0,UC1 = Eopt()
    osm_sim[:,U] = Uosm_sim.ravel()
    b0[U] = Ub0
    b1[U] = Ub1
    C0[U] = UC0
    C1[U] = UC1
    osm_fitted[:,U] = pz.fitting.osm(mols_fitted,zC,zA,
                                     T1_fitted,b0[U],b1[U],0,C0[U],C1[U],
                                     alph1,-9,omega).ravel()
    
osm_fitted_calc = pz.model.osm(mols_fitted,ions,T1_fitted,cf)
    
# Save results for MATLAB
vplbase.to_csv('pickles/simloop_vpl_test.csv')
savemat('pickles/simloop_vpl_test.mat',
        {'osm_sim'         : osm_sim,
         'tot_fitted'      : tot_fitted,
         'osm_fitted'      : osm_fitted,
         'osm_fitted_calc' : osm_fitted_calc})

## Quick results viz
#from matplotlib import pyplot as plt
#fig,ax = plt.subplots(1,1)
#ax.scatter(vplbase.m,vpl_sim-vpl_calc)
