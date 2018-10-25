# Import libraries
from autograd import numpy as np
import pandas as pd
from scipy import optimize
from scipy.io import savemat
import pickle
import pytzer as pz
from pytzer.misc import ismember, pd2vs

# Load raw datasets
datapath = 'datasets/'
vplbase,mols,ions,T = pz.data.vpl(datapath)

# Select electrolytes for analysis
vplbase,mols,ions,T = pz.data.subset_ele(vplbase,mols,ions,T,
                                         np.array(['NaCl',
                                                   'KCl',
                                                   'CaCl2']))
                                                   
## Exclude datasets with T > 373.15 (i.e. my D-H functions are out of range)
#Tx = vplbase.t <= 373.15
#vplbase = vplbase[Tx]
#mols    = mols   [Tx]
#T       = T      [Tx]

# Select only datasets at T = 298.15
Tx = vplbase.t == 298.15
vplbase = vplbase[Tx]
mols    = mols   [Tx]
T       = T      [Tx]

# Prepare model cdict
cf = pz.cdicts.MPH
eles = vplbase.ele
cf.add_zeros(eles)

## Calculate osmotic coefficient at measurement temperature
vplbase['osm_meas'] = pz.model.aw2osm(mols,pd2vs(vplbase.aw))
vplbase['osm_calc'] = pz.model.osm(mols,ions,T,cf)

# Convert temperatures to 298.15 K
vplbase['t25'] = 298.15
T25 = pd2vs(vplbase.t25)

# Create initial electrolytes pivot table
vple = pd.pivot_table(vplbase,
                      values  = ['m'],
                      index   = ['ele'],
                      aggfunc = [np.min,np.max,len])

# Convert measured VPL into osmotic coeff. at 298.15 K
vplbase['osm25_meas'] = np.nan

for ele in vple.index:
    
    Evplbase,_,Eions,_ = pz.data.subset_ele(vplbase,mols,ions,T,
                                            np.array([ele]))
    EL = ismember(vplbase.ele,np.array([ele]))
    
    vplbase.loc[EL,'osm25_meas'] = pz.tconv.osm2osm(
            pd2vs(Evplbase.m),pd2vs(Evplbase.nC),pd2vs(Evplbase.nA),
            Eions,pd2vs(Evplbase.t),pd2vs(Evplbase.t25),pd2vs(Evplbase.t25),
            cf,pd2vs(Evplbase.osm_meas))

# Calculate model osmotic coefficient at 298.15 K
vplbase['osm25_calc'] = pz.model.osm(mols,ions,T25,cf)

# Use PCHIP interpolation to get calculated osm25 for CaCl2
L = vplbase.ele == 'CaCl2'
vplbase.loc[L,'osm25_calc'] = pz.isoref.osm_CaCl2(vplbase.m[L])

L1 = np.logical_and(L,vplbase.t == 298.15)
vplbase.loc[L1,'osm_calc'] = pz.isoref.osm_CaCl2(vplbase.m[L1])

L2 = np.logical_and(L,vplbase.t != 298.15)
vplbase.loc[L2,'osm_calc'] = np.nan

# Calculate differences
vplbase['dosm'  ] = vplbase.osm_meas   - vplbase.osm_calc
vplbase['dosm25'] = vplbase.osm25_meas - vplbase.osm25_calc

## Quickly visualise residuals
#from matplotlib import pyplot as plt
#fig,ax = plt.subplots(1,1)
#vplbase.plot.scatter('m','dosm25', ax=ax)
#ax.set_xlim((0,6))
#ax.grid(alpha=0.5)

# Create electrolytes/sources pivot table
vplp = pd.pivot_table(vplbase,
                      values  = ['m','dosm25'],
                      index   = ['ele','src'],
                      aggfunc = [np.mean,np.std,len])

## Export vplbase to MATLAB
#vplbase.to_csv('pickles/simpar_vpl.csv')

#%% Run uncertainty propagation analysis [VPL]
tot = pd2vs(vplbase.m)
_,_,_,nC,nA = pz.data.znu(vplp.index.levels[0])
vplbase['dosm25_sys'] = np.nan
vplerr_sys = {}
vplerr_rdm = {}

for E,ele in enumerate(vplp.index.levels[0]):
    print('Optimising VPL fit for ' + ele + '...')
    
    EL = vplbase.ele == ele
    
    # Estimate uncertainties for each source
    vplerr_sys[ele] = {}
    vplerr_rdm[ele] = {}
    
    for src in vplp.loc[ele].index:
        
        SL = np.logical_and(EL,vplbase.src == src)
        SL = np.logical_and(SL,vplbase.t == 298.15)
                    
        # Evaluate systematic component of error
        vplerr_sys[ele][src] = optimize.least_squares(lambda syserr: \
            pz.experi.vplfit_sys(syserr,vplbase.m[SL]) \
            - vplbase.dosm[SL],0)['x']
        
        # Normalise residuals
        vplbase.loc[SL,'dosm25_sys'] = vplbase.dosm25[SL] \
            - pz.experi.vplfit_sys(vplerr_sys[ele][src],vplbase.m[SL])
        
        # Evaluate random component of error
        vplerr_rdm[ele][src] = optimize.least_squares(lambda rdmerr: \
            pz.experi.vplfit_rdm(rdmerr,vplbase.m[SL]) \
            - np.abs(vplbase.dosm25_sys[SL]), [0,0])['x']
        
        # Correct poor fits
        if (sum(SL) < 6) or (vplerr_rdm[ele][src][1] < 0):
            vplerr_rdm[ele][src][0] = np.mean(np.abs(vplbase[SL].dosm25_sys))
            vplerr_rdm[ele][src][1] = 0
         
# Add 'all' fields
for ele in vplp.index.levels[0]:
    Eksys = list(vplerr_sys[ele].keys())
    vplerr_sys[ele]['all'] = np.array( \
        [vplerr_sys[ele][src] for src in Eksys]).ravel()

vplerr_sys['all'] = np.concatenate([vplerr_sys[ele]['all'] \
                                    for ele in vplp.index.levels[0]])

# Calculate systematic simulation coefficient
vplerr_sys['sd_Sn'] = pz.misc.Sn(vplerr_sys['all'])
    
#%% Pickle outputs for simloop
with open('pickles/simpar_vpl.pkl','wb') as f:
    pickle.dump((vplbase,mols,ions,T,vplerr_sys,vplerr_rdm),f)

# Save results for MATLAB figures
vplbase.to_csv('pickles/simpar_vpl.csv')
savemat('pickles/simpar_vpl.mat',{'vplerr_sys':vplerr_sys,
                                  'vplerr_rdm':vplerr_rdm})

print('VPL fit optimisation complete!')
