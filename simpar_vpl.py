# Import libraries
from autograd import numpy as np
import pandas as pd
from scipy import optimize
from scipy.io import savemat
import pickle
import pytzer as pz
pd2vs = pz.misc.pd2vs
from mvdh import ismember

# Set whether to allow uniform systematic offset
USYS = np.float_(1) # 0 for no, 1 for yes

# Load raw datasets
datapath = 'datasets/'
vplbase,mols,ions,T = pz.data.vpl(datapath)

# Select electrolytes for analysis
vplbase,mols,ions,T = pz.data.subset_ele(vplbase,mols,ions,T,
                                         np.array(['NaCl',
                                                   'KCl',
                                                   'CaCl2']))#,
                                                   #MgCl2']))

## Exclude smoothed datasets (none for VPL)
#S = vplbase.smooth == 0
#vplbase = vplbase[S]
#mols    = mols   [S]
                                                   
# Exclude datasets with T > 373.15 (i.e. my D-H functions are out of range)
Tx = vplbase.t <= 373.15
vplbase = vplbase[Tx]
mols    = mols   [Tx]
T       = T      [Tx]

# Prepare model cdict
cf = pz.cdicts.MPH
eles = vplbase.ele
cf.add_zeros(vplbase.ele)

## Calculate osmotic coefficient at measurement temperature
#vplbase['osm_meas'] = -np.log(vplbase.aw) / (vplbase.nu * vplbase.m * Mw)
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
            syserr[1] * vplbase[SL].m + USYS * syserr[0] - vplbase[SL].dosm25,
                                             [0.,0.])['x']
        
        if USYS == 1:
            if (sum(SL) < 6) or (max(vplbase[SL].m) - min(vplbase[SL].m) < 2):
                vplerr_sys[ele][src][1] = 0
                vplerr_sys[ele][src][0] = optimize.least_squares(
                    lambda syserr: syserr - vplbase[SL].dosm25,0.)['x'][0]

        vplbase.loc[SL,'dosm25_sys'] \
            =  vplbase.dosm25[SL] \
            - (vplbase.m[SL] * vplerr_sys[ele][src][1] \
               +               vplerr_sys[ele][src][0])
                       
        # Evaluate random component of error
        vplerr_rdm[ele][src] = optimize.least_squares(lambda rdmerr: \
            rdmerr[1] * np.exp(-vplbase[SL].m) + rdmerr[0] \
            - np.abs(vplbase[SL].dosm25_sys), [0,0])['x']
        
#        if vplerr_rdm[ele][src][0] < 0:
#            vplerr_rdm[ele][src][0] = 0
#            vplerr_rdm[ele][src][1] = optimize.least_squares(lambda rdmerr: \
#                rdmerr * vplbase[SL].m \
#                - np.abs(vplbase[SL].dosm25_sys), 0.)['x']
        
        if (sum(SL) < 6) or (vplerr_rdm[ele][src][1] < 0):
            vplerr_rdm[ele][src][1] = 0
            vplerr_rdm[ele][src][0] = optimize.least_squares(lambda rdmerr: \
                rdmerr - np.abs(vplbase[SL].dosm25_sys), 0.)['x'][0]
         
# Add 'all' fields for easier plotting in MATLAB
for ele in vplp.index.levels[0]:
    Eksys = list(vplerr_sys[ele].keys())
    vplerr_sys[ele]['all_int'] = np.array( \
        [vplerr_sys[ele][src][0] for src in Eksys])
    vplerr_sys[ele]['all_grad'] = np.array( \
        [vplerr_sys[ele][src][1] for src in Eksys])
    Ekrdm = list(vplerr_rdm[ele].keys())
    vplerr_rdm[ele]['all_int'] = np.array( \
        [vplerr_rdm[ele][src][0] for src in Ekrdm])
    vplerr_rdm[ele]['all_grad'] = np.array( \
        [vplerr_rdm[ele][src][1] for src in Ekrdm])

# Pickle outputs for simloop
with open('pickles/simpar_vpl.pkl','wb') as f:
    pickle.dump((vplbase,vplerr_rdm,vplerr_sys),f)

# Save results for MATLAB figures
vplbase.to_csv('pickles/simpar_vpl.csv')
savemat('pickles/simpar_vpl.mat',{'vplerr_sys':vplerr_sys,
                                  'vplerr_rdm':vplerr_rdm})

print('VPL fit optimisation complete!')
