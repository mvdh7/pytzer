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
fpdbase,mols,ions,T = pz.data.fpd(datapath)

# Select electrolytes for analysis
fpdbase,mols,ions,T = pz.data.subset_ele(fpdbase,mols,ions,T,
                                         np.array(['NaCl',
                                                   'KCl',
                                                   'CaCl2']))#,
                                                   #MgCl2']))

#%% Exclude smoothed datasets
S = fpdbase.smooth == 0
fpdbase = fpdbase[S]
mols    = mols   [S]
T       = T      [S]

# Prepare model cdict
cf = pz.cdicts.MPH
eles = fpdbase.ele
cf.add_zeros(fpdbase.ele)

# Calculate osmotic coefficient at measurement temperature
fpd = pd2vs(fpdbase.fpd)
fpdbase['osm_meas'] = pz.tconv.fpd2osm(mols,fpd)
fpdbase['osm_calc'] = pz.model.osm(mols,ions,T,cf)

# Convert temperatures to 298.15 K
fpdbase['t25'] = 298.15
T25 = pd2vs(fpdbase.t25)

# Create initial electrolytes pivot table
fpde = pd.pivot_table(fpdbase,
                      values  = ['m'],
                      index   = ['ele'],
                      aggfunc = [np.min,np.max,len])

# Convert measured FPD into osmotic coeff. at 298.15 K
fpdbase['osm25_meas'] = np.nan

for ele in fpde.index:
    
    Efpdbase,_,Eions,_ = pz.data.subset_ele(fpdbase,mols,ions,T,
                                            np.array([ele]))
    EL = ismember(fpdbase.ele,np.array([ele]))
    
    fpdbase.loc[EL,'osm25_meas'] = pz.tconv.osm2osm(
            pd2vs(Efpdbase.m),pd2vs(Efpdbase.nC),pd2vs(Efpdbase.nA),
            Eions,pd2vs(Efpdbase.t),pd2vs(Efpdbase.t25),pd2vs(Efpdbase.t25),
            cf,pd2vs(Efpdbase.osm_meas))

# Calculate model osmotic coefficient at 298.15 K
fpdbase['osm25_calc'] = pz.model.osm(mols,ions,T25,cf)
fpdbase['dosm'  ] = fpdbase.osm_meas   - fpdbase.osm_calc
fpdbase['dosm25'] = fpdbase.osm25_meas - fpdbase.osm25_calc

## Quickly visualise residuals
#from matplotlib import pyplot as plt
#fig,ax = plt.subplots(1,1)
#fpdbase.plot.scatter('m','dosm25', ax=ax)
#ax.set_xlim((0,6))
#ax.grid(alpha=0.5)

# Create electrolytes/sources pivot table
fpdp = pd.pivot_table(fpdbase,
                      values  = ['m','dosm25'],
                      index   = ['ele','src'],
                      aggfunc = [np.mean,np.std,len])

#%% Run uncertainty propagation analysis [FPD]
tot = pd2vs(fpdbase.m)
_,_,_,nC,nA = pz.data.znu(fpdp.index.levels[0])
fpdbase['fpd_calc'] = np.nan
fpdbase['dfpd'    ] = np.nan
fpdbase['dfpd_sys'] = np.nan
fpderr_sys = {}
fpderr_rdm = {}

for E,ele in enumerate(fpdp.index.levels[0]): 
    print('Optimising FPD fit for ' + ele + '...')
    
    # Calculate expected FPD
    Eions = pz.data.ele2ions(np.array([ele]))[0]
    EL = fpdbase.ele == ele
    
    if ele == 'CaCl2':
        fpdbase.loc[EL,'fpd_calc'] = pz.tconv.tot2fpd25(tot[EL],
                                                        Eions,
                                                        nC[E],
                                                        nA[E],
                                                        cf)
    else:
        fpdbase.loc[EL,'fpd_calc'] = pz.tconv.tot2fpd(tot[EL],
                                                      Eions,
                                                      nC[E],
                                                      nA[E],
                                                      cf)
    fpdbase.loc[EL,'dfpd'] = fpdbase.fpd[EL] - fpdbase.fpd_calc[EL]
    
    # Estimate uncertainties for each source
    fpderr_sys[ele] = {}
    fpderr_rdm[ele] = {}
    
    for src in fpdp.loc[ele].index:
        
        SL = np.logical_and(EL,fpdbase.src == src)
        if ele == 'CaCl2':
            SL = np.logical_and(SL,fpdbase.m <= 1.5)
        
        # Evaluate systematic component of error
        fpderr_sys[ele][src] = optimize.least_squares(lambda syserr: \
            syserr[1] * fpdbase[SL].m + USYS * syserr[0] - fpdbase[SL].dfpd,
                                             [0.,0.])['x']
        
        if USYS == 1:
            if sum(SL) < 6:
                fpderr_sys[ele][src][1] = 0
                fpderr_sys[ele][src][0] = optimize.least_squares(
                    lambda syserr: syserr - fpdbase[SL].dfpd,0.)['x'][0]

        fpdbase.loc[SL,'dfpd_sys'] \
            =  fpdbase.dfpd[SL] \
            - (fpdbase.m[SL] * fpderr_sys[ele][src][1] \
               +               fpderr_sys[ele][src][0])
                       
        # Evaluate random component of error
        fpderr_rdm[ele][src] = optimize.least_squares(lambda rdmerr: \
            rdmerr[1] * fpdbase[SL].m + rdmerr[0] \
            - np.abs(fpdbase[SL].dfpd_sys), [0,0])['x']
        
        if fpderr_rdm[ele][src][0] < 0:
            fpderr_rdm[ele][src][0] = 0
            fpderr_rdm[ele][src][1] = optimize.least_squares(lambda rdmerr: \
                rdmerr * fpdbase[SL].m \
                - np.abs(fpdbase[SL].dfpd_sys), 0.)['x']
        
        if (sum(SL) < 6) or (fpderr_rdm[ele][src][1] < 0):
            fpderr_rdm[ele][src][1] = 0
            fpderr_rdm[ele][src][0] = optimize.least_squares(lambda rdmerr: \
                rdmerr - np.abs(fpdbase[SL].dfpd_sys), 0.)['x'][0]
         
# Add 'all' fields for easier plotting in MATLAB
for ele in fpdp.index.levels[0]:
    Eksys = list(fpderr_sys[ele].keys())
    fpderr_sys[ele]['all_int'] = np.array( \
        [fpderr_sys[ele][src][0] for src in Eksys])
    fpderr_sys[ele]['all_grad'] = np.array( \
        [fpderr_sys[ele][src][1] for src in Eksys])
    Ekrdm = list(fpderr_rdm[ele].keys())
    fpderr_rdm[ele]['all_int'] = np.array( \
        [fpderr_rdm[ele][src][0] for src in Ekrdm])
    fpderr_rdm[ele]['all_grad'] = np.array( \
        [fpderr_rdm[ele][src][1] for src in Ekrdm])

# Pickle outputs for simloop
with open('pickles/simpar_fpd.pkl','wb') as f:
    pickle.dump((fpdbase,fpderr_rdm,fpderr_sys),f)

# Save results for MATLAB figures
fpdbase.to_csv('pickles/simpar_fpd.csv')
savemat('pickles/simpar_fpd.mat',{'fpderr_sys':fpderr_sys,
                                  'fpderr_rdm':fpderr_rdm})

print('FPD fit optimisation complete!')
