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
fpdbase,mols,ions,T = pz.data.fpd(datapath)

# Select electrolytes for analysis
fpdbase,mols,ions,T = pz.data.subset_ele(fpdbase,mols,ions,T,
                                         np.array(['NaCl',
                                                   'KCl',
                                                   'CaCl2']))

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
CL = fpdbase.ele == 'CaCl2'
fpdbase.loc[CL,'osm_calc'] = np.nan

#%% Convert temperatures to 298.15 K
fpdbase['t25'] = 298.15
T25 = pd2vs(fpdbase.t25)

# Create initial electrolytes pivot table
fpde = pd.pivot_table(fpdbase,
                      values  = ['m'],
                      index   = ['ele'],
                      aggfunc = [np.min,np.max,len])

#%% Convert measured FPD into osmotic coeff. at 298.15 K
fpdbase['osm25_meas'] = np.nan

for ele in fpde.index:
    
    Efpdbase,_,Eions,_ = pz.data.subset_ele(fpdbase,mols,ions,T,
                                            np.array([ele]))
    EL = ismember(fpdbase.ele,np.array([ele]))
    
    if ele == 'CaCl2':
    
        fpdbase.loc[EL,'osm25_meas'] = pz.isoref.osm2osm25_CaCl2(
                pd2vs(Efpdbase.m),pd2vs(Efpdbase.t),pd2vs(Efpdbase.osm_meas))
        
    else:
        
        fpdbase.loc[EL,'osm25_meas'] = pz.tconv.osm2osm(
            pd2vs(Efpdbase.m),pd2vs(Efpdbase.nC),pd2vs(Efpdbase.nA),
            Eions,pd2vs(Efpdbase.t),pd2vs(Efpdbase.t25),pd2vs(Efpdbase.t25),
            cf,pd2vs(Efpdbase.osm_meas))

# Calculate model osmotic coefficient at 298.15 K and residuals
fpdbase['osm25_calc'] = pz.model.osm(mols,ions,T25,cf)
fpdbase.loc[CL,'osm25_calc'] = pz.isoref.osm_CaCl2(pd2vs(fpdbase.m[CL]))
fpdbase['dosm'  ] = fpdbase.osm_meas   - fpdbase.osm_calc
fpdbase['dosm25'] = fpdbase.osm25_meas - fpdbase.osm25_calc

# Create electrolytes/sources pivot table
fpdp = pd.pivot_table(fpdbase,
                      values  = ['m','dosm25'],
                      index   = ['ele','src'],
                      aggfunc = [np.mean,np.std,len])

#%% Solve for FPD expected from model (SLOW - a couple of minutes)
#tot = pd2vs(fpdbase.m)
#_,_,_,nC,nA = pz.data.znu(fpdp.index.levels[0])
#fpdbase['fpd_calc'] = np.nan
#
#for E,ele in enumerate(fpdp.index.levels[0]): 
#    print('Solving for model FPD for ' + ele + '...')
#
#    Eions = pz.data.ele2ions(np.array([ele]))[0]
#    EL = fpdbase.ele == ele
#    
#    if ele == 'CaCl2':
#        
#        fpdbase.loc[EL,'fpd_calc'] = pz.isoref.tot2fpd25_CaCl2(tot[EL])
#        
#    else:
#        
#        fpdbase.loc[EL,'fpd_calc'] = pz.tconv.tot2fpd25(tot[EL],
#                                                        Eions,
#                                                        nC[E],
#                                                        nA[E],
#                                                        cf)
#        
## Calculate residuals
#fpdbase['dfpd'] = fpdbase.fpd - fpdbase.fpd_calc
#    
## Save fpdbase for fast loading to skip slow step above in future
#fpdbase.to_csv('pickles/fpdbase_intermediate.csv')

# Load intermediate saved fpdbase [latest version from 2018-10-18]
fpdbase = pd.read_csv('pickles/fpdbase_intermediate.csv')

#%% Run uncertainty propagation analysis [FPD]
fpdbase['dosm25_sys'] = np.nan
fpderr_sys = {}
fpderr_rdm = {}

for E,ele in enumerate(fpdp.index.levels[0]): 
    print('Optimising FPD fit for ' + ele + '...')
    
    EL = fpdbase.ele == ele
    
    # Estimate uncertainties for each source
    fpderr_sys[ele] = {}
    fpderr_rdm[ele] = {}
    
    for src in fpdp.loc[ele].index:
        
        SL = np.logical_and(EL,fpdbase.src == src)
        SLx = np.copy(SL)
        if ele == 'CaCl2' and src == 'OBS90':
            SL = np.logical_and(SL,fpdbase.m <= 3.5)
        
        # Evaluate systematic component of error
        sysL = np.logical_and(SL,fpdbase.m >= 0.1)
        fpderr_sys[ele][src] = np.mean(fpdbase.dosm25[sysL])

        fpdbase.loc[SLx,'dosm25_sys'] \
            =  fpdbase.dosm25[SLx] - fpderr_sys[ele][src]
                       
        # Evaluate random component of error
        fpderr_rdm[ele][src] = optimize.least_squares(lambda rdmerr: \
            rdmerr[1] * np.exp(-fpdbase.m[SL]*rdmerr[2]) + rdmerr[0] \
            - np.abs(fpdbase.dosm25_sys[SL]),[0,0,0])['x']
        
        if fpderr_rdm[ele][src][1] < 0 or fpderr_rdm[ele][src][2] < 0:
            fpderr_rdm[ele][src] = np.zeros(3)
            fpderr_rdm[ele][src][0] = np.mean(np.abs(fpdbase.dosm25_sys[sysL]))
        
#        fpderr_rdm[ele][src] = optimize.least_squares(lambda rdmerr: \
#            rdmerr[1] / fpdbase.m[SL] + rdmerr[0] \
#            - np.abs(fpdbase.dosm25_sys[SL]),[0,0])['x']
         
#%% Add 'all' fields for easier plotting in MATLAB
for ele in fpde.index:
    fpderr_sys[ele]['all'] = np.array([fpderr_sys[ele][src] \
                                       for src in fpdp.loc[ele].index])

fpderr_sys['all'] = np.concatenate([fpderr_sys[ele]['all'] \
                                    for ele in fpde.index])
    
fpderr_sys['all_rms'] = pz.misc.rms(fpderr_sys['all'])
fpderr_sys['sd_Sn'] = pz.misc.Sn(fpderr_sys['all'])
    
# Pickle outputs for simloop
with open('pickles/simpar_fpd_osm25.pkl','wb') as f:
    pickle.dump((fpdbase,mols,ions,T,fpderr_sys,fpderr_rdm),f)

# Save results for MATLAB figures
fpdbase.to_csv('pickles/simpar_fpd_osm25.csv')
savemat('pickles/simpar_fpd_osm25.mat',{'fpderr_sys':fpderr_sys,
                                        'fpderr_rdm':fpderr_rdm})

print('FPD fit optimisation complete!')
