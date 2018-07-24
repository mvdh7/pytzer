# Import libraries
from autograd import numpy as np
import pandas as pd
from scipy import optimize
from scipy.io import savemat
import pytzer as pz
pd2vs = pz.misc.pd2vs
from mvdh import ismember

# Load raw datasets
datapath = 'datasets/'
fpdbase,mols,ions = pz.data.fpd(datapath)

# Select electrolytes for analysis
fpdbase,mols,ions = pz.data.subset_ele(fpdbase,mols,ions,
                                       np.array(['NaCl',
                                                 'KCl',
                                                 'CaCl2',
                                                 'MgCl2']))

# Exclude smoothed datasets
S = fpdbase.smooth == 0
fpdbase = fpdbase[S]
mols    = mols   [S]

# Prepare model cdict
cf = pz.cdicts.cdict()
eles = fpdbase.ele
cf.add_zeros(fpdbase.ele)

cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
cf.bC['K-Cl' ] = pz.coeffs.bC_K_Cl_A99
cf.bC['Ca-Cl'] = pz.coeffs.bC_Ca_Cl_GM89
cf.bC['Mg-Cl'] = pz.coeffs.bC_Mg_Cl_PP87i
cf.dh['Aosm']  = pz.coeffs.Aosm_MPH
cf.dh['AH']    = pz.coeffs.AH_MPH
cf.jfunc = pz.jfuncs.P75_eq47

# Calculate osmotic coefficient at measurement temperature
T   = pd2vs(fpdbase.t  )
fpd = pd2vs(fpdbase.fpd)
fpdbase['osm_meas'] = pz.tconv.fpd2osm(mols,fpd)
fpdbase['osm_calc'] = pz.model.osm(mols,ions,T,cf)

# Convert temperatures to 298.15 K
fpdbase['t25'] = np.full_like(fpdbase.t,298.15, dtype='float64')
T25 = pd2vs(fpdbase.t25)

# Create initial electrolytes pivot table
fpde = pd.pivot_table(fpdbase,
                      values  = ['m'],
                      index   = ['ele'],
                      aggfunc = [np.min,np.max,len])

# Convert measured FPD into osmotic coeff. at 298.15 K
fpdbase['osm25_meas'] = np.nan

for ele in fpde.index:
    
    Efpdbase,_,Eions = pz.data.subset_ele(fpdbase,mols,ions,np.array([ele]))
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
            syserr[1] * fpdbase[SL].m + syserr[0] - fpdbase[SL].dfpd,
                                             [0.,0.])['x']
        
        if sum(SL) < 6:
            fpderr_sys[ele][src][1] = 0
            fpderr_sys[ele][src][0] = optimize.least_squares(lambda syserr: \
                syserr - fpdbase[SL].dfpd,0.)['x'][0]

        fpdbase.loc[SL,'dfpd_sys'] = fpdbase.dfpd[SL] \
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

# Save results for MATLAB figures
fpdbase.to_csv('pickles/simpytz_fpd.csv')
savemat('pickles/simpytz_fpd.mat',{'fpderr_sys':fpderr_sys,
                                   'fpderr_rdm':fpderr_rdm})

print('FPD fit optimisation complete!')
