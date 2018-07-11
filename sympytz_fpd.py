# Import libraries
from autograd import numpy as np
import pandas as pd
from scipy import optimize
import pickle
import pytzer as pz
from mvdh import ismember
pd2vs = pz.sim.pd2vs

# Load raw datasets
datapath = 'datasets/'
fpdbase,mols,ions = pz.data.fpd(datapath)

# Select electrolytes for analysis
fpdbase,mols,ions = pz.data.subset_ele(fpdbase,mols,ions,
                                       np.array(['NaCl','KCl']))

# Exclude smoothed datasets
S = fpdbase.smooth == 0
fpdbase = fpdbase[S]
mols    = mols   [S]

# Prepare model cdict
cf = pz.cdicts.cdict()
cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
cf.bC['K-Cl' ] = pz.coeffs.bC_K_Cl_GM89
cf.theta['K-Na'] = pz.coeffs.theta_zero
cf.psi['K-Na-Cl'] = pz.coeffs.psi_zero
cf.dh['Aosm']  = pz.coeffs.Aosm_M88
cf.dh['AH']    = pz.coeffs.AH_MPH

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

# Prepare for uncertainty propagation analysis [FPD]
err_coeff = {var:{ele:{} for ele in fpdp.index.levels[0]} \
      for var in ['bs','fpd']} # was DD
err_cost  = {var:{ele:{} for ele in fpdp.index.levels[0]} \
      for var in ['bs','fpd']} # was DC
err_cfs_both = {ele:{src:np.zeros(2) for src in fpdp.loc[ele].index} \
            for ele in fpdp.index.levels[0]} # was D_bs_fpd
fpd_sys_std = {ele:{src:np.zeros(1) for src in fpdp.loc[ele].index} \
               for ele in fpdp.index.levels[0]}
fpdbase['dosm25_sys'] = np.nan

tot = pd2vs(fpdbase.m)
mw = np.float_(1)
bs = np.vstack(fpdbase.ele.map(pz.prop.solubility25).values)
ms = tot * mw / (bs - tot)

#%% Run uncertainty propagation analysis [FPD]
ionslist = [np.array(['Na','Cl']), np.array(['K','Cl'])]

for E,ele in enumerate(fpdp.index.levels[0]):
    
    Eions = ionslist[E]

    print('Optimising FPD fit for ' + ele + '...')

    for src in fpdp.loc[ele].index:

        print(' ... ' + src)
        
        SL = np.logical_and(fpdbase.ele == ele,fpdbase.src == src)

        # Optimise for bs
        optemp = optimize.least_squares(
            lambda Dbs: fpdbase.dosm25[SL] - Dbs \
                * pz.sim.dosm25_dbs (bs[SL],ms[SL],mw,
                                     pd2vs(fpdbase.fpd[SL]),
                                     pd2vs(fpdbase.nC[SL]),
                                     pd2vs(fpdbase.nA[SL]),
                                     Eions,
                                     pd2vs(fpdbase.t[SL]),
                                     pd2vs(fpdbase.t25[SL]),
                                     pd2vs(fpdbase.t25[SL]),
                                     cf).ravel(),
            0)

        err_coeff['bs'][ele][src] = optemp['x'][0]
        err_cost ['bs'][ele][src] = optemp['cost']

        # Optimise for FPD
        optemp = optimize.least_squares(
            lambda Dfpd: fpdbase.dosm25[SL] - Dfpd \
                * pz.sim.dosm25_dfpd(bs[SL],ms[SL],mw,
                                     pd2vs(fpdbase.fpd[SL]),
                                     pd2vs(fpdbase.nC[SL]),
                                     pd2vs(fpdbase.nA[SL]),
                                     Eions,
                                     pd2vs(fpdbase.t[SL]),
                                     pd2vs(fpdbase.t25[SL]),
                                     pd2vs(fpdbase.t25[SL]),
                                     cf).ravel(),
            0)
                
        err_coeff['fpd'][ele][src] = optemp['x'][0]
        err_cost ['fpd'][ele][src] = optemp['cost']

        # Select best fit and correct using that [FPD]
        if (err_cost['bs'][ele][src] < err_cost['fpd'][ele][src]) \
          or (fpdp.loc[ele,src]['len']['m'] < 5):
            err_cfs_both[ele][src][0] = err_coeff['bs'][ele][src]
        else:
            err_cfs_both[ele][src][1] = err_coeff['fpd'][ele][src]

        # Get fit residuals [FPD]
        fpdbase.loc[SL,'dosm25_sys'] = fpdbase.dosm25[SL] \
            - err_cfs_both[ele][src][0] \
                * pz.sim.dosm25_dbs (bs[SL],ms[SL],mw,
                                     pd2vs(fpdbase.fpd[SL]),
                                     pd2vs(fpdbase.nC[SL]),
                                     pd2vs(fpdbase.nA[SL]),
                                     Eions,
                                     pd2vs(fpdbase.t[SL]),
                                     pd2vs(fpdbase.t25[SL]),
                                     pd2vs(fpdbase.t25[SL]),
                                     cf).ravel() \
            - err_cfs_both[ele][src][1] \
                * pz.sim.dosm25_dfpd(bs[SL],ms[SL],mw,
                                     pd2vs(fpdbase.fpd[SL]),
                                     pd2vs(fpdbase.nC[SL]),
                                     pd2vs(fpdbase.nA[SL]),
                                     Eions,
                                     pd2vs(fpdbase.t[SL]),
                                     pd2vs(fpdbase.t25[SL]),
                                     pd2vs(fpdbase.t25[SL]),
                                     cf).ravel()

        # Get st. dev. of residuals [FPD]
        fpd_sys_std[ele][src] = np.std(fpdbase.dosm25_sys[SL])

# Pickle outputs for Jupyter Notebook analysis and sign off
with open('pickles/simpytz_fpd.pkl','wb') as f:
    pickle.dump((fpdbase,fpdp,err_cfs_both,fpd_sys_std),f)

print('FPD fit optimisation complete!')
