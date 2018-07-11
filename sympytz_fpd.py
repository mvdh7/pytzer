# Import libraries
from autograd import numpy as np
from autograd import elementwise_grad as egrad
import numpy as xnp
import pandas as pd
from scipy import optimize
import pickle
import pytzer as pz

# Load raw datasets
datapath = 'datasets/'
fpdbase = pz.data.fpd(datapath)

# Select data for analysis
fpdbase = fpdbase[fpdbase.smooth == 0]
fpdbase = fpdbase[xnp.logical_or.reduce((fpdbase.ele == 'NaCl',
                                         fpdbase.ele == 'KCl'))]#,
#                                         fpdbase.ele == 'CaCl2'))]
#fpdbase = fpdbase[fpdbase.ele == 'NaCl']

ions = np.array(['Na','K','Cl'])

# Prepare model cdict
cf = pz.cdicts.cdict()
cf.bC['Na-Cl'] = pz.coeffs.Na_Cl_A92ii
cf.bC['K-Cl' ] = pz.coeffs.K_Cl_GM89
cf.dh['Aosm']  = pz.coeffs.Aosm_M88
cf.dh['AH']    = pz.coeffs.AH_MPH

# Convert temperatures to 298.15 K
fpdbase['t25'] = np.full_like(fpdbase.t,298.15, dtype='float64')

def pd2np(series):
    return np.vstack(series.values)

# Create initial electrolytes pivot table
fpde = pd.pivot_table(fpdbase,
                      values  = ['m'],
                      index   = ['ele'],
                      aggfunc = [np.min,np.max,len])

# Convert measured FPD into osmotic coeff. at 298.15 K
fpdbase['osm25'] = np.nan
fpdbase['osm25'] = pz.tconv.osm2osm(pd2np(fpdbase.m),
                                    pd2np(fpdbase.nC),
                                    pd2np(fpdbase.nA),
                                    ions,
                                    pd2np(fpdbase.t),
                                    pd2np(fpdbase.t25),
                                    pd2np(fpdbase.t25),
                                    cf,
                                    pd2np(fpdbase.osm))

# Calculate model osmotic coefficient at 298.15 K
mols = np.vstack((fpdbase.m,fpdbase.m)).transpose()
fpdbase['osm25_calc'] = pz.model.osm(mols,ions,pd2np(fpdbase.t25),cf)
fpdbase['dosm25'] = fpdbase.osm25 - fpdbase.osm25_calc

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

# Propagate bs & fpd uncertainties
def fpd2osm25(bs,ms,mw,fpd,nC,nA,ions,T0,T1,TR,cf):
    
    tot = bs * ms / (ms + mw)
    
    mols = np.vstack([tot.ravel() * nC,
                      tot.ravel() * nA]).transpose()
    
    osmT0 = pz.tconv.fpd2osm(mols,fpd)
    osm25 = pz.tconv.osm2osm(tot,nC,nA,ions,T0,T1,TR,cf,osmT0)
    
    return osm25

dosm25_dbs  = egrad(fpd2osm25)
dosm25_dfpd = egrad(fpd2osm25, argnum=3)

tot = pd2np(fpdbase.m)
mw = np.float_(1)
bs = np.vstack(fpdbase.ele.map(pz.prop.solubility25).values)
ms = tot * mw / (bs - tot)

## Check propagation equation works
#bs = np.full_like(tot,pz.prop.solubility25['NaCl'])
#
#fpdbase['osm25_test'] = fpd2osm25   (bs,ms,mw,
#                                     pd2np(fpdbase.fpd),
#                                     pd2np(fpdbase.nC),
#                                     pd2np(fpdbase.nA),
#                                     ions,
#                                     pd2np(fpdbase.t),
#                                     pd2np(fpdbase.t25),
#                                     pd2np(fpdbase.t25),
#                                     cf)
#
#fpdbase['dosm25_dbs'] = dosm25_dbs  (bs,ms,mw,
#                                     pd2np(fpdbase.fpd),
#                                     pd2np(fpdbase.nC),
#                                     pd2np(fpdbase.nA),
#                                     ions,
#                                     pd2np(fpdbase.t),
#                                     pd2np(fpdbase.t25),
#                                     pd2np(fpdbase.t25),
#                                     cf)
#
#fpdbase['dosm25_dfpd'] = dosm25_dfpd(bs,ms,mw,
#                                     pd2np(fpdbase.fpd),
#                                     pd2np(fpdbase.nC),
#                                     pd2np(fpdbase.nA),
#                                     ions,
#                                     pd2np(fpdbase.t),
#                                     pd2np(fpdbase.t25),
#                                     pd2np(fpdbase.t25),
#                                     cf)

## Plot components
#from matplotlib import pyplot as plt
#fig,ax = plt.subplots(1,1)

#%% Run uncertainty propagation analysis [FPD]
ionslist = [np.array(['Na','Cl']), np.array(['K','Cl'])]

for E,ele in enumerate(fpdp.index.levels[0]):
    
    ions = ionslist[E]

    print('Optimising FPD fit for ' + ele + '...')

    for src in fpdp.loc[ele].index:

        print(' ... ' + src)
        
        SL = np.logical_and(fpdbase.ele == ele,fpdbase.src == src)

        # Optimise for bs
        optemp = optimize.least_squares(
            lambda Dbs: fpdbase.dosm25[SL] - Dbs \
                * dosm25_dbs (bs[SL],ms[SL],mw,
                              pd2np(fpdbase.fpd[SL]),
                              pd2np(fpdbase.nC),
                              pd2np(fpdbase.nA),
                              ions,
                              pd2np(fpdbase.t[SL]),
                              pd2np(fpdbase.t25[SL]),
                              pd2np(fpdbase.t25[SL]),
                              cf).ravel(),
            0)

        err_coeff['bs'][ele][src] = optemp['x'][0]
        err_cost ['bs'][ele][src] = optemp['cost']

        # Optimise for FPD
        optemp = optimize.least_squares(
            lambda Dfpd: fpdbase.dosm25[SL] - Dfpd \
                * dosm25_dfpd(bs[SL],ms[SL],mw,
                              pd2np(fpdbase.fpd[SL]),
                              pd2np(fpdbase.nC),
                              pd2np(fpdbase.nA),
                              ions,
                              pd2np(fpdbase.t[SL]),
                              pd2np(fpdbase.t25[SL]),
                              pd2np(fpdbase.t25[SL]),
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
                * dosm25_dbs (bs[SL],ms[SL],mw,
                              pd2np(fpdbase.fpd[SL]),
                              pd2np(fpdbase.nC),
                              pd2np(fpdbase.nA),
                              ions,
                              pd2np(fpdbase.t[SL]),
                              pd2np(fpdbase.t25[SL]),
                              pd2np(fpdbase.t25[SL]),
                              cf).ravel() \
            - err_cfs_both[ele][src][1] \
                * dosm25_dfpd(bs[SL],ms[SL],mw,
                              pd2np(fpdbase.fpd[SL]),
                              pd2np(fpdbase.nC),
                              pd2np(fpdbase.nA),
                              ions,
                              pd2np(fpdbase.t[SL]),
                              pd2np(fpdbase.t25[SL]),
                              pd2np(fpdbase.t25[SL]),
                              cf).ravel()

        # Get st. dev. of residuals [FPD]
        fpd_sys_std[ele][src] = np.std(fpdbase.dosm25_sys[SL])

# Pickle outputs for Jupyter Notebook analysis and sign off
with open('pickles/simpytz_fpd.pkl','wb') as f:
    pickle.dump((fpdbase,fpdp,err_cfs_both,fpd_sys_std),f)

print('FPD fit optimisation complete!')
