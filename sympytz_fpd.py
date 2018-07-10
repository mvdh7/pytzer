# Import libraries
from autograd import numpy as np
import numpy as xnp
import pandas as pd
from scipy import optimize
import pickle
import pytzer as pz

# Load raw datasets
datapath = 'E:/Dropbox/_UEA_MPH/pitzer-spritzer/python/datasets/'
fpdbase = pz.data.fpd(datapath)

# Select data for analysis
fpdbase = fpdbase[fpdbase.smooth == 0]
#fpdbase = fpdbase[xnp.logical_or.reduce((fpdbase.ele == 'NaCl',
#                                         fpdbase.ele == 'KCl'))]#,
##                                         fpdbase.ele == 'CaCl2'))]
fpdbase = fpdbase[fpdbase.ele == 'NaCl']

ions = np.array(['Na','Cl'])

# Prepare model cdict
cf = pz.cdicts.cdict()
cf.bC['Na-Cl'] = pz.coeffs.Na_Cl_A92ii
cf.dh['Aosm']  = pz.coeffs.Aosm_M88
cf.dh['AH']    = pz.coeffs.AH_MPH

# Convert temperatures to 298.15 K
fpdbase['t25'] = np.full_like(fpdbase.t,298.15, dtype='float64')

def pd2np(series):
    return np.vstack(series.values)

#fpdbase['Lapp'] = pz.model.Lapp(pd2np(fpdbase.m),
#                                np.float_(1),#pd2np(fpdbase.nC),
#                                np.float_(1),#pd2np(fpdbase.nA),
#                                ions,
#                                pd2np(fpdbase.t),
#                                cf) / 1000
#
#fpdbase['Lapp25'] = pz.model.Lapp(pd2np(fpdbase.m),
#                                np.float_(1),#pd2np(fpdbase.nC),
#                                np.float_(1),#pd2np(fpdbase.nA),
#                                ions,
#                                pd2np(fpdbase.t25),
#                                cf) / 1000

# Create initial electrolytes pivot table
fpde = pd.pivot_table(fpdbase,
                      values  = ['m'],
                      index   = ['ele'],
                      aggfunc = [np.min,np.max,len])

# Convert measured FPD into osmotic coeff. at 298.15 K
fpdbase['osm25'] = pz.tconv.osm2osm(pd2np(fpdbase.m),
                                    np.float_(1),#fpdbase.nC.values,
                                    np.float_(1),#fpdbase.nA.values,
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

# Run uncertainty propagation analysis [FPD]
for ele in fpdp.index.levels[0]:

    print('Optimising FPD fit for ' + ele + '...')

    bs = pz.prop.solubility25[ele]

    for src in fpdp.loc[ele].index:

        SL = np.logical_and(fpdbase.ele == ele,fpdbase.src == src)

#        optemp = optimize.least_squares(
#            lambda Dbs: fpdbase.dosm25[SL] - Dbs * \
#                pweb.frz.fx_dosm_dbs(fpdbase.fpd[SL],
#                                     fpdbase.m[SL],
#                                     fpdbase.nu[SL],
#                                     bs), 0)
#        DD['bs'][ele][src] = optemp['x'][0]
#        DC['bs'][ele][src] = optemp['cost']
#
#        optemp = optimize.least_squares(
#            lambda Dfpd: fpdbase.dosm25[SL] - Dfpd * \
#                pweb.frz.fx_dosm_dfpd(fpdbase.fpd[SL],
#                                      fpdbase.m[SL],
#                                      fpdbase.nu[SL]), 0)
#        DD['fpd'][ele][src] = optemp['x'][0]
#        DC['fpd'][ele][src] = optemp['cost']
#
#        # Select best fit and correct using that [FPD]
#        if (DC['bs'][ele][src] < DC['fpd'][ele][src]) \
#          or (fpdp.loc[ele,src]['len']['m'] < 5):
#            D_bs_fpd[ele][src][0] = DD['bs'][ele][src]
#        else:
#            D_bs_fpd[ele][src][1] = DD['fpd'][ele][src]
#
#        # Get fit residuals [FPD]
#        fpdbase.loc[SL,'dosm25_sys'] = fpdbase.dosm25[SL] \
#            - D_bs_fpd[ele][src][0] \
#                * pweb.frz.fx_dosm_dbs(
#                    fpdbase.fpd[SL],
#                    fpdbase.m[SL],
#                    fpdbase.nu[SL],
#                    fpdbase.ele[SL].map(pweb.prop.solubility25)) \
#            - D_bs_fpd[ele][src][1] \
#                * pweb.frz.fx_dosm_dfpd(
#                    fpdbase.fpd[SL],
#                    fpdbase.m[SL],
#                    fpdbase.nu[SL])
#
#        # Get st. dev. of residuals [FPD]
#        fpd_sys_std[ele][src] = np.std(fpdbase.dosm25_sys[SL])

## Pickle outputs for Jupyter Notebook analysis and sign off
#with open('pickles/simpar_fpd.pkl','wb') as f:
#    pickle.dump((fpdbase,fpdp,D_bs_fpd,fpd_sys_std),f)
#
#print('FPD fit optimisation complete!')
