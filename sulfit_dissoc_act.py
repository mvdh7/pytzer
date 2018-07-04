from autograd import numpy as np
from matplotlib import pyplot as plt
import pickle
import pytzer as pz

with open('pickles/bC_H2SO4_dis.pkl','rb') as f:
    bCs,bCmx = pickle.load(f)
    
# Define new bC functions
def H_HSO4_fit(T):
    
    return np.full_like(T,bCs[0], dtype='float64'), \
           np.full_like(T,bCs[1], dtype='float64'), \
           np.zeros_like(T), \
           np.full_like(T,bCs[2], dtype='float64'), \
           np.full_like(T,bCs[3], dtype='float64'), \
           np.float_(2), -9, np.float_(2.5), np.zeros_like(T, dtype='bool')

def H_SO4_fit(T):
    
    return np.full_like(T,bCs[4], dtype='float64'), \
           np.full_like(T,bCs[5], dtype='float64'), \
           np.zeros_like(T), \
           np.full_like(T,bCs[6], dtype='float64'), \
           np.full_like(T,bCs[7], dtype='float64'), \
           2 - 1842.843 * (1/T - 1/298.15), \
           -9, np.float_(2.5), np.zeros_like(T, dtype='bool')

# Get original cdict
cf     = pz.cdicts.CRP94

# Insert new bC functions into a new cdict
cf_fit = pz.cdicts.cdict()
cf_fit.dh['Aosm'] = pz.coeffs.Aosm_CRP94
cf_fit.bC['H-HSO4'] = H_HSO4_fit 
cf_fit.bC['H-SO4' ] = H_SO4_fit
cf_fit.theta['HSO4-SO4'] = pz.coeffs.HSO4_SO4_CRP94
cf_fit.jfunc = pz.jfuncs.P75_eq47
cf_fit.psi['H-HSO4-SO4'] = pz.coeffs.H_HSO4_SO4_CRP94
cf_fit.K['HSO4'] = pz.coeffs.KHSO4_CRP94

# Calculate activity coefficients with original and new bCs
sqTSO4 = np.vstack(np.linspace(0.01,np.sqrt(6),50))
TSO4 = sqTSO4**2
T = np.full_like(TSO4,298.15).ravel()

## Solve for mH (SLOW)
#mH     = pz.data.dis_sim_H2SO4(TSO4,T,cf    )
#mH_fit = pz.data.dis_sim_H2SO4(TSO4,T,cf_fit)
#with open('pickles/sda_mH.pkl','wb') as f:
#    pickle.dump((mH,mH_fit),f)
with open('pickles/sda_mH.pkl','rb') as f:
    mH,mH_fit = pickle.load(f)

# Calculate other variables
mHSO4 = 2*TSO4 - mH
mSO4  = mH - TSO4
alpha = mSO4 / TSO4

mHSO4_fit = 2*TSO4 - mH_fit
mSO4_fit  = mH_fit - TSO4
alpha_fit = mSO4_fit / TSO4

mols     = np.concatenate((mH    ,mHSO4    ,mSO4    ), axis=1)
mols_fit = np.concatenate((mH_fit,mHSO4_fit,mSO4_fit), axis=1)
ions = np.array(['H','HSO4','SO4'])

acfs     = pz.model.acfs(mols    ,ions,T,cf    )
acfs_fit = pz.model.acfs(mols_fit,ions,T,cf_fit)

# Load real data
with open('pickles/disbase.pkl','rb') as f:
    disbase = pickle.load(f)

# Plot results
fig,ax = plt.subplots(2,2)

ax[0,0].plot(sqTSO4,alpha)
ax[0,0].plot(sqTSO4,alpha_fit, c='r')
ax[0,0].set_ylabel(chr(945))
disbase.plot.scatter('sqm','a_bisulfate', ax=ax[0,0])

ax[1,0].plot(sqTSO4,acfs)
ax[1,0].set_ylim((0,5))

ax[1,0].plot(sqTSO4,acfs_fit, ls='--')
