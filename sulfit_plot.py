from autograd import numpy as np
from matplotlib import pyplot as plt
import pickle
import pytzer as pz
from scipy import io as scio

# Load simulated dataset
cf = pz.cdicts.CRP94
with open('pickles/sulfit.pkl','rb') as f:
    T,TSO4,mH = pickle.load(f)
    
# Calculate other variables
mHSO4 = 2*TSO4 - mH
mSO4  = mH - TSO4
alpha = mSO4 / TSO4

## Load real datasets
#datapath = 'E:/Dropbox/_UEA_MPH/pitzer-spritzer/python/datasets/'
#disbase = pz.data.dis(datapath)
#
## Simulate expected speciation (slow)
#disbase['mH_sim'] = pz.data.dis_sim_H2SO4(np.vstack(disbase.TSO4.values),
#                                          disbase.t.values,
#                                          cf)
#disbase['mHSO4_sim'] = 2*disbase.TSO4 - disbase.mH_sim
#disbase['mSO4_sim' ] = disbase.mH_sim - disbase.TSO4
#disbase['a_bisulfate_sim'] = disbase.mSO4_sim / disbase.TSO4
#disbase['Da_bisulfate'] = disbase.a_bisulfate - disbase.a_bisulfate_sim
#
#with open('disbase.pkl','wb') as f:
#    pickle.dump(disbase,f)

# Fast load of results from above saved 2018-07-03
with open('pickles/disbase.pkl','rb') as f:
    disbase = pickle.load(f)

# Exclude bad data
QC_H2SO4 = np.logical_not(np.logical_or( \
    np.logical_and(disbase.src == 'HR57', disbase.sqm < 1),
                   disbase.src == 'MPH56'))
# Include only 298.15 K (ish)
LT = np.logical_and(disbase.t > 293, disbase.t < 303)
# Combine QC and T range
L = np.logical_and(QC_H2SO4,LT)

# Fit for bCs!
bCs,bCmx = pz.fitting.bC_MNX_from_alpha(np.vstack(disbase[L].TSO4.values),
    np.vstack(disbase[L].a_bisulfate.values),disbase[L].t.values)
# Save results for analysis elsewhere
scio.savemat('bC_H2SO4_dis',{'bCs':bCs, 'bCmx':bCmx})
with open('bC_H2SO4_dis.pkl','wb') as f:
    pickle.dump((bCs,bCmx),f)

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

# Insert new bC functions into a new cdict
cf_fit = pz.cdicts.CRP94
cf_fit.bC['H-HSO4'] = H_HSO4_fit 
cf_fit.bC['H-SO4' ] = H_SO4_fit

# Simulate fitted dissociation (SLOW)
disbase['mH_fit'] = pz.data.dis_sim_H2SO4(np.vstack(disbase.TSO4.values),
                                          disbase.t.values,
                                          cf_fit)
disbase['mHSO4_fit'] = 2*disbase.TSO4 - disbase.mH_fit
disbase['mSO4_fit' ] = disbase.mH_fit - disbase.TSO4
disbase['a_bisulfate_fit'] = disbase.mSO4_fit / disbase.TSO4
disbase['Da_bisulfate_fit'] = disbase.a_bisulfate - disbase.a_bisulfate_fit

# Get 'official' bC values for comparison
b0_MX,b1_MX,_,C0_MX,C1_MX,_,_,_,_ = pz.coeffs.H_HSO4_CRP94(T[0])
b0_MY,b1_MY,_,C0_MY,C1_MY,_,_,_,_ = pz.coeffs.H_SO4_CRP94 (T[0])
bCs_true = np.array([b0_MX,b1_MX,C0_MX,C1_MX,b0_MY,b1_MY,C0_MY,C1_MY])

# Plot input dataset
fig,ax = plt.subplots(2,2)

ax[0,0].plot(np.sqrt(TSO4),alpha)
disbase[L].plot.scatter('sqm','a_bisulfate', ax=ax[0,0], alpha=0.6)
disbase[~QC_H2SO4].plot.scatter('sqm','a_bisulfate', ax=ax[0,0], alpha=0.6,
                                c='r')
disbase[~LT      ].plot.scatter('sqm','a_bisulfate', ax=ax[0,0], alpha=0.6,
                                c='purple')
ax[0,0].grid(alpha=0.5)

disbase[L].plot.scatter('sqm','Da_bisulfate', ax=ax[1,0], alpha=0.6)
ax[1,0].grid(alpha=0.5)

disbase[L].plot.scatter('sqm','Da_bisulfate_fit', ax=ax[1,1], alpha=0.5)
ax[1,1].grid(alpha=0.5)

disbase[L].hist('Da_bisulfate', ax=ax[0,1], bins=10)
ax[0,1].grid(alpha=0.5)

