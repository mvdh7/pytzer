from autograd import numpy as np
from matplotlib import pyplot as plt
import pickle
import pytzer as pz

# Load simulated dataset
cf = pz.cdicts.CRP94
with open('sulfit.pkl','rb') as f:
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
with open('disbase.pkl','rb') as f:
    disbase = pickle.load(f)

# Plot results
fig,ax = plt.subplots(2,2)

ax[0,0].plot(np.sqrt(TSO4),alpha)
disbase.plot.scatter('sqm','a_bisulfate', ax=ax[0,0], alpha=0.6)
ax[0,0].grid(alpha=0.5)

disbase.plot.scatter('sqm','Da_bisulfate', ax=ax[1,0], alpha=0.6)
ax[1,0].grid(alpha=0.5)

disbase.hist('Da_bisulfate', ax=ax[1,1], bins=20)
ax[1,1].grid(alpha=0.5)
