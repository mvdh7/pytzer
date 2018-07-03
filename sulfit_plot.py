from autograd import numpy as np
from matplotlib import pyplot as plt
import pickle
import pytzer as pz

# Load simulated dataset
with open('sulfit.pkl','rb') as f:
    T,TSO4,mH = pickle.load(f)
    
# Calculate parameters to be fitted
mHSO4 = 2*TSO4 - mH
mSO4  = mH - TSO4

alpha = mSO4 / TSO4

# Load real datasets
datapath = 'E:/Dropbox/_UEA_MPH/pitzer-spritzer/python/datasets/'

disbase = pz.data.dis(datapath)

# Plot results
fig,ax = plt.subplots(1,1)

ax.plot(np.sqrt(TSO4),alpha)
disbase.plot.scatter('sqm','a_bisulfate', ax=ax)

ax.grid(alpha=0.5)
