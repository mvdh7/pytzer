from autograd import numpy as np
import pytzer as pz
import pickle

# Import coefficients dict
cf = pz.cdicts.CRP94

# Load simulated dataset
with open('sulfit.pkl','rb') as f:
    T,TSO4,mH = pickle.load(f)
    
# Calculate parameters to be fitted
mHSO4 = 2*TSO4 - mH
mSO4  = mH - TSO4

alpha = mSO4 / TSO4

mols = np.concatenate((mH,mHSO4,mSO4), axis=1)
ions = np.array(['H', 'HSO4', 'SO4'])

osm = pz.model.osm(mols,ions,T,cf)
osmST = osm * (mH + mHSO4 + mSO4).ravel() / (3 * TSO4.ravel())

acfs = pz.model.acfs(mols,ions,T,cf)
acfPM = np.cbrt(acfs[:,0]**2 * acfs[:,2] * mH.ravel()**2 * mSO4.ravel() \
    / (4 * TSO4.ravel()**3))
    
# Fit parameters!

