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

# 'Official' values
b0_MX,b1_MX,_,C0_MX,C1_MX,_,_,_,_ = pz.coeffs.H_HSO4_CRP94(T[0])
b0_MY,b1_MY,_,C0_MY,C1_MY,_,_,_,_ = pz.coeffs.H_SO4_CRP94 (T[0])

test0 = pz.fitting.bC_MNX(TSO4,alpha,T,b0_MX,b1_MX,C0_MX,C1_MX,
                                       b0_MY,b1_MY,C0_MY,C1_MY)

tfit = pz.fitting.bC_MNX_from_alpha(TSO4,alpha,T)
bCs = tfit['x']

# Now, fitting given mean activity coeff?

# Do the results make sense?
zM = np.float_(+1)
zX = np.float_(-1)
zY = np.float_(-2)

alph1_MX = np.float_(2)
alph1_MY = np.float_(2)

b2_MX = 0
b2_MY = 0

alph2_MX = -9
alph2_MY = -9

omega_MX = np.float_(2.5)
omega_MY = np.float_(2.5)

acfs_fiteq = np.exp(pz.fitting.ln_acfs_MXY(mols,zM,zX,zY,T,
            b0_MX,b1_MX,b2_MX,C0_MX,C1_MX,alph1_MX,alph2_MX,omega_MX,
            b0_MY,b1_MY,b2_MY,C0_MY,C1_MY,alph1_MY,alph2_MY,omega_MY))

acfs_fitted = np.exp(pz.fitting.ln_acfs_MXY(mols,zM,zX,zY,T,
            bCs[0],bCs[1],0,bCs[2],bCs[3],alph1_MX,alph2_MX,omega_MX,
            bCs[4],bCs[5],0,bCs[6],bCs[7],alph1_MY,alph2_MY,omega_MY))

testalpha  = alpha[0] * (1 + alpha[0]) / (1 - alpha[0])
testalpha2 = (pz.coeffs.KHSO4_CRP94(T[0])[0] * acfs[0,0] \
    / (acfs[0,1] * acfs[0,2])) / TSO4[0]

# Define mean activity fitting function - NOWHERE NEAR COMPLETE
def bC_MNX2(TSO4,alpha,b0_MX,b1_MX,C0_MX,C1_MX,b0_MY,b1_MY,C0_MY,C1_MY):
    
    mSO4  = TSO4 * alpha
    mH    = TSO4 + mSO4
    mHSO4 = 2*TSO4 - mH
    
    mols = np.concatenate((mH,mHSO4,mSO4), axis=1)
    
    zM = np.float_(+1)
    zX = np.float_(-1)
    zY = np.float_(-2)
    
    alph1_MX = np.float_(2)
    alph1_MY = np.float_(2)
    
    b2_MX = 0
    b2_MY = 0
    
    alph2_MX = -9
    alph2_MY = -9
    
    omega_MX = np.float_(2.5)
    omega_MY = np.float_(2.5)
    
    ln_acfs = pz.fitting.ln_acfs_MXY(mols,zM,zX,zY,T,
        b0_MX,b1_MX,b2_MX,C0_MX,C1_MX,alph1_MX,alph2_MX,omega_MX,
        b0_MY,b1_MY,b2_MY,C0_MY,C1_MY,alph1_MY,alph2_MY,omega_MY)
    
    gH    = np.exp(ln_acfs[:,0])
    gHSO4 = np.exp(ln_acfs[:,1])
    gSO4  = np.exp(ln_acfs[:,2])
    
    KHSO4 = pz.coeffs.KHSO4_CRP94(T)[0]
    
    return mH.ravel() * TSO4.ravel() / (KHSO4 * gHSO4 / (gH * gSO4) \
                    + mH.ravel()) - mHSO4.ravel()
