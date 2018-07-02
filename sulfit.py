from autograd import numpy as np
from autograd import elementwise_grad as egrad
from scipy import optimize
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
    

# Unsymmetric mixing functions
def xij(T,I,z0,z1):
    
    return 6 * z0*z1 * pz.coeffs.Aosm_CRP94(T)[0] * np.sqrt(I)

def etheta(T,I,z0,z1):
    
    x00 = xij(T,I,z0,z0)
    x01 = xij(T,I,z0,z1)
    x11 = xij(T,I,z1,z1)
    
    jfunc = pz.jfuncs.P75_eq47
    
    etheta = z0*z1 * (jfunc(x01)[0] \
                      - 0.5 * (jfunc(x00)[0] + jfunc(x11)[0])) / (4 * I)
    
    return etheta

# Excess Gibbs energy - mixed electrolyte (e.g. H2SO4)
def Gex_MXY(mols,zM,zX,zY,T,
            b0_MX,b1_MX,b2_MX,C0_MX,C1_MX,alph1_MX,alph2_MX,omega_MX,
            b0_MY,b1_MY,b2_MY,C0_MY,C1_MY,alph1_MY,alph2_MY,omega_MY):
    
    mM = mols[:,0]
    mX = mols[:,1]
    mY = mols[:,2]
    
    I = (mM*zM**2 + mX*zX**2 + mY*zY**2) / 2
    Z = mM*np.abs(zM) + mX*np.abs(zX) + mY*np.abs(zY)

    Aosm = pz.coeffs.Aosm_CRP94(T)[0]
    
    B_MX  = b0_MX + b1_MX * pz.model.g(alph1_MX*np.sqrt(I)) \
                  + b2_MX * pz.model.g(alph2_MX*np.sqrt(I))
    CT_MX = C0_MX + C1_MX * pz.model.h(omega_MX*np.sqrt(I)) * 4
    
    B_MY  = b0_MY + b1_MY * pz.model.g(alph1_MY*np.sqrt(I)) \
                  + b2_MY * pz.model.g(alph2_MY*np.sqrt(I))
    CT_MY = C0_MY + C1_MY * pz.model.h(omega_MY*np.sqrt(I)) * 4
        
    Gex_nRT = pz.fitting.fG(T,I,Aosm) \
        + mM*mX * (2*B_MX + Z*CT_MX)  \
        + mM*mY * (2*B_MY + Z*CT_MY)  \
        + mX*mY * 2*etheta(T,I,zX,zY)
    
    return Gex_nRT

# Corresponding activity coefficients
ln_acfs_MXY = egrad(Gex_MXY)


# Define fitting function
def mfunc(TSO4,alpha,b0_MX,b1_MX,C0_MX,C1_MX,b0_MY,b1_MY,C0_MY,C1_MY):
    
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
    
    ln_acfs = ln_acfs_MXY(mols,zM,zX,zY,T,
        b0_MX,b1_MX,b2_MX,C0_MX,C1_MX,alph1_MX,alph2_MX,omega_MX,
        b0_MY,b1_MY,b2_MY,C0_MY,C1_MY,alph1_MY,alph2_MY,omega_MY)
    
    gH    = np.exp(ln_acfs[:,0])
    gHSO4 = np.exp(ln_acfs[:,1])
    gSO4  = np.exp(ln_acfs[:,2])
    
    KHSO4 = pz.coeffs.KHSO4_CRP94(T)[0]
    
    return mH.ravel() * TSO4.ravel() / (KHSO4 * gHSO4 / (gH * gSO4) \
                    + mH.ravel()) - mHSO4.ravel()

# 'Official' values
b0_MX,b1_MX,_,C0_MX,C1_MX,_,_,_,_ = pz.coeffs.H_HSO4_CRP94(T[0])
b0_MY,b1_MY,_,C0_MY,C1_MY,_,_,_,_ = pz.coeffs.H_SO4_CRP94 (T[0])

test0 = mfunc(TSO4,alpha,b0_MX,b1_MX,C0_MX,C1_MX,b0_MY,b1_MY,C0_MY,C1_MY)

tfit = optimize.least_squares(lambda bCs: mfunc(TSO4,alpha,
    bCs[0],bCs[1],bCs[2],bCs[3],bCs[4],bCs[5],bCs[6],bCs[7]),
    np.zeros(8))

bCs = tfit['x']

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

acfs_fiteq = np.exp(ln_acfs_MXY(mols,zM,zX,zY,T,
            b0_MX,b1_MX,b2_MX,C0_MX,C1_MX,alph1_MX,alph2_MX,omega_MX,
            b0_MY,b1_MY,b2_MY,C0_MY,C1_MY,alph1_MY,alph2_MY,omega_MY))

acfs_fitted = np.exp(ln_acfs_MXY(mols,zM,zX,zY,T,
            bCs[0],bCs[1],0,bCs[2],bCs[3],alph1_MX,alph2_MX,omega_MX,
            bCs[4],bCs[5],0,bCs[6],bCs[7],alph1_MY,alph2_MY,omega_MY))
