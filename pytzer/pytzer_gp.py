from autograd import numpy as np
import pytzer as pz
import pickle

mC = np.float_(1.2)
mA = np.float_(1.2)

zC = np.float_(+1)
zA = np.float_(-1)

T = np.float_(298.15)

I = (mC*zC**2 + mA*zA**2) / 2
Z = mC*np.abs(zC) + mA*np.abs(zA)

b0,b1,_,C0,_,a1,_,_,_ = pz.coeffs.Na_Cl_M88(T)

bC_mults  = np.float_([np.full_like(I,2, dtype='float64'),
                       2*pz.model.g(a1*np.sqrt(I)),
                       Z]).transpose()
bC_coeffs = np.vstack([b0, b1, C0])

def fx_BC(b0,b1,C0,a1,I,Z):
    return 2 * (b0 + pz.model.g(a1*np.sqrt(I)) * b1) + Z * C0

BCx = fx_BC(b0,b1,C0,a1,I,Z)
BCm = (bC_mults @ bC_coeffs).ravel()

with open('testbCmx.pkl','rb') as f:
    bCmx = pickle.load(f)

# Run Monte-Carlo simulation - bC uncertainty only
Ureps = int(1e3)
BCu = np.full(Ureps,np.nan)

for i in range(Ureps):
    ibC = np.random.multivariate_normal(bC_coeffs.ravel(),bCmx)
    
    BCu[i] = fx_BC(ibC[0],ibC[1],ibC[2],a1,I,Z)
    
BCu_vr = np.var(BCu)

# Directly calculate variance - bC uncertainty only - SPOT ON!
BCm_vr = bC_mults @ bCmx @ bC_mults.transpose()
