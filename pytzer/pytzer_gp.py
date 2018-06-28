from autograd import numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian as jac
import pytzer as pz
import pickle

mC = np.float_(1.2)
mA = np.float_(1.2)

zC = np.float_(+1)
zA = np.float_(-1)

T = np.float_(298.15)

def fx_I(mC,zC,mA,zA):
    return (mC*zC**2 + mA*zA**2) / 2
def fx_Z(mC,zC,mA,zA):
    return mC*np.abs(zC) + mA*np.abs(zA)

b0,b1,_,C0,_,a1,_,_,_ = pz.coeffs.Na_Cl_M88(T)

I = fx_I(mC,zC,mA,zA)
Z = fx_Z(mC,zC,mA,zA)

bC_mults  = np.float_([np.full_like(I,2, dtype='float64'),
                       2*pz.model.g(a1*np.sqrt(I)),
                       Z]).transpose()
bC_coeffs = np.vstack([b0, b1, C0])

def fx_BC(b0,b1,C0,a1,I,Z):
    return 2 * (b0 + pz.model.g(a1*np.sqrt(I)) * b1) + Z * C0

BCx = fx_BC(b0,b1,C0,a1,I,Z)
BCm = (bC_mults @ bC_coeffs).ravel()

with open('testbCmx.pkl','rb') as f:
    bCmx = pickle.load(f) * 1e-5 # *1e-5 is to scale uncertainty from this to
                                 #  match uncertainty from molality magnitude

# Run Monte-Carlo simulation - bC uncertainty only
Ureps = int(1e3)
BCu = np.full(Ureps,np.nan)

for i in range(Ureps):
    ibC = np.random.multivariate_normal(bC_coeffs.ravel(),bCmx)
    
    BCu[i] = fx_BC(ibC[0],ibC[1],ibC[2],a1,I,Z)
    
BCu_vr = np.var(BCu)

# Directly calculate variance - bC uncertainty only - SPOT ON!
BCd_vr = bC_mults @ bCmx @ bC_mults.transpose()

# Now do molality uncertainty only - Monte-Carlo
mCu = np.random.normal(loc=1.2, scale=0.001, size=Ureps)   
mAu = mCu

Iu = fx_I(mCu,zC,mAu,zA)
Zu = fx_Z(mCu,zC,mAu,zA)

BCumol = fx_BC(b0,b1,C0,a1,Iu,Zu)
BCumol_vr = np.var(BCumol)

# ... and directly - spot on!
fx_dBC_dm = egrad(lambda m: fx_BC(b0,b1,C0,a1,fx_I(m,zC,m,zA),fx_Z(m,zC,m,zA)))
BCdmol_var = fx_dBC_dm(np.float_(1.2))**2 * 0.001**2

# Both together - Monte Carlo
BCU = np.full(Ureps,np.nan)

for i in range(Ureps):
    ibC = np.random.multivariate_normal(bC_coeffs.ravel(),bCmx)
    im  = np.random.normal(loc=1.2, scale=0.001)
    
    BCU[i] = fx_BC(ibC[0],ibC[1],ibC[2],a1,fx_I(im,zC,im,zA),fx_Z(im,zC,im,zA))

BCU_vr = np.var(BCU)

# Both together - direct?
BCD_vr_fail0 = np.sqrt(BCdmol_var**2 + BCd_vr**2) # nope

# Get Jacobian
J_BC = jac(lambda i: fx_BC(i[0],i[1],i[2],a1,
                           fx_I(i[3],zC,i[3],zA),fx_Z(i[3],zC,i[3],zA)))
testJ = J_BC(np.array([b0,b1,C0,1.2]))

# Add covariance between molality and bCs (i.e. zero) to the matrix
bCm_mx = np.zeros((4,4))
bCm_mx[:3,:3] = bCmx
bCm_mx[3,3] = 0.001**2

# Calculate variance - YES!!!
BCD_vr = testJ @ bCm_mx @ testJ.transpose()
