from autograd import numpy as np
from pytzer import model, coeffs
from pytzer.constants import b

# Define f function
def fG(T,I,Aosm):
    
    return -4 * Aosm * I * np.log(1 + b*np.sqrt(I)) / b

# Excess Gibbs energy - single electrolyte
def Gex_MX(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega):
    
    mC = np.vstack(mCmA[:,0])
    mA = np.vstack(mCmA[:,1])
    
    I = (mC*zC**2 + mA*zA**2) / 2
    Z = mC*np.abs(zC) + mA*np.abs(zA)

    Aosm = coeffs.Aosm_MPH(T)[0]
    
    B  = b0 + b1 * model.g(alph1*np.sqrt(I)) + b2 * model.g(alph2*np.sqrt(I))
    CT = C0 + C1 * model.h(omega*np.sqrt(I)) * 4
    
    Gex_nRT = fG(T,I,Aosm) + mC*mA * (2*B + Z*CT)
    
    return Gex_nRT

test = Gex_MX(mols,zC,zA,T1,b0[0],b1[0],0,C0[0],C1[0],alph1,-9,omega)
