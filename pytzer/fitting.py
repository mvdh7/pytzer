from autograd import numpy as np
from autograd import elementwise_grad as egrad
from .constants import b, R
from . import coeffs, miami

def fG(T,I,Aosm):
    
    return -4 * Aosm * I * np.log(1 + b*np.sqrt(I)) / b


def Gex_MX(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega):
    
    mC = mCmA[:,0]
    mA = mCmA[:,1]
    
    I = (mC*zC**2 + mA*zA**2) / 2
    Z = mC*np.abs(zC) + mA*np.abs(zA)

    Aosm = coeffs.Aosm_CRP94(T)[0]
    
    B  = b0 + b1 * miami.g(alph1*np.sqrt(I)) + b2 * miami.g(alph2*np.sqrt(I))
    CT = C0 + C1 * miami.h(omega*np.sqrt(I)) * 4
    
    Gex_nRT = fG(T,I,Aosm) + mC*mA * (2*B + Z*CT)
    
    return Gex_nRT


ln_acf = egrad(Gex_MX)


def acf(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega):
    
    return np.exp(ln_acf(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega))


def osmfunc(ww,mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega):
    
    return ww * R * T \
        * Gex_MX(np.array([mCmA[:,0]/ww,mCmA[:,1]/ww]).transpose(),
                 zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)


osmD = egrad(osmfunc)


def osm(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega):
    
    ww = np.full_like(T,1, dtype='float64')
    
    return 1 - osmD(ww,mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega) \
        / (R * T * (np.sum(mCmA,axis=1)))

