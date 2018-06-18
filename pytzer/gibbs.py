import numpy as np
from scipy.misc import derivative
from pytzer import bcao, dh, fx
from pytzer.constants import b

# Define ionic charges
zNa = np.float_(+1)
zK  = np.float_(+1)
zCl = np.float_(-1)

def chargeBalance(tNa=0,tK=0,tCl=0):
    return tNa*zNa + tK*zK == -(tCl*zCl)

def Gex(T,tNa=None,tK=None,tCl=None):
        
#    # Check charge balance
#    assert all(chargeBalance(tNa,tK,tCl)), \
#        'Invalid charge balance - check input total molalities!'
    
    # Get molalities (accounting for incomplete dissociation)
    if tNa is not None: mNa = tNa
    else:               mNa = np.zeros_like(T)
    if tK  is not None: mK  = tK
    else:               mK  = np.zeros_like(T)
    if tCl is not None: mCl = tCl
    else:               mCl = np.zeros_like(T)
    
    # Calculate ionic strength etc.
    I = (mNa*zNa**2 + mK*zK**2 + mCl*zCl**2) * 0.5
    Z = mNa*np.abs(zNa) + mK*np.abs(zK) + mCl*np.abs(zCl)
    
    # Get Pitzer model coefficients
    bC_NaCl,ao_NaCl,_ = bcao.NaCl_A92ii(T)
    
    # Calculate excess Gibbs energy
    Gex = fG(T,I) \
        + mNa*mCl * (2*fx.B(bC_NaCl,ao_NaCl,I) + Z*fx.CT(bC_NaCl,ao_NaCl,I))
    
    return Gex  

def fG(T,I):
    return -4 * I * dh.Aosm_CRP94(T)[0] * np.log(1 + b*np.sqrt(I)) / b

def ln_act(T,tNa=None,tK=None,tCl=None):
    
    if tNa is None: tNa = np.zeros_like(T)
    if tK  is None: tK  = np.zeros_like(T)
    if tCl is None: tCl = np.zeros_like(T)
    
    ln_act = np.full((np.size(T),3),np.nan)
    dx = 1e-8
    
    for i,t in enumerate(T):
        ln_act[i,0] = derivative(lambda tX: \
              Gex(np.array([t]),tNa=tX,tK=tK[i],tCl=tCl[i]), tNa[i], dx=dx)[0]
        ln_act[i,1] = derivative(lambda tX: \
              Gex(np.array([t]),tNa=tNa[i],tK=tX,tCl=tCl[i]), tK[i], dx=dx)[0]
        ln_act[i,2] = derivative(lambda tX: \
              Gex(np.array([t]),tNa=tNa[i],tK=tK[i],tCl=tX), tCl[i], dx=dx)[0]
    return ln_act

T = np.array([298.15,298.15,298.15])
tNa = np.array([1.,2.,3.])
tCl = np.array([1.,2.,3.])
test = Gex(T,tNa=tNa,tCl=tCl)

tln_act = ln_act(T,tNa=tNa,tCl=tCl)
tact = np.exp(tln_act)
