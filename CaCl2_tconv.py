from pytzer import model
from pytzer.coeffs import AH_MPH as AL
from pytzer.coeffs import AC_MPH as AJ
#from pytzer.tconv import y,z,O
from pytzer.constants import R, Mw
from autograd import numpy as np
from autograd import elementwise_grad as egrad

# Enthalpy - CaCl2 direct
def Lapp_CaCl2(tot):
    
    # Coefficients from Fit_L.res from SLC
    b0L = np.float_( 0.607735e-04)
    b1L = np.float_( 0.369990E-02)
    C0L = np.float_(-0.434061E-04)
    C1L = np.float_( 0.862546E-03)
    
    # Set temperature (coeffs only valid at 298.15 K)
    T = np.float_(298.15)
    
    # Pitzer model coefficients
    b     = np.float_(1.2)
    alpha = np.float_(2.0)
    omega = np.float_(2.5)
    
    # Stoichiometry and charges
    nC = np.float_(1)
    nA = np.float_(2)
    
    zC = np.float_(+2)
    zA = np.float_(-1)
    
    # Ionic strength
    I = (nC*zC**2 + nA*zA**2) * tot / 2
    
    # B and C derivatives wrt. temperature
    BL = b0L +     b1L * model.g(alpha * np.sqrt(I))
    CL = C0L + 4 * C1L * model.h(omega * np.sqrt(I))
    
    # Debye-Hueckel function
    DHL = (nC+nA) * np.abs(zC*zA) * (AL(T) / (2*b)) * np.log(1 + b*np.sqrt(I))
    
    # Evaluate apparent molal enthalpy
    return DHL - 2*nC*nA * R * T**2 * (tot * BL + tot**2 * nC*zC * CL)

# Derivatives wrt. molality - tested vs SLC's Calc_L1.res
dLapp_CaCl2_dm = egrad(Lapp_CaCl2)

def L1_CaCl2(tot): # HO58 Ch. 8 Eq. (8-4-9)
    return -Mw * tot**2 * dLapp_CaCl2_dm(tot)

def L2_CaCl2(tot): # HO58 Ch. 8 Eq. (8-4-7)
    return Lapp_CaCl2(tot) + tot * dLapp_CaCl2_dm(tot)

# Heat capacity - CaCl2 direct
def Cpapp_CaCl2(tot,T,bCJ):
    
#    # Coefficients from Fit_Cp.res from SLC
#    b0J = np.float_(-0.113651E-04)
#    b1J = np.float_( 0.883048E-05)
#    C0J = np.float_( 0.148164E-06)
#    C1J = np.float_( 0.479265E-04)
#    
#    # Set temperature (coeffs only valid at 298.15 K)
#    T = np.float_(298.15)
    
    # Unpack coefficients
    b0J = bCJ[0]
    b1J = bCJ[1]
    C0J = bCJ[2]
    C1J = bCJ[3]
    
    # Pitzer model coefficients
    b     = np.float_(1.2)
    alpha = np.float_(2.0)
    omega = np.float_(2.5)
    
    # Stoichiometry and charges
    nC = np.float_(1)
    nA = np.float_(2)
    
    zC = np.float_(+2)
    zA = np.float_(-1)
    
    # Ionic strength
    I = (nC*zC**2 + nA*zA**2) * tot / 2
    
    # B and C second derivatives wrt. temperature
    BJ = b0J +     b1J * model.g(alpha * np.sqrt(I))
    CJ = C0J + 4 * C1J * model.h(omega * np.sqrt(I))
    
    # Debye-Hueckel function
    DHJ = (nC+nA) * np.abs(zC*zA) * (AJ(T)[0] / (2*b)) \
        * np.log(1 + b*np.sqrt(I))
    
    # Evaluate apparent molal enthalpy
    return DHJ - 2*nC*nA * R * T**2 * (tot * BJ + tot**2 * nC*zC * CJ)

# Derivatives wrt. molality - tested vs SLC's Calc_J1.res
dCpapp_CaCl2_dm = egrad(Cpapp_CaCl2)

def J1_CaCl2(tot,T,bCJ): # HO58 Ch. 8 Eq. (8-4-9)
    return -Mw * tot**2 * dCpapp_CaCl2_dm(tot,T,bCJ)

def J2_CaCl2(tot,T,bCJ): # HO58 Ch. 8 Eq. (8-4-7)
    return tot * dCpapp_CaCl2_dm(tot,T,bCJ)

bCJdict = {'25': np.float_([-0.113651E-04,
                             0.883048E-05,
                             0.148164E-06,
                             0.479265E-04])}

bCJ = bCJdict['25']
T = np.float_(298.15)

tot   = np.float_([[6.8]])
Lapp  = Lapp_CaCl2 (tot)
L1    = L1_CaCl2   (tot)
Cpapp = Cpapp_CaCl2(tot,T,bCJ)
J1    = J1_CaCl2   (tot,T,bCJ)

## Osmotic coefficient - CaCl2 direct
#def osm2osm25_CaCl2(tot,T0,osm_T0):
#    
#    tot = np.vstack(tot)
#    T0  = np.vstack(T0)
#    T1  = np.full_like(T0,298.15)
#    TR  = np.full_like(T0,298.15)
#    
#    nC = np.float_(1)
#    nA = np.float_(2)
#    
#    lnAW_T0 = -osm_T0 * tot * (nC + nA) * Mw
#    
#    lnAW_T1 = lnAW_T0 - y(T0,T1) * L1(tot,n1,n2,ions,TR,cf) \
#                      + z(T0,T1) * J1(tot,n1,n2,ions,TR,cf) \
#                      - O(T0,T1) * G1(tot,n1,n2,ions,TR,cf)
#
#    return -lnAW_T1 / (tot * (nC + nA) * Mw)
